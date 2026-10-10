"""Score intake extractors on two test sets: python -m intake.evaluate [--llm]

An extractor is `text -> raw JSON string` (see intake/extraction.py). Sets:
- corpus: the corpus test split. Template text, same phrase bank as train, so rules or models
  built from train look better here than they are.
- challenge: 48 hand-written items (intake/fixtures/challenge.jsonl), 12 per language, real Delhi
  places, wording outside the phrase bank. Frozen before any rules were written. The honest set.

Speech: `evaluate_audio` scores a transcriber with WER/CER on clips named `<challenge id>.<ext>`
in data/intake/audio/. The reference transcript is that challenge item's text.

Translation (Step 4.2): the 36 non-English challenge items have a reference English translation
and a `keep` list of names and numbers (intake/fixtures/challenge_en.jsonl), written by Claude
before any translation model was run. Scores: corpus chrF (character n-gram F-score, sacrebleu)
and the share of `keep` items found in the translation.

Intake API (Step 4.4, --api): RawReport -> Report end to end on the challenge set, against
the contracts stub that intake served before v1.
"""

import json
import unicodedata
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, SEED
from contracts.api import Cues, IssueType, load_corpus
from intake.extraction import parse

Extractor = Callable[[str], str]
Transcriber = Callable[[Path], str]
Detector = Callable[[str], str]  # text -> language code
Translator = Callable[[str, str], str]  # (text, language) -> English text

CUE_NAMES = ("children", "accident", "flooding")
LANGUAGES = ("hi", "mixed", "en", "pa")
CHALLENGE_PATH = Path(__file__).parent / "fixtures" / "challenge.jsonl"
REFERENCES_PATH = Path(__file__).parent / "fixtures" / "challenge_en.jsonl"
AUDIO_DIR = DATA_DIR / "intake" / "audio"
AUDIO_EXTS = (".wav", ".m4a", ".mp3", ".ogg", ".flac")
# The clips the user records (user decision, 4.1): 3 per language, spread over issue types and
# cues. Any challenge id works as a clip; these are the ones asked for.
AUDIO_IDS = ("ch-en-02", "ch-en-05", "ch-en-08", "ch-hi-01", "ch-hi-03", "ch-hi-11",
             "ch-pa-01", "ch-pa-04", "ch-pa-07", "ch-mx-02", "ch-mx-04", "ch-mx-06")
GOLD_COLUMNS = ["id", "language", "text", "issue_type", "location_text", *CUE_NAMES]


# ---------- test sets ----------

def load_challenge(path: Path = CHALLENGE_PATH) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows)[GOLD_COLUMNS]


def load_corpus_test() -> pd.DataFrame:
    """Corpus test split in the gold shape. Spam is kept: truth is 'other', no location."""
    df = load_corpus("test").rename(columns={"complaint_id": "id"})
    df["location_text"] = df["location_text"].fillna("")
    return df[GOLD_COLUMNS].reset_index(drop=True)


def load_references(path: Path = REFERENCES_PATH) -> pd.DataFrame:
    """The non-English challenge items with columns `english` (reference) and `keep`."""
    refs = pd.read_json(path, lines=True, dtype={"id": str})
    return load_challenge().merge(refs, on="id", validate="one_to_one")


# ---------- text helpers ----------

def normalize(s: str) -> str:
    """NFC, lowercase, punctuation (incl. the Devanagari danda) to spaces, single spaces."""
    s = unicodedata.normalize("NFC", s or "").lower()
    s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)
    return " ".join(s.split())


def token_f1(gold: str, pred: str) -> float:
    """Bag-of-words F1 between two spans. Both empty counts as a perfect match."""
    g, p = Counter(normalize(gold).split()), Counter(normalize(pred).split())
    if not g or not p:
        return float(not g and not p)
    overlap = sum((g & p).values())
    return 0.0 if overlap == 0 else 2 * overlap / (sum(g.values()) + sum(p.values()))


def edit_distance(a: list | str, b: list | str) -> int:
    """Levenshtein distance between two sequences (words or characters)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _words(s: str) -> list[str]:
    return normalize(s).split()


def _chars(s: str) -> str:
    return normalize(s).replace(" ", "")


def wer(ref: str, hyp: str) -> float:
    return edit_distance(_words(ref), _words(hyp)) / max(len(_words(ref)), 1)


def cer(ref: str, hyp: str) -> float:
    """Character error rate, spaces excluded: fairer than WER for spelling variants."""
    return edit_distance(_chars(ref), _chars(hyp)) / max(len(_chars(ref)), 1)


# ---------- scoring ----------

def predict(extract: Extractor, gold: pd.DataFrame) -> pd.DataFrame:
    """Run the extractor on every text. Invalid output: valid=False and every field wrong."""
    rows = []
    for text in gold["text"]:
        ex = parse(extract(text))
        rows.append({"valid": ex is not None,
                     "language": ex.language if ex else None,
                     "issue_type": ex.issue_type.value if ex else None,
                     "location_text": ex.location_text if ex else "",
                     **{c: bool(ex and getattr(ex.cues, c)) for c in CUE_NAMES}})
    return pd.DataFrame(rows, index=gold.index)


def _binary_f1(gold: pd.Series, pred: pd.Series) -> float:
    tp = int((gold & pred).sum())
    denominator = 2 * tp + int((~gold & pred).sum()) + int((gold & ~pred).sum())
    return 2 * tp / denominator if denominator else np.nan  # nan: no positives either side


def _macro_f1(gold: pd.Series, pred: pd.Series) -> float:
    """Macro-F1 over the issue types present in gold (an invalid prediction is just wrong)."""
    return float(np.mean([_binary_f1(gold == label, pred == label) for label in gold.unique()]))


def score(gold: pd.DataFrame, pred: pd.DataFrame) -> dict:
    # Invalid output earns nothing, not even for an empty gold location.
    locs = list(zip(gold.location_text, pred.location_text, pred.valid, strict=True))
    loc_f1 = [token_f1(g, p) if ok else 0.0 for g, p, ok in locs]
    loc_exact = [ok and normalize(g) == normalize(p) for g, p, ok in locs]
    cue_f1 = {c: _binary_f1(gold[c].astype(bool), pred[c].astype(bool)) for c in CUE_NAMES}
    return {"n": len(gold),
            "json_valid": pred.valid.mean(),
            "lang_acc": (gold.language == pred.language).mean(),
            "issue_acc": (gold.issue_type == pred.issue_type).mean(),
            "issue_f1": _macro_f1(gold.issue_type, pred.issue_type),
            "loc_exact": float(np.mean(loc_exact)),
            "loc_f1": float(np.mean(loc_f1)),
            **{f"{c}_f1": f for c, f in cue_f1.items()}}


def evaluate(extract: Extractor, gold: pd.DataFrame) -> pd.DataFrame:
    """One row per language plus 'all'."""
    pred = predict(extract, gold)
    rows = {"all": score(gold, pred)}
    for lang in LANGUAGES:
        mask = gold.language == lang
        if mask.any():
            rows[lang] = score(gold[mask], pred[mask])
    return pd.DataFrame(rows).T.astype({"n": int})


def errors(extract: Extractor, gold: pd.DataFrame, field: str = "issue_type") -> pd.DataFrame:
    """Rows where `field` is wrong, gold next to prediction: for reading, not for scoring."""
    pred = predict(extract, gold)
    wrong = gold[field] != pred[field]
    return pd.DataFrame({"text": gold.text, "gold": gold[field], "pred": pred[field]})[wrong]


# ---------- speech ----------

def audio_clips(audio_dir: Path = AUDIO_DIR) -> pd.DataFrame:
    """Clips on disk joined to their challenge item (reference transcript and labels)."""
    gold = load_challenge().set_index("id")
    paths = {p.stem: p for p in sorted(Path(audio_dir).glob("*")) if p.suffix in AUDIO_EXTS}
    unknown = set(paths) - set(gold.index)
    if unknown:
        raise ValueError(f"Audio files with no challenge item: {sorted(unknown)}")
    clips = gold.loc[sorted(paths)].reset_index()
    clips["path"] = [paths[i] for i in clips.id]
    return clips


def evaluate_audio(transcribe: Transcriber, audio_dir: Path = AUDIO_DIR) -> pd.DataFrame:
    """Pooled WER/CER per language and 'all' (total edits / total reference words or chars)."""
    clips = audio_clips(audio_dir)
    hyps = [transcribe(p) for p in clips.path]
    pairs = list(zip(clips.text, hyps, strict=True))
    clips["word_edits"] = [edit_distance(_words(r), _words(h)) for r, h in pairs]
    clips["words"] = [len(_words(r)) for r in clips.text]
    clips["char_edits"] = [edit_distance(_chars(r), _chars(h)) for r, h in pairs]
    clips["chars"] = [len(_chars(r)) for r in clips.text]
    groups = {"all": clips, **{lang: g for lang, g in clips.groupby("language")}}
    return pd.DataFrame({name: {"clips": len(g),
                                "wer": g.word_edits.sum() / g.words.sum(),
                                "cer": g.char_edits.sum() / g.chars.sum()}
                         for name, g in groups.items()}).T


# ---------- language ID ----------

def langid_accuracy(detect: Detector, gold: pd.DataFrame) -> pd.Series:
    """Accuracy per language plus 'all'."""
    right = gold.text.map(detect) == gold.language
    return pd.Series({"all": right.mean(), **right.groupby(gold.language).mean()})


def langdetect_detector() -> Detector:
    """A learned n-gram language identifier (langdetect, 55 languages). It has no code for
    Hinglish; anything outside hi/pa/en counts as 'und'."""
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = SEED  # langdetect samples n-grams at random

    def run(text: str) -> str:
        try:
            code = detect(text)
        except LangDetectException:  # no letters at all
            return "und"
        return code if code in ("hi", "pa", "en") else "und"

    return run


# ---------- translation ----------

def kept(item: str, translation: str) -> bool:
    """True if any '|'-separated spelling of `item` appears in `translation` as whole words."""
    padded = f" {normalize(translation)} "
    return any(f" {normalize(v)} " in padded for v in item.split("|"))


def translate_items(translate: Translator, refs: pd.DataFrame) -> pd.DataFrame:
    """Per item: the translation, its sentence chrF and how many `keep` items survived."""
    import sacrebleu

    hyps = [translate(t, lang) for t, lang in zip(refs.text, refs.language, strict=True)]
    pairs = list(zip(hyps, refs.english, refs.keep, strict=True))
    return pd.DataFrame({
        "id": refs.id, "language": refs.language, "reference": refs.english, "hyp": hyps,
        "chrf": [sacrebleu.sentence_chrf(h, [r]).score for h, r, _ in pairs],
        "kept": [sum(kept(k, h) for k in keep) for h, _, keep in pairs],
        "to_keep": refs.keep.map(len)}, index=refs.index)


def score_translations(items: pd.DataFrame) -> pd.DataFrame:
    """Corpus chrF and keep rate per language plus 'all'."""
    import sacrebleu

    groups = {"all": items, **{lang: g for lang, g in items.groupby("language")}}
    return pd.DataFrame({name: {
        "n": len(g),
        "chrf": sacrebleu.corpus_chrf(list(g.hyp), [list(g.reference)]).score,
        "keep": g.kept.sum() / max(g.to_keep.sum(), 1)} for name, g in groups.items()}).T


# ---------- baselines ----------

def majority_extractor() -> Extractor:
    """Trivial floor: the commonest train issue type and language, no location, no cues."""
    train = load_corpus("train")
    out = json.dumps({"language": train.language.mode()[0],
                      "issue_type": IssueType(train.issue_type.mode()[0]).value,
                      "location_text": "", "cues": Cues().model_dump()})
    return lambda text: out


def translation_report(llm_model: str | None) -> None:
    """Step 4.2: language ID on both sets, then translators on the challenge references."""
    import time
    from functools import partial

    from intake import keywords, llm
    from intake import translate as T

    fmt = {"float_format": lambda x: f"{x:.2f}"}
    sets = {"corpus": load_corpus_test(), "challenge": load_challenge()}
    majority = load_corpus("train").language.mode()[0]
    detectors = {"majority": lambda text: majority, "script rule": keywords.detect_language,
                 "langdetect": langdetect_detector()}
    lid = pd.DataFrame({f"{name}/{s}": langid_accuracy(fn, gold)
                        for name, fn in detectors.items() for s, gold in sets.items()}).T
    print("Language ID accuracy\n" + lid.to_string(**fmt) + "\n", flush=True)

    confidence = []

    def indictrans2(text: str, language: str) -> str:
        """IndicTrans2 on every language, Hinglish as Hindi: the evidence for the routing."""
        result = T.indictrans([text], "pan_Guru" if language == "pa" else "hin_Deva")[0]
        confidence.append(result.confidence)
        return result.text

    translators = {"identity": lambda text, language: text, "indictrans2": indictrans2}
    if llm_model:  # both need Ollama: the LLM alone, and the served route (Hinglish -> LLM)
        llm_translate = partial(llm.translate, model=llm_model)
        translators["llm"] = lambda text, language: llm_translate(text)
        translators["served"] = lambda text, language: T.translate(
            text, language, llm_translate=llm_translate).text
    refs, items, rows = load_references(), {}, {}
    T.translate("नमस्ते", "hi")  # load IndicTrans2 outside the timing
    for name, fn in translators.items():  # print as each run ends, as in main()
        start = time.perf_counter()
        items[name] = translate_items(fn, refs)
        table = score_translations(items[name])
        rows[name] = {**{f"chrf_{g}": table.chrf[g] for g in table.index},
                      **{f"keep_{g}": table.keep[g] for g in table.index},
                      "s_per_item": (time.perf_counter() - start) / len(refs)}
        print(pd.DataFrame(rows).T.loc[[name]].to_string(header=name == "identity", **fmt),
              flush=True)

    print("\nTranslation, challenge set (36 non-English items)\n"
          + pd.DataFrame(rows).T.to_string(**fmt))
    it2 = items["indictrans2"].assign(confidence=confidence)
    for label, part in [("all", it2), ("hi+pa", it2[it2.language != "mixed"])]:
        rho = part.confidence.corr(part.chrf, method="spearman")
        print(f"IndicTrans2 confidence vs sentence chrF, {label}: Spearman {rho:.2f}")
    out = DATA_DIR / "intake" / "translations.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(items, names=["translator"]).to_csv(out)
    print(f"Every translation: {out}")
    for name, it in items.items():
        if name != "identity":
            worst = it.nsmallest(4, "chrf")[["id", "chrf", "hyp"]]
            print(f"\nLowest chrF, {name}\n" + worst.to_string(index=False, **fmt))


def report_extractor(process: Callable, gold: pd.DataFrame, reports: dict) -> Extractor:
    """An intake `process` (RawReport -> Report) as an Extractor, keeping each Report by id."""
    from datetime import UTC, datetime

    from contracts.api import RawReport

    ids = dict(zip(gold.text, gold.id, strict=True))

    def extract(text: str) -> str:
        raw = RawReport(report_id=ids[text], submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
                        text=text)
        r = reports[raw.report_id] = process(raw)
        return json.dumps({"language": r.language, "issue_type": r.issue_type.value,
                           "location_text": r.location_text, "cues": r.cues.model_dump()},
                          ensure_ascii=False)
    return extract


def api_report() -> None:
    """Step 4.4: the whole path, RawReport -> Report, on the challenge set. Baseline: the
    contracts stub, which intake served before v1. Checks the API reproduces the 4.2/4.3
    numbers, scores english_text, shows confidence per language, and checks the log."""
    import tempfile
    import time
    from functools import partial

    from contracts.api import StubIntake, read_log, verify_chain
    from intake import api

    fmt = {"float_format": lambda x: f"{x:.2f}"}
    gold, refs = load_challenge(), load_references()
    log = Path(tempfile.mkdtemp()) / "decision_log.jsonl"
    runs = {"stub": StubIntake().process, "v1": partial(api.process, log_path=log)}
    reports, rows, chrf = {name: {} for name in runs}, {}, {}
    for name, process in runs.items():
        start = time.perf_counter()
        table = evaluate(report_extractor(process, gold, reports[name]), gold)
        rows[name] = {**table.loc["all"], "s_per_item": (time.perf_counter() - start) / len(gold)}
        english = {r.original_text: r.english_text for r in reports[name].values()}
        items = translate_items(lambda text, language, english=english: english[text], refs)
        chrf[name] = score_translations(items).chrf
        print(f"{name} done", flush=True)
    summary = pd.DataFrame(rows).T.astype({"n": int})
    print("\nChallenge set through process(), all languages\n" + summary.to_string(**fmt))
    print("\nenglish_text chrF (36 non-English items)\n" + pd.DataFrame(chrf).T.to_string(**fmt))

    v1 = pd.DataFrame([r.model_dump() for r in reports["v1"].values()]).merge(
        gold[["id", "issue_type"]].rename(columns={"id": "report_id", "issue_type": "gold"}))
    v1["issue_ok"] = v1.issue_type == v1.gold
    v1["weakest"] = v1.explanation.str.extract(r"weakest stage \((\w+)\)")
    by_lang = v1.groupby("language").agg(n=("report_id", "size"), issue_acc=("issue_ok", "mean"),
                                         conf_mean=("confidence", "mean"),
                                         conf_min=("confidence", "min"),
                                         weakest=("weakest", lambda s: s.mode()[0]))
    print("\nv1 confidence by language\n" + by_lang.to_string(**fmt))
    print("Versions: " + str(v1.model_version.value_counts().to_dict()))
    wrong = v1[~v1.issue_ok][["report_id", "gold", "issue_type", "confidence"]]
    print("\nIssue errors (gold, v1, confidence)\n" + (wrong.to_string(index=False, **fmt)
                                                       if len(wrong) else "none"))
    entries = read_log(log)
    print(f"\nDecision log: {len(entries)} entries for {len(gold)} reports, all module="
          f"{sorted({e.module for e in entries})}; verify_chain first broken = "
          f"{verify_chain(log)}")


def main() -> None:
    import argparse
    import time
    from functools import partial

    from intake import llm
    from intake.keywords import extract as keyword_extract

    parser = argparse.ArgumentParser(description="Score intake extractors.")
    parser.add_argument("--llm", action="store_true",
                        help="also score the LLM extractor (needs `ollama serve`, ~25 min)")
    parser.add_argument("--translate", action="store_true",
                        help="score language ID and translation instead (Step 4.2); with --llm "
                             "the LLM translates too")
    parser.add_argument("--api", action="store_true",
                        help="score the whole intake API against the stub instead (Step 4.4; "
                             "needs `ollama serve`, ~5 min)")
    parser.add_argument("--model", default=llm.MODEL)
    args = parser.parse_args()
    if args.api:
        api_report()
        return
    if args.translate:
        translation_report(args.model if args.llm else None)
        return

    sets = {"corpus": load_corpus_test(), "challenge": load_challenge()}
    runs = [(name, fn, s) for name, fn in [("majority", majority_extractor()),
                                           ("keywords", keyword_extract)] for s in sets]
    stats = {s: Counter() for s in sets}
    if args.llm:
        runs += [("llm", partial(llm.extract, model=args.model, stats=stats[s]), s) for s in sets]
        # Ablations, challenge set only: the model alone (no retry, fallback or span check),
        # with and without the schema constraint.
        runs += [("llm-once", partial(llm.extract_once, model=args.model), "challenge"),
                 ("llm-unconstrained",
                  partial(llm.extract_once, model=args.model, constrained=False), "challenge")]

    pd.set_option("display.width", 180)
    fmt = {"float_format": lambda x: f"{x:.2f}"}
    tables, seconds = {}, {}
    for name, fn, s in runs:  # print as each run ends: a stall must not lose finished runs
        start = time.perf_counter()
        tables[name, s] = evaluate(fn, sets[s])
        seconds[name, s] = (time.perf_counter() - start) / len(sets[s])
        row = tables[name, s].loc[["all"]].rename(index={"all": f"{name}/{s}"})
        print(f"[{seconds[name, s]:.2f} s/item] " + row.to_string(header=False, **fmt), flush=True)
    summary = pd.DataFrame({k: t.loc["all"] for k, t in tables.items()}).T.astype({"n": int})
    summary["s_per_item"] = pd.Series(seconds)
    print("\nAll languages\n" + summary.to_string(**fmt))
    for (name, s), table in tables.items():
        if name in ("keywords", "llm"):
            print(f"\n{name} by language, {s} set\n" + table.to_string(**fmt))
    if args.llm:
        print(f"\nLLM path counts ({args.model}): "
              + "; ".join(f"{s} {dict(c) or 'none'}" for s, c in stats.items()))

    found = set(audio_clips().id) if AUDIO_DIR.exists() else set()
    missing = [i for i in AUDIO_IDS if i not in found]
    print(f"\nAudio: {len(found)} clips in {AUDIO_DIR}; WER needs a transcriber (Step 4.2).")
    if missing:
        print(f"Missing {len(missing)}: " + ", ".join(missing))


if __name__ == "__main__":
    main()
