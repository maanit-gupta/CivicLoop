"""Keyword-rule baseline for intake: the floor every model in Phase 4 must beat.

Rules: language by script; issue type by counting keyword hits; cues by keyword; location as the
span around gazetteer hits (ward names, landmarks, road words). Vocabulary was taken from the
corpus *train* split only. The challenge set was frozen before these lists were written and was
never used to tune them, so its score shows how far train vocabulary carries.
"""

import json
import re

from contracts.api import IssueType
from contracts.corpus_templates import WARD_NAMES
from contracts.world import load_world

# Issue keywords, all scripts together (scripts don't overlap). Latin keywords match at a word
# start, so "pothole" also hits "potholes". Ties go to the earlier issue in this dict.
ISSUE_KEYWORDS = {
    IssueType.WATER_LEAK: ["pipe", "leak", "burst", "jal board", "water main",
                           "पाइप", "लीक", "जल बोर्ड", "ਪਾਈਪ", "ਲੀਕ", "ਜਲ ਬੋਰਡ"],
    IssueType.DRAIN_BLOCKAGE: ["drain", "nala", "naali", "sewer", "clogged",
                               "नाली", "नाला", "सीवर", "ਨਾਲੀ", "ਨਾਲਾ", "ਸੀਵਰ"],
    IssueType.POTHOLE: ["pothole", "crater", "gaddh", "गड्ढ", "ਟੋਆ", "ਟੋਏ", "ਟੋਇਆਂ"],
    IssueType.STREETLIGHT: ["streetlight", "street light", "street lamp", "light", "andhera",
                            "लाइट", "बत्ती", "अंधेरा", "ਲਾਈਟ", "ਬੱਤੀ", "ਹਨੇਰਾ"],
    IssueType.GARBAGE: ["garbage", "trash", "dustbin", "heap of waste", "kooda", "kachra",
                        "कूड़", "कचर", "ਕੂੜ"],
    IssueType.DAMAGED_SIGN: ["sign", "board", "साइन", "बोर्ड", "ਸਾਈਨ", "ਬੋਰਡ"],
    IssueType.OTHER: ["tree", "cattle", "encroachment", "dogs", "ped gir", "stray",
                      "पेड़", "पशु", "कब्ज़ा", "कुत्ते", "ਦਰਖ਼ਤ", "ਪਸ਼ੂ", "ਕਬਜ਼ਾ", "ਕੁੱਤੇ"],
}

CUE_KEYWORDS = {
    "children": ["children", "kids", "bacche", "बच्चे", "ਬੱਚੇ"],
    "accident": ["accident", "rider fell", "wala gir", "एक्सीडेंट", "वाला गिर", "ਐਕਸੀਡੈਂਟ",
                 "ਵਾਲਾ ਡਿੱਗ"],
    "flooding": ["waterlogged", "knee-deep", "paani bhar", "ghutno", "पानी भर", "घुटनों",
                 "ਪਾਣੀ ਭਰ", "ਗੋਡਿਆਂ"],
}

# Location gazetteer: ward names in every script, landmark and road words from the train split.
PLACE_WORDS = ["govt school", "school", "hospital", "market", "bus stop", "main road",
               "colony road", "lane", "gali",
               "सरकारी स्कूल", "अस्पताल", "बाज़ार", "बस स्टॉप", "मेन रोड", "कॉलोनी की सड़क", "गली",
               "ਸਰਕਾਰੀ ਸਕੂਲ", "ਹਸਪਤਾਲ", "ਬਾਜ਼ਾਰ", "ਬੱਸ ਸਟਾਪ", "ਮੇਨ ਰੋਡ", "ਕਲੋਨੀ ਦੀ ਸੜਕ", "ਗਲੀ"]
ARTICLES = ["the", "a"]
LEFT_WORDS = ["near", "opposite", "next to", "on", "in", "at"]
RIGHT_WORDS = ["के पास", "के सामने", "में", "पर", "ਦੇ ਕੋਲ", "ਦੇ ਸਾਹਮਣੇ", "ਵਿੱਚ", "ਉੱਤੇ",
               "ke paas", "ke saamne", "ke bagal mein", "mein", "pe"]
MAX_GAP = 12  # chars between two place hits that still belong to one location phrase

HINGLISH_MARKERS = {"hai", "h", "hain", "nahi", "nhi", "ka", "ki", "ke", "mein", "pe", "se",
                    "ho", "raha", "rahi", "gaya", "gayi", "bhai", "yaar", "koi", "paani", "pani"}


def _pattern(word: str) -> re.Pattern:
    # Latin words must start at a word boundary; Indic words match anywhere (matras break \b).
    return re.compile((r"\b" if word.isascii() else "") + re.escape(word), re.IGNORECASE)


def _count(text: str, words: list[str]) -> int:
    return sum(len(_pattern(w).findall(text)) for w in words)


def detect_language(text: str) -> str:
    deva = sum(0x0900 <= ord(c) <= 0x097F for c in text)  # Devanagari block
    guru = sum(0x0A00 <= ord(c) <= 0x0A7F for c in text)  # Gurmukhi block
    latin = sum(c.isascii() and c.isalpha() for c in text)
    if max(deva, guru, latin) == 0:
        return "und"
    if deva >= max(guru, latin):
        return "hi"
    if guru >= latin:
        return "pa"
    words = set(re.findall(r"[a-z]+", text.lower()))
    return "mixed" if words & HINGLISH_MARKERS else "en"


def classify(text: str) -> IssueType:
    counts = {issue: _count(text, words) for issue, words in ISSUE_KEYWORDS.items()}
    best = max(counts, key=counts.get)  # first max wins: dict order is the tie-break
    return best if counts[best] else IssueType.OTHER


def _place_hits(text: str) -> list[tuple[int, int]]:
    """Non-overlapping gazetteer matches, longest first where they overlap."""
    wards = [w.name for w in load_world().wards]
    wards += [name for names in WARD_NAMES.values() for name in names.values()]
    found = sorted((m.start(), -m.end()) for w in wards + PLACE_WORDS
                    for m in _pattern(w).finditer(text))
    hits = []
    for start, neg_end in found:
        if not hits or start >= hits[-1][1]:
            hits.append((start, -neg_end))
    return hits


def _extend(text: str, start: int, end: int) -> tuple[int, int]:
    """Grow the span by at most one article and one relation word on the left ('near the ...')
    and one postposition on the right ('... के पास'). Once each: 'caved in in a lane' must not
    swallow the verb's 'in'."""
    for words in (ARTICLES, LEFT_WORDS):
        for w in words:
            m = re.search(r"(?:^|\s)(" + re.escape(w) + r")\s+$", text[:start], re.IGNORECASE)
            if m:
                start = m.start(1)
                break
    for w in RIGHT_WORDS:
        m = re.match(r"\s*(" + re.escape(w) + r")(?=$|[\s,.।!?])", text[end:], re.IGNORECASE)
        if m:
            end += m.end(1)
            break
    return start, end


def find_location(text: str) -> str:
    clusters: list[list[tuple[int, int]]] = []
    for hit in _place_hits(text):
        gap = text[clusters[-1][-1][1]:hit[0]] if clusters else None
        if gap is not None and len(gap) <= MAX_GAP and not re.search(r"[.।!?]", gap):
            clusters[-1].append(hit)
        else:
            clusters.append([hit])
    if not clusters:
        return ""
    best = max(clusters, key=len)  # first of the largest clusters
    start, end = _extend(text, best[0][0], best[-1][1])
    return text[start:end].strip(" ,")


def extract(text: str) -> str:
    """Text -> raw JSON in the intake.extraction.Extraction shape."""
    return json.dumps({
        "language": detect_language(text),
        "issue_type": classify(text).value,
        "location_text": find_location(text),
        "cues": {cue: _count(text, words) > 0 for cue, words in CUE_KEYWORDS.items()},
    }, ensure_ascii=False)
