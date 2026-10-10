"""MLflow model registry for triage and router (Step 3.4): python -m trainer.registry

One registered model per module: `civicloop-triage`, `civicloop-router`. Each registered
version points at a folder in models/<module>/<version> and carries a `civicloop_version` tag
(v1, v2, ...) plus a `status` tag (champion, challenger, retired, rolled_back, rejected_offline).

MLflow 3 replaced stages (Staging/Production) with aliases. Two aliases per model:
  @champion    what serving should load
  @challenger  the candidate under canary
Promotion and rollback only move aliases; no version is ever deleted, so history stays.

Serving still reads config.MODEL_VERSIONS: making the APIs resolve `@champion` at load time is
integration work, out of scope for now. The trainer itself always asks the registry.
"""

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

import config

URI = f"sqlite:///{config.MLRUNS_DIR / 'mlflow.db'}"
MODULES = ("triage", "router")
CHAMPION, CHALLENGER = "champion", "challenger"


def client(uri: str = URI) -> MlflowClient:
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def name(module: str) -> str:
    return f"civicloop-{module}"


def _versions(c: MlflowClient, module: str) -> list:
    return c.search_model_versions(f"name = '{name(module)}'")


def register(module: str, version: str, tags: dict | None = None, uri: str = URI) -> str:
    """Register models/<module>/<version>; returns the registry version number. Idempotent:
    a version already registered gets its tags updated instead of a duplicate."""
    c = client(uri)
    try:
        c.create_registered_model(name(module))
    except MlflowException:
        pass  # already exists
    tags = {k: str(v) for k, v in (tags or {}).items()}
    for mv in _versions(c, module):
        if mv.tags.get("civicloop_version") == version:
            for k, v in tags.items():
                c.set_model_version_tag(name(module), mv.version, k, v)
            return mv.version
    mv = c.create_model_version(
        name(module), source=(config.MODELS_DIR / module / version).as_uri(),
        tags={"civicloop_version": version, **tags})
    return mv.version


def alias(module: str, which: str, uri: str = URI) -> str | None:
    """The civicloop version (v1, v2...) behind an alias, or None if the alias is not set."""
    try:
        mv = client(uri).get_model_version_by_alias(name(module), which)
    except MlflowException:
        return None
    return mv.tags["civicloop_version"]


def _set(module: str, which: str, version: str, status: str, uri: str) -> None:
    c = client(uri)
    number = register(module, version, {"status": status}, uri)
    c.set_registered_model_alias(name(module), which, number)


def bootstrap(uri: str = URI) -> dict[str, str]:
    """Make sure every module has a champion; the first one is what config serves today."""
    for m in MODULES:
        if alias(m, CHAMPION, uri) is None:
            _set(m, CHAMPION, config.MODEL_VERSIONS[m], "champion", uri)
    return {m: alias(m, CHAMPION, uri) for m in MODULES}


def make_challenger(module: str, version: str, tags: dict | None = None,
                    uri: str = URI) -> None:
    register(module, version, tags, uri)
    _set(module, CHALLENGER, version, "challenger", uri)


def promote(module: str, uri: str = URI) -> str:
    """Challenger becomes champion; the old champion is retired. Returns the new champion."""
    new, old = alias(module, CHALLENGER, uri), alias(module, CHAMPION, uri)
    if new is None:
        raise ValueError(f"{module}: no challenger to promote")
    register(module, old, {"status": "retired"}, uri)
    _set(module, CHAMPION, new, "champion", uri)
    client(uri).delete_registered_model_alias(name(module), CHALLENGER)
    return new


def rollback(module: str, uri: str = URI) -> str:
    """Drop the challenger; the champion keeps serving. Returns the champion."""
    old = alias(module, CHALLENGER, uri)
    if old is not None:
        register(module, old, {"status": "rolled_back"}, uri)
        client(uri).delete_registered_model_alias(name(module), CHALLENGER)
    return alias(module, CHAMPION, uri)


def next_version(module: str, uri: str = URI) -> str:
    """One past the highest version ever registered: a rolled-back v2 is never overwritten."""
    numbers = [int(mv.tags["civicloop_version"][1:]) for mv in _versions(client(uri), module)
               if mv.tags.get("civicloop_version", "").startswith("v")]
    return f"v{max(numbers, default=0) + 1}"


def reset(uri: str = URI) -> None:
    """Delete the civicloop registered models (not the files, not the training runs)."""
    c = client(uri)
    for m in MODULES:
        try:
            c.delete_registered_model(name(m))
        except MlflowException:
            pass


def table(uri: str = URI) -> list[dict]:
    c, rows = client(uri), []
    for m in MODULES:
        try:
            aliases = c.get_registered_model(name(m)).aliases  # {alias: number}
        except MlflowException:
            continue
        for mv in sorted(_versions(c, m), key=lambda v: int(v.version)):
            rows.append({"model": name(m), "registry_version": int(mv.version),
                         "version": mv.tags.get("civicloop_version"),
                         "status": mv.tags.get("status", ""),
                         "aliases": ",".join(a for a, n in aliases.items()
                                             if int(n) == int(mv.version))})
    return rows


def main() -> None:
    import pandas as pd
    print(pd.DataFrame(table()).to_string(index=False))


if __name__ == "__main__":
    main()
