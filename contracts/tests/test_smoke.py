"""Smoke test: the repo skeleton is installed and wired the way CLAUDE.md says."""

import importlib

import config


def test_every_module_imports_with_a_public_api():
    for name in config.MODULES:
        api = importlib.import_module(f"{name}.api")
        assert api.__doc__, f"{name}/api.py needs a docstring"
        assert (config.ROOT / name / "tests").is_dir()


def test_config_is_reproducible_and_points_inside_the_repo():
    assert isinstance(config.SEED, int)
    for path in (config.DATA_DIR, config.MODELS_DIR, config.NOTEBOOKS_DIR):
        assert path.is_dir() and path.parent == config.ROOT
    assert set(config.MODEL_VERSIONS) == set(config.MODULES) - {"contracts", "trainer"}
