import pytest

from trainer import registry


@pytest.fixture
def uri(tmp_path):
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def _aliases(uri):
    return {(r["model"], r["version"]): r["aliases"] for r in registry.table(uri)}


def test_bootstrap_registers_config_versions_as_champion(uri):
    assert registry.bootstrap(uri) == {"triage": "v1", "router": "v1"}
    assert registry.bootstrap(uri) == {"triage": "v1", "router": "v1"}  # idempotent
    assert len(registry.table(uri)) == 2
    assert registry.next_version("triage", uri) == "v2"


def test_register_is_idempotent_and_updates_tags(uri):
    n1 = registry.register("triage", "v2", {"status": "x"}, uri)
    n2 = registry.register("triage", "v2", {"status": "y"}, uri)
    assert n1 == n2
    assert [r["status"] for r in registry.table(uri)] == ["y"]


def test_promote_moves_aliases_and_retires_old_champion(uri):
    registry.bootstrap(uri)
    registry.make_challenger("triage", "v2", uri=uri)
    assert registry.alias("triage", "challenger", uri) == "v2"
    assert registry.promote("triage", uri) == "v2"
    assert registry.alias("triage", "champion", uri) == "v2"
    assert registry.alias("triage", "challenger", uri) is None
    status = {r["version"]: r["status"] for r in registry.table(uri)
              if r["model"] == "civicloop-triage"}
    assert status == {"v1": "retired", "v2": "champion"}


def test_rollback_keeps_champion_and_version_numbers_never_reused(uri):
    registry.bootstrap(uri)
    registry.make_challenger("router", "v2", uri=uri)
    assert registry.rollback("router", uri) == "v1"
    assert registry.alias("router", "challenger", uri) is None
    assert _aliases(uri)[("civicloop-router", "v1")] == "champion"
    assert registry.next_version("router", uri) == "v3"  # the rolled-back v2 stays v2


def test_promote_without_challenger_fails(uri):
    registry.bootstrap(uri)
    with pytest.raises(ValueError):
        registry.promote("triage", uri)


def test_reset(uri):
    registry.bootstrap(uri)
    registry.reset(uri)
    assert registry.table(uri) == []
