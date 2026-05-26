"""Registry-level tests with probes stubbed out.

The real probes import heavy native packages; here we monkey-patch
``PROBES`` to verify the registry's own caching, refresh, and report
construction logic in isolation.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

import pytest

from src.shared.python.feature_registry import (
    CapabilityRegistry,
    FeatureReport,
    get_registry,
)
from src.shared.python.feature_registry import probes as probes_mod
from src.shared.python.feature_registry.probes import ProbeOutcome

pytestmark = pytest.mark.unit


@pytest.fixture
def stub_probes(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[int]]:
    """Replace every probe with a counter-incrementing stub returning AVAILABLE.

    Yields a dict so tests can assert call counts per probe key.
    """
    call_counts: dict[str, list[int]] = {key: [0] for key in probes_mod.PROBES}

    def make_stub(key: str) -> Callable[[Path], ProbeOutcome]:
        def _stub(_root: Path) -> ProbeOutcome:
            call_counts[key][0] += 1
            return ProbeOutcome(
                available=True,
                version="stub-1.0",
                message=f"stub {key}",
            )

        return _stub

    stub_probes = {key: make_stub(key) for key in probes_mod.PROBES}
    monkeypatch.setattr(probes_mod, "PROBES", stub_probes)
    # The registry imports PROBES directly — patch that name too.
    from src.shared.python.feature_registry import registry as registry_mod

    monkeypatch.setattr(registry_mod, "PROBES", stub_probes)
    return call_counts


def test_snapshot_returns_one_report_per_feature(stub_probes) -> None:
    from src.shared.python.feature_registry.features import FEATURES

    registry = CapabilityRegistry()
    snapshot = registry.snapshot()
    assert len(snapshot) == len(FEATURES)
    assert all(isinstance(r, FeatureReport) for r in snapshot)


def test_snapshot_caches_probe_results(stub_probes) -> None:
    registry = CapabilityRegistry()
    registry.snapshot()
    registry.snapshot()
    # Every probe should have been called exactly once.
    for key, counter in stub_probes.items():
        assert counter[0] == 1, f"probe {key} ran {counter[0]} times; expected 1"


def test_refresh_reruns_probes(stub_probes) -> None:
    registry = CapabilityRegistry()
    registry.snapshot()
    registry.refresh()
    for counter in stub_probes.values():
        assert counter[0] == 2


def test_refresh_one_reruns_only_that_feature(stub_probes) -> None:
    registry = CapabilityRegistry()
    registry.snapshot()
    registry.refresh_one("mujoco")
    assert stub_probes["mujoco"][0] == 2
    # Sibling probes must still show exactly one call.
    for key, counter in stub_probes.items():
        if key == "mujoco":
            continue
        assert counter[0] == 1


def test_check_unknown_feature_raises(stub_probes) -> None:
    registry = CapabilityRegistry()
    with pytest.raises(KeyError):
        registry.check("not-a-real-feature")


def test_feature_with_no_probe_is_always_available() -> None:
    """``api`` has ``probe_key=None`` — must report available without probing."""
    registry = CapabilityRegistry()
    report = registry.check("api")
    assert report.available is True


def test_check_without_probe_does_not_eagerly_refresh_everything(stub_probes) -> None:
    """Single-feature checks should not execute unrelated probes on first use."""
    registry = CapabilityRegistry()
    report = registry.check("api")
    assert report.available is True
    for counter in stub_probes.values():
        assert counter[0] == 0


def test_check_only_runs_requested_probe_on_first_use(stub_probes) -> None:
    """The first targeted check should evaluate only that feature's probe."""
    registry = CapabilityRegistry()
    report = registry.check("mujoco")
    assert report.available is True
    assert stub_probes["mujoco"][0] == 1
    for key, counter in stub_probes.items():
        if key == "mujoco":
            continue
        assert counter[0] == 0


def test_probe_exception_surfaces_as_unavailable(monkeypatch) -> None:
    """A misbehaving probe must not crash the registry."""

    def explode(_root: Path) -> ProbeOutcome:
        raise RuntimeError("boom")

    from src.shared.python.feature_registry import registry as registry_mod

    monkeypatch.setitem(registry_mod.PROBES, "mujoco", explode)

    registry = CapabilityRegistry()
    report = registry.check("mujoco")
    assert report.available is False
    assert "boom" in report.message


def test_get_registry_returns_singleton() -> None:
    a = get_registry()
    b = get_registry()
    assert a is b


def test_to_dict_round_trips_lists() -> None:
    """``FeatureReport.to_dict`` converts tuple fields to lists for JSON."""
    registry = CapabilityRegistry()
    report = registry.check("api")
    data = report.to_dict()
    assert isinstance(data["missing"], list)
    assert isinstance(data["depends_on"], list)


def test_probe_invariant_validates_at_import_time() -> None:
    """Every feature with probe_key must have a registered probe.

    This is checked at module import; if it ever drifts, the package
    won't import. We re-assert it here so a regression surfaces as a
    failing test rather than a startup error in production.
    """
    from src.shared.python.feature_registry.features import FEATURES
    from src.shared.python.feature_registry.probes import PROBES

    for feature in FEATURES:
        if feature.probe_key is not None:
            assert feature.probe_key in PROBES, (
                f"Feature {feature.name!r} declares probe_key "
                f"{feature.probe_key!r} but it is not registered in PROBES"
            )
