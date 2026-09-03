"""ADR-0046 Stage 2 (G2): the wave-1 modules are retired onto the canonical layer.

Wave 1 re-pointed `dispersion`, `multivariate`, `trends`, `comparison`,
`schema`, and `treatment` at the canonical launch-monitor layer vendored from
Tools (`vendor/ud-tools/src/shared/python/launch_monitor/`, Tools#4899-#4900,
pinned by UD#9400) and deleted UpstreamDrift's private copies. This file is
what keeps the retirement real.

**Before the retirement** this file measured two preconditions. The first was
twin identity — the UpstreamDrift copy and the vendored copy had to be the
same program, compared as normalised syntax trees, or the re-point would not
have been a no-op. That assertion is obsolete by construction: there is no
longer a second copy to compare, and comparing a deleted file to itself is not
a gate. It is replaced below by the assertion that actually holds now, that
the UpstreamDrift file is gone and the canonical module imports in its place.

The second precondition was that ``shared.python.launch_monitor`` resolve to
the vendored package at all. Until 2026-09-02 it did not: it resolved to
UpstreamDrift's own ``src/shared/python/launch_monitor``, because both were
regular packages on ``shared.python.__path__`` and the UpstreamDrift entry
preceded the vendor entry, so the import rewrite ADR-0048's port order
prescribes was self-referential. ADR-0048's "Stage 2 Blocker (G2)" Option 1
cleared it (#9420) by moving the UpstreamDrift copy out of that namespace to
``src/tools/launch_monitor_model/``. That precondition is still a live gate
and is still asserted here: it is the provenance probe every remaining wave
depends on, and it goes red if a UpstreamDrift package ever re-enters the
namespace.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UD_PACKAGE = _REPO_ROOT / "src" / "tools" / "launch_monitor_model"
_VENDORED_PACKAGE = (
    _REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python" / "launch_monitor"
)

# ADR-0046 Stage 2 wave 1: the six modules ADR-0048 orders first (P1-P6),
# every one of them a tier-0 leaf with no intra-package dependency.
WAVE_1_MODULES = (
    "comparison",
    "dispersion",
    "multivariate",
    "schema",
    "treatment",
    "trends",
)

_MISSING_VENDOR_HINT = (
    f"The vendored Tools tree is missing at {_VENDORED_PACKAGE}. Run "
    "`git submodule update --init vendor/ud-tools` to materialise it. In CI "
    "this is a hard failure: these modules are no longer served by "
    "UpstreamDrift at all, so a gate that silently skips would hide a "
    "workbench that cannot import."
)


def _require_vendored_package() -> Path:
    """Return the vendored canonical package, or fail closed in CI."""
    if _VENDORED_PACKAGE.is_dir():
        return _VENDORED_PACKAGE
    if os.environ.get("CI"):
        raise AssertionError(_MISSING_VENDOR_HINT)
    pytest.skip(_MISSING_VENDOR_HINT)


@pytest.mark.parametrize("module_name", WAVE_1_MODULES)
def test_wave_1_module_is_retired_and_served_by_the_canonical_layer(
    module_name: str,
) -> None:
    """No UpstreamDrift copy remains, and the canonical module imports.

    Both halves matter. Deleting the file without the canonical import
    resolving would break the workbench; the canonical import resolving while
    a UpstreamDrift copy quietly returned would mean the retirement never
    happened and the two could drift again. Asserting them together is what
    makes "retired onto the canonical layer" a checkable claim rather than a
    changelog sentence.
    """
    vendored_package = _require_vendored_package()

    ud_copy = _UD_PACKAGE / f"{module_name}.py"
    assert not ud_copy.exists(), (
        f"{ud_copy.relative_to(_REPO_ROOT)} exists again. ADR-0046 Stage 2 "
        "wave 1 retired this module; UpstreamDrift consumes the canonical "
        "implementation from Tools through "
        f"shared.python.launch_monitor.{module_name}. A re-added copy shadows "
        "nothing here but does fork the implementation, which is the "
        "divergence ADR-0046 was accepted to end. Land the change in Tools "
        "and bump the vendor pin."
    )

    module = importlib.import_module(f"shared.python.launch_monitor.{module_name}")
    assert module.__file__ is not None
    resolved = Path(module.__file__).resolve()
    assert resolved == (vendored_package / f"{module_name}.py").resolve(), (
        f"shared.python.launch_monitor.{module_name} imported from {resolved}, "
        f"not from the vendored canonical package at "
        f"{vendored_package / f'{module_name}.py'}."
    )


def test_trends_exposes_the_renamed_result_without_a_back_compat_alias() -> None:
    """ADR-0048 P3's rename is the one symbol change wave 1 carries.

    `rate_of_closure` exports `TrendResult` for a different estimand — a
    cumulative session-ordinal mean, not a per-day robust slope — so the
    canonical module was deliberately landed as `TemporalTrendResult` with no
    alias (Tools#4899). Keeping an alias here would re-create exactly the
    silent name collision the rename exists to prevent, so its absence is
    asserted rather than assumed.
    """
    _require_vendored_package()

    trends = importlib.import_module("shared.python.launch_monitor.trends")
    assert hasattr(trends, "TemporalTrendResult"), (
        "shared.python.launch_monitor.trends must export TemporalTrendResult."
    )
    assert not hasattr(trends, "TrendResult"), (
        "shared.python.launch_monitor.trends exports TrendResult again. The "
        "P3 rename was deliberate and alias-free because rate_of_closure "
        "exports that name for a different estimand; re-adding it re-creates "
        "the collision. See ADR-0048's port order, P3."
    )

    facade = importlib.import_module("src.tools.launch_monitor_model")
    assert "TemporalTrendResult" in facade.__all__
    assert "TrendResult" not in facade.__all__, (
        "The UpstreamDrift façade re-exports TrendResult again. Wave 1 moved "
        "its consumers to TemporalTrendResult with no alias."
    )


def test_canonical_import_path_resolves_into_the_vendored_tools_layer() -> None:
    """``shared.python.launch_monitor`` must be the vendored canonical package.

    This is the provenance probe for every wave of ADR-0046 Stage 2. The
    re-point ADR-0048's port order prescribes is only a re-point if the
    canonical name resolves somewhere other than UpstreamDrift's own tree;
    while both packages sat on ``shared.python.__path__`` it was a
    self-referential alias, because the first entry that carries an
    ``__init__.py`` wins outright and ``src/shared/python`` precedes
    ``vendor/ud-tools/src/shared/python``.

    ADR-0048 "Stage 2 Blocker (G2)" Option 1 resolved that by moving the
    UpstreamDrift copy to ``src/tools/launch_monitor_model/``. This test keeps
    it resolved: if any UpstreamDrift package named ``launch_monitor`` ever
    reappears under ``src/shared/python/``, the retirements above would
    silently re-point at UpstreamDrift code while claiming to consume Tools,
    and that regression fails here first.
    """
    vendored_package = _require_vendored_package()

    spec = importlib.util.find_spec("shared.python.launch_monitor")
    assert spec is not None and spec.origin is not None, (
        "shared.python.launch_monitor did not resolve at all. Stage 2 "
        "consumers import through this name, so it must resolve to the "
        "vendored canonical package. Check that vendor/ud-tools is "
        "materialised."
    )

    resolved = Path(spec.origin).resolve()
    assert resolved == (vendored_package / "__init__.py").resolve(), (
        f"shared.python.launch_monitor resolves to {resolved}, not to the "
        f"vendored canonical package at "
        f"{(vendored_package / '__init__.py')}. A UpstreamDrift package has "
        "re-entered the shared.python namespace and is shadowing the "
        "canonical layer again — see ADR-0048, 'Stage 2 Blocker (G2)', and "
        "scripts/config/shadow_modules.yaml."
    )
