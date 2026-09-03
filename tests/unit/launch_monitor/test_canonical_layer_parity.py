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

Wave 2 retires four more tier-1 modules — ADR-0048's port order steps P7
(`relationships`), P8 (`modeling`), and P9 (`profiles` + `importer`) — onto
the same canonical layer. `modeling`, `profiles`, and `importer` are AST-
identical twins exactly like every wave-1 module, so
:func:`test_wave_2_module_is_retired_and_served_by_the_canonical_layer` below
is the same shape as wave 1's assertion. `relationships` is not identical: its
canonical twin carries owner ruling **D17** (ADR-0048 "Owner Rulings
(2026-09-02)") — booleans are still analysed as 0/1, but the projection is now
explicit rather than silent. That is not a rename like P3's `TrendResult`; it
is additive (`CorrelationResult.boolean_projected`,
`DependencyEdge.includes_boolean_projection`), so retiring it is a behaviour
*addition*, not a behaviour change, and
:func:`test_relationships_gains_the_d17_boolean_projection_fields_additively`
pins that distinction directly rather than folding `relationships` into the
identical-twin parametrization.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import os
import subprocess  # nosec B404 - fixed interpreter invocation, no shell
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

# ADR-0046 Stage 2 wave 2: P7-P9 of ADR-0048's port order. All four retire
# together and share the same "file gone, canonical import resolves"
# assertion below regardless of twin status; `relationships` additionally
# gets test_relationships_gains_the_d17_boolean_projection_fields_additively
# because unlike the other three it is not an identical twin — owner ruling
# D17 makes it an additive behaviour change, not a pure re-point.
WAVE_2_MODULES = (
    "importer",
    "modeling",
    "profiles",
    "relationships",
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


@pytest.mark.parametrize("module_name", WAVE_2_MODULES)
def test_wave_2_module_is_retired_and_served_by_the_canonical_layer(
    module_name: str,
) -> None:
    """Wave 2's four modules pass the same retirement check wave 1's did.

    Identical in shape to
    :func:`test_wave_1_module_is_retired_and_served_by_the_canonical_layer`;
    kept as a second parametrization rather than merged into
    ``WAVE_1_MODULES`` so a wave-2 regression reads as a wave-2 failure. This
    check does not care whether the retired module is an identical twin —
    that is a separate claim, made for `relationships` by
    :func:`test_relationships_gains_the_d17_boolean_projection_fields_additively`
    below.
    """
    vendored_package = _require_vendored_package()

    ud_copy = _UD_PACKAGE / f"{module_name}.py"
    assert not ud_copy.exists(), (
        f"{ud_copy.relative_to(_REPO_ROOT)} exists again. ADR-0046 Stage 2 "
        "wave 2 retired this module; UpstreamDrift consumes the canonical "
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


def _shots(n: int = 80) -> pd.DataFrame:
    """The deterministic synthetic shot frame every relationships test uses.

    Same seed, same columns, same coefficients as
    ``tests/unit/launch_monitor/test_analysis.py``'s private ``_shots`` helper
    and Tools' ported ``build_shots``
    (``vendor/ud-tools/tests/shared/python/launch_monitor/conftest.py``) --
    duplicated here rather than imported across test modules so this file
    stays self-contained, matching its existing style.
    """
    rng = np.random.default_rng(42)
    club = np.linspace(35.0, 50.0, n)
    attack = rng.normal(-0.04, 0.025, n)
    ball = 1.47 * club + 3.0 * attack + rng.normal(0.0, 0.7, n)
    return pd.DataFrame(
        {
            "shot_id": [f"s{i}" for i in range(n)],
            "session_id": np.where(np.arange(n) < n / 2, "a", "b"),
            "monitor_vendor": np.where(np.arange(n) % 2, "Garmin", "TrackMan"),
            "captured_at": pd.date_range("2026-01-01", periods=n, freq="D"),
            "club_speed": club,
            "attack_angle": attack,
            "ball_speed": ball,
            "smash_factor": ball / club,
            "carry_distance": 3.4 * ball + rng.normal(0.0, 2.0, n),
            "lateral_carry": rng.normal(2.0, 8.0, n),
        }
    )


def test_relationships_gains_the_d17_boolean_projection_fields_additively() -> None:
    """Owner ruling D17 lands as an additive field pair, not a rename.

    ADR-0048's port order table lists P7 (`relationships`) as retiring
    verbatim first, with a note that a follow-up applies D15/D17-class
    rulings afterward; by the time wave 2 retires it, D17 (ADR-0048 "Owner
    Rulings (2026-09-02)") is already applied in the canonical module. Unlike
    the P3 `trends` rename this is additive: `CorrelationResult` gains
    `boolean_projected` and `DependencyEdge` gains
    `includes_boolean_projection`; every field either dataclass carried before
    D17 is still there, unrenamed, so a consumer that only reads the fields it
    already knew about -- the workbench Relationships tab
    (`gui.py::run_relationship_analysis`, which reads only `.coefficients`,
    `.method`, and `.edges`) -- does not break on the addition.

    The math is unchanged: a boolean column still projects to 0/1 and its
    coefficient is bit-identical to analysing the same values pre-cast to
    float. The pinned value reproduces Tools' own pin
    (``tests/shared/python/launch_monitor/test_relationships.py``,
    ``test_boolean_column_projection_is_labelled_and_math_is_unchanged``)
    against the same fixture, rather than trusting the vendored test alone.
    """
    _require_vendored_package()

    relationships = importlib.import_module(
        "shared.python.launch_monitor.relationships"
    )

    result_fields = {
        field.name for field in dataclasses.fields(relationships.CorrelationResult)
    }
    edge_fields = {
        field.name for field in dataclasses.fields(relationships.DependencyEdge)
    }
    assert "boolean_projected" in result_fields, (
        "shared.python.launch_monitor.relationships.CorrelationResult lost "
        "the D17 boolean_projected field."
    )
    assert "includes_boolean_projection" in edge_fields, (
        "shared.python.launch_monitor.relationships.DependencyEdge lost the "
        "D17 includes_boolean_projection field."
    )
    # D17 adds; it does not remove or rename what was already there.
    assert {
        "method",
        "coefficients",
        "p_values",
        "adjusted_p_values",
        "pair_counts",
        "partial_coefficients",
        "derived_metrics",
        "edges",
    } <= result_fields
    assert {
        "source",
        "target",
        "coefficient",
        "p_value",
        "adjusted_p_value",
        "sample_count",
        "includes_derived_metric",
    } <= edge_fields

    frame = _shots(40)
    frame["is_trackman"] = frame["monitor_vendor"] == "TrackMan"
    boolean = relationships.compute_correlations(
        frame, metrics=("club_speed", "is_trackman")
    )
    r = boolean.coefficients.loc["club_speed", "is_trackman"]
    assert r == pytest.approx(-0.04331480818242096), (
        "The D17 boolean-projection coefficient drifted from its pinned "
        "value; the ruling only labels the projection, it must not change "
        "the math."
    )
    assert boolean.boolean_projected == ("is_trackman",)
    assert boolean.pair_counts.loc["club_speed", "is_trackman"] == 40

    facade = importlib.import_module("src.tools.launch_monitor_model")
    assert "CorrelationResult" in facade.__all__
    assert "DependencyEdge" in facade.__all__
    assert "compute_correlations" in facade.__all__


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


def test_facade_imports_without_the_pytest_path_wiring() -> None:
    """The façade must import in a plain source-checkout process, not just here.

    This test's own session is the easy case: ``pyproject.toml``'s pytest
    ``pythonpath`` and ``tests/conftest.py`` both put the vendored Tools source
    within reach, so `shared.python.launch_monitor` resolves before anything
    in the façade runs. A running API server, the launcher, and the
    launch-monitor companion workflow get neither, and after wave 1 the façade
    cannot import at all without them — which is a shipped-behaviour break
    that no in-session test would notice. The façade's
    ``_ensure_canonical_layer_importable`` bootstrap exists for that case and
    this subprocess is what measures it.
    """
    _require_vendored_package()

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((".", "src", "src/shared/python"))
    env.pop("TOOLS_REPO_PATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"

    completed = subprocess.run(  # nosec B603 - fixed argv, no shell
        [
            sys.executable,
            "-c",
            "import src.tools.launch_monitor_model as m;"
            "import shared.python.launch_monitor.trends as t;"
            "print(t.__file__)",
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )

    assert completed.returncode == 0, (
        "The launch-monitor façade does not import in a plain source-checkout "
        "process. After ADR-0046 Stage 2 wave 1 it depends on "
        "shared.python.launch_monitor, which only resolves where the vendored "
        "Tools source is reachable; the pytest session gets that for free and "
        "a running server does not.\n\n"
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    )
    assert str(_VENDORED_PACKAGE.resolve()) in completed.stdout, (
        "The subprocess imported the trends module from "
        f"{completed.stdout.strip()!r}, not from the vendored canonical "
        "package."
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
