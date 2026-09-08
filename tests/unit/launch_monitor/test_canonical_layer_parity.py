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

Wave 3a retires the contract spine and the longitudinal tier: `corpus`,
`flexible_analysis`, `contract_v2`, `longitudinal_types`,
`longitudinal_statistics`, `longitudinal`, and the four `dataset_reference*`
modules. Unlike waves 1 and 2 this wave could not be assembled out of
behaviour-neutral modules, because the canonical dependency graph does not
contain a behaviour-neutral downward-closed set: canonical `contract_v2`
imports canonical `flexible_analysis` (owner rulings D15/D17) and canonical
`dataset_reference_contract` imports canonical `corpus` (the P19 merge), so
retiring either consumer while its dependency still resolved to UpstreamDrift
would leave two copies of the same dataclass in one process. The wave is
therefore ordered by that graph rather than by how quiet each module is, and
the three rulings it carries are pinned directly:

* **D15** (`flexible_analysis`) — under-sampled predictors leave the
  Benjamini-Hochberg pool before correction. Measured in
  ``tests/integration/launch_monitor_drift/test_flexible_analysis_drift.py``.
* **D17** (`flexible_analysis`) — the boolean 0/1 projection is preserved and
  now labelled, carried up from `relationships` (wave 2) onto
  ``CorrelationEstimate.is_boolean_projected``.
* **G1-D1** (`longitudinal*`) — the pooled estimator is a named-method pair,
  pinned by :func:`test_wave_3a_longitudinal_carries_the_g1_d1_named_method_pair`
  below and measured against ``rate_of_closure`` in
  ``tests/integration/launch_monitor_drift/test_longitudinal_drift.py``.

Wave 3b retires the last eight — `strokes_gained_types`,
`_scoring_statistics`, `strokes_gained`, `outcome_proxy`, the
`player_covariation` trio and `conformance_bundle` — and with them **ADR-0046
Stage 2's module retirement is complete**. What is left in
``src/tools/launch_monitor_model/`` is asserted exhaustively by
:func:`test_stage_2_module_retirement_is_complete`: the façade, the app-local
`project`, and `strokes_gained_baseline`. That third file is not a leftover.
ADR-0048's port order marks P12 as "``strokes_gained_types.py`` (minus
baseline half)" because Tools' ``rate_of_closure.launch_monitor_strokes_gained
_baseline`` is already the authority for that artifact, so the canonical layer
types its ``baseline`` argument against runtime-checkable *protocols*. A
protocol validates nothing at a trust boundary, and the analytics API parses a
baseline off the wire, so UpstreamDrift keeps the hash-verifying pydantic model
app-local. :func:`test_app_local_baseline_satisfies_the_canonical_protocols`
pins the seam between the two.

Wave 3b's rulings:

* **G1-D2** (`strokes_gained*`) — the canonical inference unit is the session
  cell; UpstreamDrift's shot-level fit survives as `shot-level-sg-trend/1`.
  Measured in ``test_strokes_gained_drift.py``.
* **G1-D3** (`strokes_gained`) — exclude-and-audit. Already this module's
  posture; the legacy half was re-pinned in #9419.
* **D22 / D23** (`player_covariation*`) — the between-player Fisher interval is
  withheld below five groups with the absence explained, and units come from
  the registry rather than from a column-name suffix. Both rulings adopted
  *UpstreamDrift's* posture, so no UpstreamDrift-side number moves.
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

# ADR-0046 Stage 2 wave 3a: ADR-0048's P10, P11, P15, P16, P19 and P20, plus
# the P19 `corpus` merge that P20 sits on. Ordered by the canonical dependency
# graph, not by port-order number: `contract_v2` (P11) cannot retire before
# `flexible_analysis` (P10) because the canonical module imports it, and the
# `dataset_reference*` tier (P20) cannot retire before `corpus` (P19) for the
# same reason. Retiring a consumer ahead of its dependency would put two
# copies of `FlexibleAnalysisResult` (or of `CORPUS_COLUMN_MAP`) in one
# process, which is the fork ADR-0046 exists to end rather than to introduce.
WAVE_3A_MODULES = (
    "contract_v2",
    "corpus",
    "dataset_reference",
    "dataset_reference_contract",
    "dataset_reference_operations",
    "dataset_reference_verification",
    "flexible_analysis",
    "longitudinal",
    "longitudinal_statistics",
    "longitudinal_types",
)

# ADR-0046 Stage 2 wave 3b: the last eight. Downward-closed given wave 3a —
# every one of these depends only on `contract_v2`, `schema` and each other,
# all of which the canonical layer already serves.
WAVE_3B_MODULES = (
    "_scoring_statistics",
    "conformance_bundle",
    "outcome_proxy",
    "player_covariation",
    "player_covariation_core",
    "player_covariation_types",
    "strokes_gained",
    "strokes_gained_types",
)

# What ADR-0046 Stage 2 leaves behind in UpstreamDrift, exhaustively.
APP_LOCAL_MODULES = frozenset(
    {
        "__init__",  # the re-export façade and the canonical-layer bootstrap
        "project",  # workbench project file I/O; never had a Tools twin
        "strokes_gained_baseline",  # ADR-0048 P12's deliberate exclusion
    }
)

_MISSING_VENDOR_HINT = (
    f"The vendored Tools tree is missing at {_VENDORED_PACKAGE}. Run "
    "`git submodule update --init vendor/ud-tools` to materialise it. In CI "
    "this is a hard failure: these modules are no longer served by "
    "UpstreamDrift at all, so a gate that silently skips would hide a "
    "workbench that cannot import."
)


def _require_vendored_package() -> Path:
    """Return the vendored canonical package, or fail closed unless opted out."""
    from tests.helpers.seam_guards import require_vendor_path

    return require_vendor_path(_VENDORED_PACKAGE)


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


@pytest.mark.parametrize("module_name", WAVE_3A_MODULES)
def test_wave_3a_module_is_retired_and_served_by_the_canonical_layer(
    module_name: str,
) -> None:
    """Wave 3a's ten modules pass the same retirement check waves 1-2 did.

    Same shape as the two parametrizations above, kept separate so a wave-3a
    regression reads as a wave-3a failure. Twin status is deliberately not
    asserted here: three of these ten carry owner rulings and are *not*
    identical twins, and the rulings are pinned by the dedicated tests below
    and by the drift gates rather than by a blanket identity claim.
    """
    vendored_package = _require_vendored_package()

    ud_copy = _UD_PACKAGE / f"{module_name}.py"
    assert not ud_copy.exists(), (
        f"{ud_copy.relative_to(_REPO_ROOT)} exists again. ADR-0046 Stage 2 "
        "wave 3a retired this module; UpstreamDrift consumes the canonical "
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


def test_wave_3a_retirements_are_dependency_legal() -> None:
    """No retired module may depend on one still served by UpstreamDrift.

    This is the constraint that decided wave 3a's contents, and it is not
    visible from ADR-0048's port order, which is ordered by UpstreamDrift's
    intra-package graph. The canonical modules import each other by the
    canonical name, so a canonical module whose dependency is still an
    UpstreamDrift file gets the *canonical* dependency while the façade
    exports the *UpstreamDrift* one — two `FlexibleAnalysisResult` classes,
    two `AnalysisContextV2` classes, in one process, with pydantic validation
    failing across the seam. Asserting the property directly is cheaper than
    rediscovering it one ValidationError at a time in a later wave.
    """
    vendored_package = _require_vendored_package()

    retired = set(WAVE_1_MODULES) | set(WAVE_2_MODULES) | set(WAVE_3A_MODULES)
    prefix = "from shared.python.launch_monitor."
    still_local = {
        path.stem
        for path in _UD_PACKAGE.glob("*.py")
        if path.stem not in {"__init__", "project"}
    }

    for module_name in sorted(retired):
        source = (vendored_package / f"{module_name}.py").read_text(encoding="utf-8")
        dependencies = {
            line.split(prefix, 1)[1].split(" ", 1)[0].rstrip(".,")
            for line in source.splitlines()
            if line.startswith(prefix)
        }
        leaked = dependencies & still_local
        assert not leaked, (
            f"shared.python.launch_monitor.{module_name} is retired but "
            f"imports {sorted(leaked)}, which UpstreamDrift still serves from "
            f"{_UD_PACKAGE.relative_to(_REPO_ROOT)}. Retire the dependency in "
            "the same wave, or the two layers hold separate copies of the "
            "same classes."
        )


def test_wave_3a_longitudinal_carries_the_g1_d1_named_method_pair() -> None:
    """ADR-0048 Decision G1-D1 lands as a renamed, required method identifier.

    Unlike wave 2's D17, this one is *not* purely additive: the pooled
    estimator's `method` was a single-valued `Literal` defaulting to
    ``"player_fixed_effects_ols_clustered_by_player"``, and it is now a
    required two-valued identifier with no default and no back-compat alias —
    the same posture as wave 1's `TrendResult` rename, and for the same
    reason. G1-D1 states it plainly: "results from different estimators are
    never numerically compared without the names attached", which is only
    enforceable if naming is mandatory. The old string's absence is asserted
    rather than assumed, because a re-added alias would let a
    ``dl-random-effects/1`` number be read under the cluster-robust name.

    The per-player and heterogeneity fields the decision adds are additive and
    optional, so a caller that reads only what it read before still works.
    """
    _require_vendored_package()

    types_module = importlib.import_module(
        "shared.python.launch_monitor.longitudinal_types"
    )
    pooled = types_module.PooledAssociationV1
    assert set(types_module.POOLED_METHOD_DESCRIPTIONS) == {
        "ud-cluster-robust-fe/1",
        "dl-random-effects/1",
    }
    assert pooled.model_fields["method"].is_required(), (
        "PooledAssociationV1.method gained a default again. G1-D1 makes the "
        "estimator identifier mandatory precisely so a pooled number can "
        "never be read without knowing which estimator produced it."
    )
    assert "player_fixed_effects_ols_clustered_by_player" not in str(
        pooled.model_fields["method"].annotation
    ), (
        "The pre-G1-D1 method string is back in PooledAssociationV1. It was "
        "renamed to ud-cluster-robust-fe/1 with no alias; re-adding it "
        "reintroduces the unnamed-estimator hazard the decision removed."
    )

    player = types_module.LongitudinalPlayerAssociationV1
    for added in (
        "standard_error",
        "ci_lower",
        "ci_upper",
        "p_value",
        "r_squared",
        "first_to_last_change",
    ):
        assert added in player.model_fields, (
            f"LongitudinalPlayerAssociationV1 lost the D11 field {added}."
        )
        assert not player.model_fields[added].is_required(), (
            f"{added} became required. D11's fields are optional so a fit "
            "that cannot support one says so by absence rather than by "
            "inventing a number."
        )

    facade = importlib.import_module("src.tools.launch_monitor_model")
    assert "PooledAssociationV1" in facade.__all__
    assert "LongitudinalPlayerAssociationV1" in facade.__all__


def test_wave_3a_flexible_analysis_carries_d15_and_d17() -> None:
    """The two flexible-analysis rulings are visible on the canonical module.

    D17 is additive: `CorrelationEstimate` gains `is_boolean_projected` and
    keeps every field it had. D15 is not additive and cannot be — it changes
    a reported number — so it is pinned where a number can be measured, in
    ``test_flexible_analysis_drift.py``'s
    ``test_resolved_d15_the_fdr_denominator_agrees``. What is asserted here is
    the structural half: that the pool really is filtered before correction,
    read off the source rather than inferred from a value that could coincide.
    """
    vendored_package = _require_vendored_package()

    flexible = importlib.import_module("shared.python.launch_monitor.flexible_analysis")
    fields = {field.name for field in dataclasses.fields(flexible.CorrelationEstimate)}
    assert "is_boolean_projected" in fields, (
        "shared.python.launch_monitor.flexible_analysis.CorrelationEstimate "
        "lost the D17 is_boolean_projected label."
    )
    assert {
        "predictor",
        "coefficient",
        "p_value",
        "adjusted_p_value",
        "ci_lower",
        "ci_upper",
        "sample_count",
        "method",
    } <= fields

    source = (vendored_package / "flexible_analysis.py").read_text(encoding="utf-8")
    assert "correction_input" in source and "min_samples" in source, (
        "The D15 correction pool no longer filters by min_samples before "
        "calling _adjust_p_values. Under-sampled predictors are back in the "
        "Benjamini-Hochberg denominator; see ADR-0048's owner ruling on D15."
    )

    facade = importlib.import_module("src.tools.launch_monitor_model")
    assert "CorrelationEstimate" in facade.__all__
    assert "analyze_variables" in facade.__all__


def test_wave_3a_corpus_keeps_the_dataframe_entry_point_and_gains_provenance() -> None:
    """P19's merge is additive at the entry point UpstreamDrift consumers use.

    ``load_private_corpus`` still returns a plain ``DataFrame`` — that is the
    signature `gui.py` and the analytics routes call — and the merge adds
    ``load_private_corpus_with_provenance`` beside it for callers that want
    the content-addressed manifest identity. The mandatory manifest gate is
    unchanged in kind: UpstreamDrift's copy already carried the same five
    fail-closed checks (#9401), which is what made this retirement a
    re-point rather than a behaviour change.
    """
    _require_vendored_package()

    corpus = importlib.import_module("shared.python.launch_monitor.corpus")
    assert corpus.load_private_corpus.__annotations__["return"] in {
        "pd.DataFrame",
        pd.DataFrame,
    }
    assert hasattr(corpus, "load_private_corpus_with_provenance")
    assert corpus.MAX_RETAINED_ROWS == 300_000
    assert corpus.SUPPORTED_MANIFEST_SCHEMA_VERSION == 1

    facade = importlib.import_module("src.tools.launch_monitor_model")
    assert "load_private_corpus" in facade.__all__
    assert "CORPUS_COLUMN_MAP" in facade.__all__


@pytest.mark.parametrize("module_name", WAVE_3B_MODULES)
def test_wave_3b_module_is_retired_and_served_by_the_canonical_layer(
    module_name: str,
) -> None:
    """Wave 3b's eight modules pass the same retirement check waves 1-3a did."""
    vendored_package = _require_vendored_package()

    ud_copy = _UD_PACKAGE / f"{module_name}.py"
    assert not ud_copy.exists(), (
        f"{ud_copy.relative_to(_REPO_ROOT)} exists again. ADR-0046 Stage 2 "
        "wave 3b retired this module; UpstreamDrift consumes the canonical "
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


def test_stage_2_module_retirement_is_complete() -> None:
    """Nothing is left in the package but the façade and two app-local files.

    This is the terminal assertion of ADR-0046 Stage 2. All twenty-eight modules
    ADR-0048's inventory classified ``port-up`` or ``merge`` are gone from
    UpstreamDrift and served by the canonical layer; what remains is exactly
    the set the ADR classified ``app-local``, plus P12's documented baseline
    exclusion. Written as an equality rather than a subset on purpose: a
    *new* module appearing here is the beginning of the next fork, and this
    test is where it should be argued for, not noticed later.
    """
    _require_vendored_package()

    present = {path.stem for path in _UD_PACKAGE.glob("*.py")}
    assert present == set(APP_LOCAL_MODULES), (
        "src/tools/launch_monitor_model/ no longer holds exactly the app-local "
        f"set. Present: {sorted(present)}; expected: {sorted(APP_LOCAL_MODULES)}. "
        "ADR-0046 Stage 2 retired every port-up and merge module onto the "
        "canonical layer; a new file here forks the implementation again. If "
        "one is genuinely app-local, add it to APP_LOCAL_MODULES with the "
        "reason, and say so in ADR-0048."
    )

    retired = (
        set(WAVE_1_MODULES)
        | set(WAVE_2_MODULES)
        | set(WAVE_3A_MODULES)
        | set(WAVE_3B_MODULES)
    )
    assert len(retired) == 28, (
        f"Expected 28 retired modules across the four waves, found {len(retired)}."
    )


def test_app_local_baseline_satisfies_the_canonical_protocols() -> None:
    """P12's seam: UpstreamDrift's model is what the canonical protocols accept.

    The canonical ``strokes_gained``/``outcome_proxy`` modules type their
    ``baseline`` argument as ``ExpectedStrokesBaselineLike``, a
    ``runtime_checkable`` ``Protocol``, precisely so the already-home Tools
    loader *and* UpstreamDrift's validating model both flow in without the
    canonical package importing ``rate_of_closure``. Asserting ``isinstance``
    against the protocol is the honest form of that claim: it is what the
    structural contract actually promises, and it survives a field being
    renamed on either side in a way a hand-copied field list would not.
    """
    _require_vendored_package()

    canonical = importlib.import_module(
        "shared.python.launch_monitor.strokes_gained_types"
    )
    local = importlib.import_module(
        "src.tools.launch_monitor_model.strokes_gained_baseline"
    )

    # The baseline half is gone from the canonical module, by decision.
    assert not hasattr(canonical, "ExpectedStrokesBaselineV2")
    assert not hasattr(canonical, "baseline_table_sha256")
    assert canonical.BASELINE_CONTRACT_VERSION == local.BASELINE_CONTRACT_VERSION

    states = tuple(
        local.ExpectedStrokesStateV2(
            lie="fairway",
            context="approach",
            target="green",
            distance_yards=float(distance),
            expected_strokes=2.5 + index * 0.1,
            standard_error=0.05,
        )
        for index, distance in enumerate((100.0, 150.0))
    )
    baseline = local.ExpectedStrokesBaselineV2(
        baseline_id="parity-fixture",
        version="1",
        source_url="https://example.invalid/baseline.json",
        license="CC0-1.0",
        table_sha256=local.baseline_table_sha256(states),
        states=states,
    )

    assert isinstance(baseline, canonical.ExpectedStrokesBaselineLike)
    assert all(
        isinstance(state, canonical.ExpectedStrokesStateLike) for state in states
    )

    # And the model is a real gate, not a shape: the digest is verified.
    with pytest.raises(ValueError, match="table_sha256"):
        local.ExpectedStrokesBaselineV2(
            baseline_id="parity-fixture",
            version="1",
            source_url="https://example.invalid/baseline.json",
            license="CC0-1.0",
            table_sha256="0" * 64,
            states=states,
        )

    facade = importlib.import_module("src.tools.launch_monitor_model")
    assert "ExpectedStrokesBaselineV2" in facade.__all__
    assert "ExpectedStrokesBaselineLike" in facade.__all__


def test_wave_3b_strokes_gained_carries_the_g1_d2_named_estimand_pair() -> None:
    """G1-D2 lands as a named estimand on the request *and* on every result.

    Naming it in only one place would not be enough: a request could select
    ``shot-level-sg-trend/1`` and hand back a summary that never says so, which
    is the unlabelled-comparison hazard the decision exists to close. The
    canonical default is the session cell, and the result field is required
    with no default so a summary cannot be built without naming its estimand.
    """
    _require_vendored_package()

    types_module = importlib.import_module(
        "shared.python.launch_monitor.strokes_gained_types"
    )
    dimension = types_module.LongitudinalDimensionV1
    summary = types_module.LongitudinalSummaryV1

    assert "session-cell-sg-trend/1" in str(types_module.LongitudinalMethod)
    assert "shot-level-sg-trend/1" in str(types_module.LongitudinalMethod)
    assert dimension.model_fields["method"].default == "session-cell-sg-trend/1", (
        "The canonical default stopped being the session cell. G1-D2 makes "
        "session-cell the canonical estimand; shot-level is the preserved "
        "variant, not the default."
    )
    assert summary.model_fields["method"].is_required(), (
        "LongitudinalSummaryV1.method gained a default. Every summary must "
        "name the estimand that produced it."
    )


def test_wave_3b_player_covariation_carries_d22_and_d23() -> None:
    """D22 and D23 adopted UpstreamDrift's postures; the union adds to them.

    Neither ruling moves an UpstreamDrift number — that is the point, and it
    is why ``test_player_covariation_drift.py``'s pins are unchanged. What is
    new is that the withholding is now *explained* rather than a bare ``None``
    (D22) and that the threshold behind it is a documented constant rather
    than an anonymous literal.
    """
    _require_vendored_package()

    types_module = importlib.import_module(
        "shared.python.launch_monitor.player_covariation_types"
    )
    assert types_module.BETWEEN_PLAYER_INTERVAL_MIN_GROUPS == 5, (
        "The D22 threshold moved. Five player means, because the Fisher-z "
        "standard error is 1/sqrt(n-3): at n=4 that is exactly 1.0, and "
        "tanh(+/-1.96) then spans [-0.96, +0.96] whatever the estimate."
    )
    assert types_module.MIN_FISHER_SAMPLES == 4

    estimate_fields = types_module.AssociationEstimateV1.model_fields
    assert "interval_withheld_reason" in estimate_fields, (
        "D22 requires that a withheld interval carry a typed reason rather "
        "than reading as a silent None."
    )

    uncertainty_fields = types_module.CovariationUncertaintyV1.model_fields
    assert "between_player_interval_min_groups" in uncertainty_fields

    facade = importlib.import_module("src.tools.launch_monitor_model")
    assert "AssociationEstimateV1" in facade.__all__
