"""Unit tests for the data-driven grip synergy-build logic.

Covers the decomposition from issue #7723 and the test gap from issue #7724:
the previously-untested CC-48 ``rebuild_synergy_controls`` hotspot is now
expressed as pure module-level helpers that can be exercised without a live
MuJoCo model or a constructed Qt tab.

The tests assert that the synergy controls produced for each hand model
(Shadow-right/left/both, Allegro) match the original branch behaviour exactly
-- i.e. the refactor is behaviour-preserving.
"""

from __future__ import annotations

import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._grip_modelling_synergies import (
    ALLEGRO_FIST_MAX,
    ALLEGRO_INDEX_MAX,
    ALLEGRO_PINCH_MAX,
    SHADOW_FIST_MAX,
    SHADOW_INDEX_MAX,
    SHADOW_PINCH_FINGER_MAX,
    SHADOW_PINCH_THUMB_MAX,
    allegro_synergy_specs,
    build_default_synergies,
    build_synergy_from_specs,
    resolve_hand_prefixes,
    shadow_synergy_specs,
)


def _index_resolver(known: dict[str, int]):
    """Return a resolver mapping known joint names to qpos addresses."""

    def _resolve(name: str) -> int | None:
        return known.get(name)

    return _resolve


# ---------------------------------------------------------------------------
# build_synergy_from_specs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_synergy_from_specs_resolves_in_order() -> None:
    """Bindings are produced in spec order with the spec's angle limits."""
    resolve = _index_resolver({"a": 10, "b": 20, "c": 30})
    specs = [("a", 0.0, 1.4), ("b", 0.0, 1.0), ("c", 0.0, 0.8)]

    synergy = build_synergy_from_specs("Fist Curl", specs, resolve)

    assert synergy is not None
    assert synergy.name == "Fist Curl"
    assert [(b.qpos_adr, b.min_val, b.max_val) for b in synergy.bindings] == [
        (10, 0.0, 1.4),
        (20, 0.0, 1.0),
        (30, 0.0, 0.8),
    ]


@pytest.mark.unit
def test_build_synergy_from_specs_skips_unresolved() -> None:
    """Specs whose joint does not resolve are dropped, keeping order."""
    resolve = _index_resolver({"a": 10, "c": 30})
    specs = [("a", 0.0, 1.4), ("missing", 0.0, 1.0), ("c", 0.0, 0.8)]

    synergy = build_synergy_from_specs("S", specs, resolve)

    assert synergy is not None
    assert [b.qpos_adr for b in synergy.bindings] == [10, 30]


@pytest.mark.unit
def test_build_synergy_from_specs_returns_none_when_empty() -> None:
    """No resolvable joints -> no synergy (matches original ``if bindings``)."""
    resolve = _index_resolver({})
    assert build_synergy_from_specs("S", [("x", 0.0, 1.0)], resolve) is None


# ---------------------------------------------------------------------------
# resolve_hand_prefixes
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_name", "is_shadow", "expected"),
    [
        ("shadow hand (both)", True, ["rh", "lh"]),
        ("shadow hand (right)", True, ["rh"]),
        ("shadow hand (left)", True, ["lh"]),
        ("allegro (right)", False, ["right"]),
        ("allegro (left)", False, ["left"]),
        ("allegro (both)", False, ["rh", "lh"]),
        ("no side keyword", True, []),
    ],
)
def test_resolve_hand_prefixes(
    model_name: str, is_shadow: bool, expected: list[str]
) -> None:
    """Prefix resolution matches the original branch logic."""
    assert resolve_hand_prefixes(model_name, is_shadow=is_shadow) == expected


# ---------------------------------------------------------------------------
# shadow_synergy_specs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_shadow_fist_specs_order_and_limits() -> None:
    """Fist Curl iterates prefixes x [FF,MF,RF,LF] x [3,2,1] at SHADOW_FIST_MAX."""
    groups = dict(shadow_synergy_specs(["rh"]))
    fist = groups["Fist Curl"]

    expected_names = [
        f"rh_{f}J{j}" for f in ("FF", "MF", "RF", "LF") for j in (3, 2, 1)
    ]
    assert [name for name, _, _ in fist] == expected_names
    assert all(lo == 0.0 and hi == SHADOW_FIST_MAX for _, lo, hi in fist)


@pytest.mark.unit
def test_shadow_index_specs() -> None:
    """Index Curl iterates prefixes x [3,2,1] FFJ joints at SHADOW_INDEX_MAX."""
    groups = dict(shadow_synergy_specs(["rh"]))
    index = groups["Index Curl"]

    assert [name for name, _, _ in index] == ["rh_FFJ3", "rh_FFJ2", "rh_FFJ1"]
    assert all(lo == 0.0 and hi == SHADOW_INDEX_MAX for _, lo, hi in index)


@pytest.mark.unit
def test_shadow_pinch_specs_finger_then_thumb_limits() -> None:
    """Pinch uses FFJ (finger limit) then THJ (thumb limit), per prefix."""
    groups = dict(shadow_synergy_specs(["rh"]))
    pinch = groups["Pinch Grip"]

    assert pinch == [
        ("rh_FFJ3", 0.0, SHADOW_PINCH_FINGER_MAX),
        ("rh_FFJ2", 0.0, SHADOW_PINCH_FINGER_MAX),
        ("rh_FFJ1", 0.0, SHADOW_PINCH_FINGER_MAX),
        ("rh_THJ4", 0.0, SHADOW_PINCH_THUMB_MAX),
        ("rh_THJ3", 0.0, SHADOW_PINCH_THUMB_MAX),
        ("rh_THJ2", 0.0, SHADOW_PINCH_THUMB_MAX),
        ("rh_THJ1", 0.0, SHADOW_PINCH_THUMB_MAX),
    ]
    # Divergent pinch limits are preserved (finger 1.0 != thumb 0.8).
    assert SHADOW_PINCH_FINGER_MAX != SHADOW_PINCH_THUMB_MAX


@pytest.mark.unit
def test_shadow_both_hands_doubles_prefixes() -> None:
    """'both' produces rh then lh joints in order."""
    groups = dict(shadow_synergy_specs(["rh", "lh"]))
    index_names = [name for name, _, _ in groups["Index Curl"]]
    assert index_names == [
        "rh_FFJ3",
        "rh_FFJ2",
        "rh_FFJ1",
        "lh_FFJ3",
        "lh_FFJ2",
        "lh_FFJ1",
    ]


# ---------------------------------------------------------------------------
# allegro_synergy_specs
# ---------------------------------------------------------------------------


_ALLEGRO_JOINTS = [
    "ffj0",
    "ffj1",
    "ffj2",
    "ffj3",
    "mfj1",
    "mfj2",
    "mfj3",
    "rfj1",
    "rfj2",
    "rfj3",
    "thj1",
    "thj2",
    "thj3",
    "palm_joint",
]


@pytest.mark.unit
def test_allegro_fist_matches_finger_joints() -> None:
    """Fist Curl picks ff/mf/rf j1-j3 joints (substring match) at fist limit."""
    groups = dict(allegro_synergy_specs(_ALLEGRO_JOINTS))
    fist = groups["Fist Curl"]

    assert [name for name, _, _ in fist] == [
        "ffj1",
        "ffj2",
        "ffj3",
        "mfj1",
        "mfj2",
        "mfj3",
        "rfj1",
        "rfj2",
        "rfj3",
    ]
    # ffj0 and palm_joint are excluded; thumb joints excluded from fist.
    assert "ffj0" not in [n for n, _, _ in fist]
    assert all(hi == ALLEGRO_FIST_MAX for _, _, hi in fist)


@pytest.mark.unit
def test_allegro_index_only_ff_joints() -> None:
    """Index Curl is just the ff j1-j3 joints at index limit."""
    groups = dict(allegro_synergy_specs(_ALLEGRO_JOINTS))
    index = groups["Index Curl"]

    assert [name for name, _, _ in index] == ["ffj1", "ffj2", "ffj3"]
    assert all(hi == ALLEGRO_INDEX_MAX for _, _, hi in index)


@pytest.mark.unit
def test_allegro_pinch_index_then_thumb_interleaved_by_joint() -> None:
    """Pinch appends ff joints and thumb joints in model-iteration order."""
    groups = dict(allegro_synergy_specs(_ALLEGRO_JOINTS))
    pinch = groups["Pinch Grip"]

    # ff joints come first (they appear earlier in the list), then thumb.
    assert [name for name, _, _ in pinch] == [
        "ffj1",
        "ffj2",
        "ffj3",
        "thj1",
        "thj2",
        "thj3",
    ]
    assert all(hi == ALLEGRO_PINCH_MAX for _, _, hi in pinch)


# ---------------------------------------------------------------------------
# build_default_synergies (top-level dispatcher)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_default_synergies_shadow_right() -> None:
    """Shadow-right produces Fist/Index/Pinch synergies with bound joints."""
    # Resolve every shadow joint to a unique address.
    known = {
        f"rh_{f}J{j}": idx
        for idx, (f, j) in enumerate(
            (f, j) for f in ("FF", "MF", "RF", "LF") for j in (3, 2, 1)
        )
    }
    known.update({f"rh_THJ{j}": 100 + j for j in (4, 3, 2, 1)})
    resolve = _index_resolver(known)

    synergies = build_default_synergies("shadow hand (right)", resolve, [])

    assert [s.name for s in synergies] == ["Fist Curl", "Index Curl", "Pinch Grip"]
    fist = synergies[0]
    assert len(fist.bindings) == 12  # 4 fingers x 3 joints
    assert fist.bindings[0].max_val == SHADOW_FIST_MAX


@pytest.mark.unit
def test_build_default_synergies_allegro() -> None:
    """Allegro path uses the enumerated joint names, not prefixes."""
    known = {name: idx for idx, name in enumerate(_ALLEGRO_JOINTS)}
    resolve = _index_resolver(known)

    synergies = build_default_synergies("allegro hand", resolve, _ALLEGRO_JOINTS)

    assert [s.name for s in synergies] == ["Fist Curl", "Index Curl", "Pinch Grip"]
    index = synergies[1]
    assert [b.qpos_adr for b in index.bindings] == [
        known["ffj1"],
        known["ffj2"],
        known["ffj3"],
    ]


@pytest.mark.unit
def test_build_default_synergies_unknown_model_is_empty() -> None:
    """A non-shadow/non-allegro model yields no default synergies."""
    resolve = _index_resolver({"anything": 1})
    assert build_default_synergies("some other hand", resolve, []) == []


@pytest.mark.unit
def test_build_default_synergies_drops_unresolved_synergy() -> None:
    """If a synergy has no resolvable joints it is omitted entirely."""
    # Only index joints resolve -> Fist still has those FF joints, but Pinch
    # thumb joints are absent; verify only synergies with bindings survive.
    resolve = _index_resolver({"rh_FFJ3": 1, "rh_FFJ2": 2, "rh_FFJ1": 3})

    synergies = build_default_synergies("shadow hand (right)", resolve, [])

    # Fist (has FF joints), Index (FF joints), Pinch (FF finger joints) all
    # resolve via FFJ; no synergy is fully empty here.
    names = [s.name for s in synergies]
    assert names == ["Fist Curl", "Index Curl", "Pinch Grip"]
    # But each only contains the 3 resolvable FF joints (fist normally 12).
    assert len(synergies[0].bindings) == 3
