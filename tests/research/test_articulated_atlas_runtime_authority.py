"""Runtime contracts for injecting governed authorities into headline atlases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
    scientific_model_sha256,
)
from scripts.research.proximal_distal_energy.articulated_atlas_runtime_authority import (
    build_atlas_case_model,
    resolve_atlas_states,
    runtime_authority_provenance,
)
from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    load_scaled_authority,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
pytestmark = pytest.mark.scientific


def _load(corner_id: str) -> ArticulatedAtlasAuthority:
    scaled = load_scaled_authority(
        DATA / f"articulated_structural_authority_{corner_id}.json",
        DATA / f"articulated_structural_authority_{corner_id}.npz",
    )
    return ArticulatedAtlasAuthority.from_scaled(scaled)


def test_runtime_selects_feasible_states_and_retains_registered_failure() -> None:
    authority = _load("height_scale_low")

    selection = resolve_atlas_states(authority, (0, 8, 9, 17), (0, 6, 12))

    assert len(selection.planned_states) == 12
    assert len(selection.feasible_states) == 11
    assert selection.retained_failures == (
        {"case_index": 0, "phase_index": 12, "failure_class": "ik_nonconvergence"},
    )
    assert (0, 12) not in selection.feasible_states
    assert set(selection.planned_states) == set(selection.feasible_states) | {(0, 12)}


def test_runtime_builds_and_revalidates_exact_scaled_case_model() -> None:
    authority = _load("height_scale_high")

    resolved = build_atlas_case_model(authority, 8)

    assert resolved.metadata["profile"] == asdict(authority.profile_for_case(8))
    assert resolved.model_sha256 == scientific_model_sha256(resolved.model)
    assert resolved.model_sha256 == authority.model_hashes()["8"]


def test_runtime_provenance_is_digest_bound_and_detached() -> None:
    authority = _load("nominal")

    first = runtime_authority_provenance(authority)
    first["scales"]["height"] = -1.0
    second = runtime_authority_provenance(authority)

    assert second["authority_sha256"] == authority.authority_sha256
    assert second["scales"]["height"] == 1.0
    assert second["retained_failures"] == []


@dataclass(frozen=True)
class _UngovernedAuthority:
    time_s: np.ndarray
    profile_index: np.ndarray
    grip_span_m: np.ndarray
    solution_q: np.ndarray


def test_runtime_rejects_untyped_authority_instead_of_duck_typing() -> None:
    authority = _UngovernedAuthority(
        time_s=np.arange(13, dtype=float),
        profile_index=np.zeros(18, dtype=int),
        grip_span_m=np.ones(18, dtype=float),
        solution_q=np.zeros((18, 13, 20), dtype=float),
    )

    with pytest.raises(TypeError, match="ArticulatedAtlasAuthority"):
        resolve_atlas_states(authority, (0,), (0,))
    with pytest.raises(TypeError, match="ArticulatedAtlasAuthority"):
        build_atlas_case_model(authority, 0)
    with pytest.raises(TypeError, match="ArticulatedAtlasAuthority"):
        runtime_authority_provenance(authority)


@pytest.mark.parametrize(
    ("case_indices", "sample_indices", "message"),
    [
        ((0, 0), (0,), "case_indices"),
        ((0,), (0, 0), "sample_indices"),
        ((1,), (0,), "selected case"),
        ((0,), (13,), "sample_indices"),
    ],
)
def test_runtime_rejects_invalid_or_unregistered_designs(
    case_indices: tuple[int, ...],
    sample_indices: tuple[int, ...],
    message: str,
) -> None:
    authority = _load("nominal")

    with pytest.raises((ValueError, TypeError), match=message):
        resolve_atlas_states(authority, case_indices, sample_indices)
