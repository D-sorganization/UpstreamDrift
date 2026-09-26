"""Tests verifying retrieval trial leakage fail-closed remediation (#10960 P1-2).

Acceptance criteria:
(a) retrieval for trial X never returns a seed built from X;
(b) a library containing the query trial raises ValueError;
(c) with a genuine other-trial entry whose descriptor differs, the residual/distance is > 0;
(d) empty library fails closed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.club_only.observation import (
    ClubObservation,
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.profiles import get_club_only_profile
from src.shared.python.motion_matching.club_only.retrieval import (
    LibraryEntry,
    RigidPlacement,
    build_observable_descriptor,
    retrieve_starting_seeds,
)
from src.shared.python.motion_matching.club_only.seeds import (
    _library_for_observation,
    geometry_content_hash,
    profile_content_hash,
)
from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
)

pytestmark = pytest.mark.unit


def _geometry_hash(obs: ClubObservation) -> str:
    return geometry_content_hash(
        club_type=obs.club_type,
        catalog_length_m=obs.catalog_length_m,
        tool_to_model_residual_m=0.0,
    )


def test_retrieval_never_returns_seed_built_from_query_trial() -> None:
    """Criterion (a): retrieval for trial X never returns a seed built from X."""
    profile = get_club_only_profile("full_body_pinocchio")
    p_hash = profile_content_hash(profile)
    for trial_id in CANONICAL_TRIAL_SHEETS:
        obs = build_calibrated_observation_fixture(trial_id)
        g_hash = _geometry_hash(obs)
        library = _library_for_observation(obs, profile, g_hash)
        assert len(library) > 0
        for entry in library:
            assert entry.descriptor.trial_id != trial_id
            assert trial_id not in entry.reference_id

        seeds = retrieve_starting_seeds(
            observation=obs,
            profile=profile,
            library=library,
            geometry_hash=g_hash,
            profile_hash=p_hash,
            max_seeds=3,
        )
        assert len(seeds) > 0
        for seed in seeds:
            assert seed.trial_id == trial_id
            assert f"tour-prior-{trial_id}" not in seed.seed_id


def test_library_containing_query_trial_raises_value_error() -> None:
    """Criterion (b): a library containing the query trial raises ValueError."""
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("reconstruction_golfer")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    desc_self = build_observable_descriptor(
        obs,
        model_id=profile.model_id,
        geometry_hash=g_hash,
    )

    entry_self_desc = LibraryEntry(
        reference_id="ref-unrelated-name",
        descriptor=desc_self,
        body_q0=np.array([0.0, 0.05, -0.02, 0.0]),
        rigid_placement=RigidPlacement.identity(),
        source_clock_times_s=obs.native_time_s.copy(),
        body_is_prior=True,
    )
    with pytest.raises(ValueError, match="[Tt]rial leakage|query trial"):
        retrieve_starting_seeds(
            observation=obs,
            profile=profile,
            library=(entry_self_desc,),
            geometry_hash=g_hash,
            profile_hash=p_hash,
        )


def test_differing_other_trial_descriptor_has_positive_residual() -> None:
    """Criterion (c): with genuine other-trial entry whose descriptor differs, residual is > 0."""
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("reconstruction_golfer")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)

    # Build other observation with genuine differences (displaced hands and face centroids)
    other_obs_base = build_calibrated_observation_fixture("TW_ProV1")
    other_mid = other_obs_base.mid_hands_xyz + np.array([0.05, -0.03, 0.02])
    other_face = other_obs_base.face_xyz + np.array([0.04, -0.02, 0.01])
    other_obs = ClubObservation(
        native_time_s=other_obs_base.native_time_s.copy(),
        mid_hands_xyz=other_mid,
        face_xyz=other_face,
        mid_hands_quat=other_obs_base.mid_hands_quat.copy(),
        face_quat=other_obs_base.face_quat.copy(),
        mask=other_obs_base.mask,
        derivation=other_obs_base.derivation,
        uncertainty=other_obs_base.uncertainty,
        events=other_obs_base.events,
        sample_rate_hz=other_obs_base.sample_rate_hz,
        club_type=other_obs_base.club_type,
        catalog_length_m=other_obs_base.catalog_length_m,
        source=other_obs_base.source,
        trial_id=other_obs_base.trial_id,
    )
    other_desc = build_observable_descriptor(
        other_obs,
        model_id=profile.model_id,
        geometry_hash=g_hash,
    )
    entry = LibraryEntry(
        reference_id=f"tour-prior-{other_obs.trial_id}-0",
        descriptor=other_desc,
        body_q0=np.array([0.0, 0.05, -0.02, 0.0]),
        rigid_placement=RigidPlacement.identity(),
        source_clock_times_s=other_obs.native_time_s.copy(),
        body_is_prior=True,
    )

    seeds = retrieve_starting_seeds(
        observation=obs,
        profile=profile,
        library=(entry,),
        geometry_hash=g_hash,
        profile_hash=p_hash,
        max_seeds=1,
    )
    assert len(seeds) == 1
    seed = seeds[0]
    assert seed.observed_residual_m > 0.0
    assert seed.seed_id == f"retrieval:{entry.reference_id}"
    assert seed.trial_id == obs.trial_id


def test_empty_library_fails_closed() -> None:
    """Criterion (d): empty library fails closed."""
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("reconstruction_golfer")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)

    with pytest.raises(ValueError, match="library must be non-empty"):
        retrieve_starting_seeds(
            observation=obs,
            profile=profile,
            library=(),
            geometry_hash=g_hash,
            profile_hash=p_hash,
        )

    # When no other-trial source is available, _library_for_observation returns empty tuple
    isolated_obs = ClubObservation(
        native_time_s=obs.native_time_s.copy(),
        mid_hands_xyz=obs.mid_hands_xyz.copy(),
        face_xyz=obs.face_xyz.copy(),
        mid_hands_quat=obs.mid_hands_quat.copy(),
        face_quat=obs.face_quat.copy(),
        mask=obs.mask,
        derivation=obs.derivation,
        uncertainty=obs.uncertainty,
        events=obs.events,
        sample_rate_hz=obs.sample_rate_hz,
        club_type=obs.club_type,
        catalog_length_m=obs.catalog_length_m,
        source=obs.source,
        trial_id="isolated_unknown_trial",
    )
    empty_lib = _library_for_observation(isolated_obs, profile, g_hash)
    assert empty_lib == ()

    with pytest.raises(ValueError, match="library must be non-empty"):
        retrieve_starting_seeds(
            observation=isolated_obs,
            profile=profile,
            library=empty_lib,
            geometry_hash=g_hash,
            profile_hash=p_hash,
        )
