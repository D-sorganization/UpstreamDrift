"""Adapters between :class:`ClubObservation` and legacy :class:`ClubTarget`.

Legacy ``ClubTarget.butt`` remains the mid-hands position stream. This adapter
never silently renames mid-hands as butt-end; an explicit offset transform is
required for butt-end geometry (see ``club_calibration.mid_hands_to_butt_end``).
"""

from __future__ import annotations

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.club_models import CLUBS, DRIVER
from src.shared.python.motion_matching.club_only.observation import (
    ClubObservation,
    ComponentMask,
    ComponentStatus,
    DerivationMetadata,
    ObservationEvent,
    UncertaintyMetadata,
)
from src.shared.python.motion_matching.club_only.workbook_identity import (
    NATIVE_SAMPLE_RATE_HZ,
)
from src.shared.python.motion_matching.club_target import (
    AlignOptions,
    ClubTarget,
)
from src.shared.python.motion_matching.loaders._align import detect_impact_index


@precondition(
    lambda obs, opts: isinstance(obs, ClubObservation),
    "obs must be a ClubObservation",
)
@postcondition(
    lambda result: isinstance(result, ClubTarget),
    "adapter must return ClubTarget",
)
def observation_to_club_target(
    obs: ClubObservation, opts: AlignOptions | None = None
) -> ClubTarget:
    """Project a fully oriented observation onto the legacy ClubTarget surface.

    Rejects unobserved face orientation rather than inventing identity quats.
    Mid-hands positions map to ``butt`` as the historical grip alias without
    claiming butt-end geometry.
    """
    _ = opts  # reserved for future resample-on-adapt; native clock preferred
    if obs.mask.face_orientation is ComponentStatus.UNOBSERVED:
        raise ValueError(
            "cannot adapt ClubObservation with unobserved face orientation "
            "to ClubTarget (missing orientation is not identity)"
        )
    if obs.mask.mid_hands_position is ComponentStatus.UNOBSERVED:
        raise ValueError("cannot adapt observation with unobserved mid-hands position")
    if obs.mask.face_position is ComponentStatus.UNOBSERVED:
        raise ValueError("cannot adapt observation with unobserved face position")
    time = np.asarray(obs.native_time_s, dtype=np.float64).copy()
    # ClubTarget requires time[0] == 0.
    time -= float(time[0])
    # Prefer authoritative "I" event from observation; fall back to detection.
    impact_event = next((e for e in obs.events if e.label == "I"), None)
    if impact_event is not None:
        impact_idx = impact_event.sample_index + 1  # convert 0-based to 1-based
    else:
        impact_idx = int(detect_impact_index(time, obs.face_xyz)) + 1
    return ClubTarget(
        time=time,
        butt=np.asarray(obs.mid_hands_xyz, dtype=np.float64).copy(),
        clubhead=np.asarray(obs.face_xyz, dtype=np.float64).copy(),
        club_quat=np.asarray(obs.face_quat, dtype=np.float64).copy(),
        impact_idx=impact_idx,
        source=obs.source,
    )


@precondition(
    lambda target, trial_id: isinstance(target, ClubTarget),
    "target must be a ClubTarget",
)
@postcondition(
    lambda result: isinstance(result, ClubObservation),
    "adapter must return ClubObservation",
)
def club_target_to_observation(
    target: ClubTarget,
    *,
    trial_id: str,
    club_type: str | None = None,
) -> ClubObservation:
    """Lift a legacy ClubTarget into ClubObservation without inventing grip quat.

    Face orientation is treated as measured. Mid-hands orientation and twist are
    unobserved (NaN), matching historical loaders that discarded grip orientation.
    """
    n = int(target.time.shape[0])
    nan_quat = np.full((n, 4), np.nan, dtype=np.float64)
    resolved_type = club_type or DRIVER.name
    if resolved_type not in CLUBS:
        raise ValueError(f"unknown club type {resolved_type!r}")
    length_m = float(CLUBS[resolved_type].length_m)
    dt = (
        float(np.median(np.diff(target.time))) if n > 1 else 1.0 / NATIVE_SAMPLE_RATE_HZ
    )
    rate = (1.0 / dt) if dt > 0 else NATIVE_SAMPLE_RATE_HZ
    # Preserve impact_idx as an "I" event (convert 1-based to 0-based).
    impact_sample_idx = int(target.impact_idx) - 1
    impact_time_s = float(target.time[impact_sample_idx])
    impact_event = ObservationEvent(
        label="I", sample_index=impact_sample_idx, time_s=impact_time_s
    )
    return ClubObservation(
        native_time_s=np.asarray(target.time, dtype=np.float64).copy(),
        mid_hands_xyz=np.asarray(target.butt, dtype=np.float64).copy(),
        face_xyz=np.asarray(target.clubhead, dtype=np.float64).copy(),
        mid_hands_quat=nan_quat,
        face_quat=np.asarray(target.club_quat, dtype=np.float64).copy(),
        mask=ComponentMask(
            mid_hands_position=ComponentStatus.MEASURED,
            face_position=ComponentStatus.MEASURED,
            mid_hands_orientation=ComponentStatus.UNOBSERVED,
            face_orientation=ComponentStatus.MEASURED,
            twist=ComponentStatus.UNOBSERVED,
        ),
        derivation=DerivationMetadata(
            orientation_axis_status="legacy_club_target",
            degenerate_axes=False,
            notes=(
                "lifted from ClubTarget; mid-hands orientation unobserved",
                "butt field retains mid-hands semantics, not butt-end",
            ),
        ),
        uncertainty=UncertaintyMetadata(
            position_sigma_m=0.0,
            orientation_sigma_rad=0.0,
        ),
        events=(impact_event,),
        sample_rate_hz=float(rate),
        club_type=resolved_type,
        catalog_length_m=length_m,
        source=target.source,
        trial_id=trial_id,
    )
