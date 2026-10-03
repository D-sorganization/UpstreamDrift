"""Load a whole OpenCap session: markers, scaled model, IK and subject (#11403).

:class:`OpenCapSessionAdapter` turns one trial's augmented markers into
canonical observations. A session holds more than that: OpenCap has already
scaled its LaiUhlrich2022 OpenSim model to the subject and run inverse
kinematics. :func:`load_opencap_session` returns all of it for one trial.

Kinematics are converted to SI using the scaled model to tell rotations
(degrees -> radians) from translations (metres, unchanged). When the session
has no scaled model, the kinematics are left out and ``notes`` says why;
loading them with a guessed unit would silently corrupt the pelvis
translation.

OpenCap kinematics come from a learned marker augmenter followed by IK, so
they are model-conditioned estimates, not observed 3-D motion (ADR-0041).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    CanonicalObservations,
    MotionTrajectory,
)
from src.shared.python.motion_pipeline.sources.opencap_adapter import (
    OpenCapSessionAdapter,
)
from src.shared.python.motion_pipeline.sources.opencap_layout import (
    OpenCapSessionLayout,
    OpenCapSubject,
)
from src.shared.python.motion_pipeline.sources.osim_coordinates import (
    CoordinateKind,
    read_osim_coordinates,
)
from src.shared.python.motion_pipeline.sources.sto_mot_adapter import (
    OpenSimSTOMOTAdapter,
)


@dataclass(frozen=True)
class OpenCapSession:
    """One trial of an OpenCap session.

    Attributes:
        trial: Trial name (the ``MarkerData/<trial>.trc`` stem).
        trials: Every trial in the session, including ``neutral``.
        observations: Augmented markers plus detector keypoints, in metres.
        kinematics: IK coordinates in SI units, one column per model
            coordinate, or ``None`` when unavailable (see ``notes``).
        model_file: OpenCap's scaled ``.osim`` model, if the session has one.
        subject: Mass, height and model name from ``sessionMetadata``.
        notes: Human-readable reasons for anything left out.
    """

    trial: str
    trials: list[str]
    observations: CanonicalObservations
    kinematics: MotionTrajectory | None
    model_file: Path | None
    subject: OpenCapSubject
    notes: tuple[str, ...] = ()


def _load_kinematics(mot_file: Path, model_file: Path) -> MotionTrajectory:
    kinds = read_osim_coordinates(model_file)
    translational = {
        name for name, kind in kinds.items() if kind is CoordinateKind.TRANSLATIONAL
    }
    motion = OpenSimSTOMOTAdapter(translational).load_checked(mot_file)
    if not isinstance(motion, MotionTrajectory):
        raise TypeError("OpenCap kinematics import expected a MotionTrajectory")
    unknown = [c for c in motion.skeleton.joints if c not in kinds]
    if unknown:
        raise ValueError(
            f"OpenCap kinematics {mot_file.name} has columns not in the scaled "
            f"model {model_file.name}: {', '.join(unknown)}"
        )
    motion.trajectory.metadata["opensim_model"] = str(model_file)
    return motion


def load_opencap_session(
    session_dir: Path,
    trial: str | None = None,
    calibration: Calibration | None = None,
) -> OpenCapSession:
    """Load one trial of an OpenCap session directory.

    Args:
        session_dir: Root of the session (holds ``MarkerData/``).
        trial: Trial to load. May be omitted when the session has exactly one
            trial other than ``neutral``.
        calibration: Passed through to the marker parser.

    Raises:
        FileNotFoundError: the session has no marker data.
        KeyError: ``trial`` is not in the session.
        ValueError: ``trial`` is ambiguous, the subject metadata is invalid,
            or the kinematics name coordinates the scaled model lacks.
    """
    layout = OpenCapSessionLayout.discover(Path(session_dir))
    chosen = layout.resolve_trial(trial)
    subject = layout.subject
    observations = OpenCapSessionAdapter().load_trial(
        layout.marker_files[chosen], layout.session_dir, calibration=calibration
    )
    model_file = layout.model_file()
    mot_file = layout.kinematics_files.get(chosen)
    notes: list[str] = []
    kinematics = None
    if mot_file is not None and model_file is None:
        notes.append(
            f"Kinematics {mot_file.name} skipped: the session has no scaled "
            "model to tell rotations from translations"
        )
    elif mot_file is not None and model_file is not None:
        kinematics = _load_kinematics(mot_file, model_file)
    return OpenCapSession(
        trial=chosen,
        trials=layout.trials,
        observations=observations,
        kinematics=kinematics,
        model_file=model_file,
        subject=subject,
        notes=tuple(notes),
    )
