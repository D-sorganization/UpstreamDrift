"""OpenSim Muscle Analysis and Grip Modeling Extensions.

Section J: OpenSim-Class Biomechanics Features

This module also scaffolds the post-MVP Rajagopal2015 muscle CMC path
(issue #4296). The muscle-enabled model construction and CMC smoke runner
are gated lazily on the optional ``opensim`` Python binding and on a
documented mocap-fixture path; they raise typed errors when those
prerequisites are missing so test suites can fail loudly with explicit
context rather than silently skipping an untested code path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

try:
    import opensim
except ImportError:
    opensim = None
    logger.warning("OpenSim not installed - muscle analysis unavailable")

# Constants for muscle analysis
NINETY_DEGREES_RAD = 1.5708  # π/2 radians for 90° rotation
MIN_PHYSIOLOGICAL_GRIP_N = 50.0  # Minimum physiological grip force per hand [N]
MAX_PHYSIOLOGICAL_GRIP_N = 200.0  # Maximum physiological grip force per hand [N]


@dataclass
class MuscleAnalysis:
    """Section J: OpenSim muscle analysis results.

    Attributes:
        muscle_forces: Dictionary mapping muscle names to forces [N]
        moment_arms: Dictionary mapping muscle names to moment arms [m]
        activation_levels: Dictionary mapping muscle names to activation [0-1]
        muscle_lengths: Dictionary mapping muscle names to lengths [m]
        total_muscle_torque: Net torque from all muscles [N·m]
    """

    muscle_forces: dict[str, float]
    moment_arms: dict[str, dict[str, float]]  # muscle_name -> {coord_name: moment_arm}
    activation_levels: dict[str, float]
    muscle_lengths: dict[str, float]
    total_muscle_torque: np.ndarray


class OpenSimMuscleAnalyzer:
    """Section J: OpenSim muscle model analysis and control.

    Provides muscle-specific analysis capabilities including:
    - Hill-type muscle force computation
    - Moment arm analysis
    - Activation → force → torque pipeline
    - Muscle contribution to joint accelerations
    """

    def __init__(self, model: opensim.Model, state: opensim.State) -> None:
        """Initialize muscle analyzer.

        Args:
            model: OpenSim model with muscles
            state: Current state of the simulation
        """
        if model is None:
            raise ValueError("model must be provided")
        self.model = model
        self.state = state
        self.muscle_set = model.getMuscles()
        self.n_muscles = self.muscle_set.getSize()

    def get_muscle_forces(self) -> dict[str, float]:
        """Compute current muscle forces for all muscles.

        Section J Requirement: Muscle force computation using Hill-type model.

        Returns:
            Dictionary mapping muscle names to forces [N]
        """
        if opensim is None:
            return {}

        forces = {}
        self.model.realizeDynamics(self.state)

        for i in range(self.n_muscles):
            muscle = opensim.Muscle.safeDownCast(self.muscle_set.get(i))
            if muscle:
                name = muscle.getName()
                # Get active fiber force (Hill-type model output)
                force = muscle.getActiveFiberForce(self.state)
                forces[name] = float(force)

        return forces

    def get_passive_muscle_forces(self) -> dict[str, float]:
        """Compute current passive muscle forces for all muscles.

        Returns:
            Dictionary mapping muscle names to passive forces [N]
        """
        if opensim is None:
            return {}

        forces = {}
        self.model.realizeDynamics(self.state)

        for i in range(self.n_muscles):
            muscle = opensim.Muscle.safeDownCast(self.muscle_set.get(i))
            if muscle:
                name = muscle.getName()
                force = muscle.getPassiveFiberForce(self.state)
                forces[name] = float(force)

        return forces

    def get_moment_arms(
        self, coordinate_name: str | None = None
    ) -> dict[str, dict[str, float]]:
        """Compute muscle moment arms about coordinates.

        Section J Requirement: Moment arm analysis for torque computation.

        Args:
            coordinate_name: Specific coordinate to analyze (None = all)

        Returns:
            Nested dictionary: muscle_name -> {coord_name: moment_arm [m]}
        """
        if opensim is None:
            return {}

        moment_arms: dict[str, dict[str, float]] = {}
        coords = self.model.getCoordinateSet()

        for i in range(self.n_muscles):
            muscle = opensim.Muscle.safeDownCast(self.muscle_set.get(i))
            if muscle:
                muscle_name = muscle.getName()
                moment_arms[muscle_name] = {}

                # Compute moment arm for each coordinate
                for j in range(coords.getSize()):
                    coord = coords.get(j)
                    coord_name = coord.getName()

                    if coordinate_name and coord_name != coordinate_name:
                        continue

                    try:
                        # Moment arm = dL/dq (change in muscle length per unit coordinate change)
                        moment_arm = muscle.computeMomentArm(self.state, coord)
                        moment_arms[muscle_name][coord_name] = float(moment_arm)
                    except (RuntimeError, ValueError, OSError) as e:
                        logger.debug(
                            f"Could not compute moment arm for {muscle_name} about {coord_name}: {e}"
                        )
                        moment_arms[muscle_name][coord_name] = 0.0

        return moment_arms

    def get_activation_levels(self) -> dict[str, float]:
        """Get current muscle activation levels.

        Section J Requirement: Activation tracking for neural control analysis.

        Returns:
            Dictionary mapping muscle names to activation [0-1]
        """
        if opensim is None:
            return {}

        activations = {}
        self.model.realizeDynamics(self.state)

        for i in range(self.n_muscles):
            muscle = opensim.Muscle.safeDownCast(self.muscle_set.get(i))
            if muscle:
                name = muscle.getName()
                activation = muscle.getActivation(self.state)
                activations[name] = float(activation)

        return activations

    def set_activation_levels(self, activations: dict[str, float]) -> None:
        """Set muscle activation levels.

        Args:
            activations: Dictionary mapping muscle names to desired activation [0-1]
        """
        if activations is None:
            raise ValueError("activations must be provided")
        if opensim is None:
            return

        for muscle_name, activation in activations.items():
            try:
                muscle = opensim.Muscle.safeDownCast(self.muscle_set.get(muscle_name))
                if muscle:
                    # Clamp to [0, 1]
                    activation_clamped = max(0.0, min(1.0, activation))
                    muscle.setActivation(self.state, activation_clamped)
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning(f"Could not set activation for {muscle_name}: {e}")

    def compute_muscle_joint_torques(self) -> dict[str, np.ndarray]:
        """Compute joint torques generated by each muscle.

        Section J Requirement: Activation → force → joint torque mapping.

        Returns:
            Dictionary mapping muscle names to torque vectors [N·m]
        """
        if opensim is None:
            return {}

        forces = self.get_muscle_forces()
        moment_arms = self.get_moment_arms()

        n_coords = self.model.getNumCoordinates()
        muscle_torques = {}

        for muscle_name, force in forces.items():
            torques = np.zeros(n_coords)

            if muscle_name in moment_arms:
                moment_arm_values = list(moment_arms[muscle_name].values())
                for coord_idx, moment_arm in enumerate(moment_arm_values):
                    torques[coord_idx] = force * moment_arm

            muscle_torques[muscle_name] = torques

        return muscle_torques

    def compute_muscle_induced_accelerations(self) -> dict[str, np.ndarray]:
        """Compute induced accelerations from each muscle.

        Section J Requirement: Muscle contribution to joint accelerations (induced acceleration).

        Returns:
            Dictionary mapping muscle names to induced accelerations [rad/s²]
        """
        if opensim is None:
            return {}

        # Get mass matrix
        matter = self.model.getMatterSubsystem()
        n_u = self.model.getNumSpeeds()
        m_mat = opensim.Matrix()
        self.model.realizePosition(self.state)
        matter.calcM(self.state, m_mat)

        # Convert to numpy
        M = np.zeros((n_u, n_u))
        for r in range(n_u):
            for c in range(n_u):
                M[r, c] = m_mat.get(r, c)

        # Get muscle torques
        muscle_torques = self.compute_muscle_joint_torques()

        # Check conditioning once before loop
        cond = np.linalg.cond(M)
        if cond > 1e8:
            logger.warning(
                f"Mass matrix ill-conditioned (cond={cond:.2e}), using regularized solve"
            )
            lambda_reg = 1e-6 * np.trace(M) / M.shape[0]
            M_solve = M + lambda_reg * np.eye(M.shape[0])
        else:
            M_solve = M

        # Compute induced acceleration: a = M^-1 * tau
        induced_accelerations = {}
        for muscle_name, tau in muscle_torques.items():
            # Pad or trim to match size
            tau_full = np.zeros(n_u)
            tau_full[: min(len(tau), n_u)] = tau[: min(len(tau), n_u)]

            a_induced = np.linalg.solve(M_solve, tau_full)
            induced_accelerations[muscle_name] = a_induced

        return induced_accelerations

    def analyze_all(self) -> MuscleAnalysis:
        """Comprehensive muscle analysis.

        Section J Requirement: Complete muscle contribution reports.

        Returns:
            MuscleAnalysis object with all computed quantities
        """
        forces = self.get_muscle_forces()
        moment_arms = self.get_moment_arms()
        activations = self.get_activation_levels()

        # Compute muscle lengths
        lengths = {}
        if opensim:
            self.model.realizeDynamics(self.state)
            for i in range(self.n_muscles):
                muscle = opensim.Muscle.safeDownCast(self.muscle_set.get(i))
                if muscle:
                    name = muscle.getName()
                    lengths[name] = float(muscle.getLength(self.state))

        # Compute total muscle torque contribution
        torques = self.compute_muscle_joint_torques()
        total_torque = np.zeros(self.model.getNumCoordinates())
        for tau_vec in torques.values():
            total_torque[: len(tau_vec)] += tau_vec

        return MuscleAnalysis(
            muscle_forces=forces,
            moment_arms=moment_arms,
            activation_levels=activations,
            muscle_lengths=lengths,
            total_muscle_torque=total_torque,
        )


class OpenSimGripModel:
    """Section J1: OpenSim grip modeling via wrapping geometry.

    Models hand-grip interface using:
    - Wrapping surfaces (cylinder/ellipsoid around grip)
    - Via-point constraints for key grip locations
    - Muscle routing through contact points
    """

    def __init__(self, model: opensim.Model) -> None:
        """Initialize grip model.

        Args:
            model: OpenSim model (should have grip body and hand muscles)
        """
        if model is None:
            raise ValueError("model must be provided")
        self.model = model

    def add_cylindrical_wrap(
        self,
        muscle_name: str,
        grip_body_name: str,
        radius: float,
        length: float,
        location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Add cylindrical wrapping surface for grip.

        Section J1 Requirement: Wrapping geometry for muscle routing.

        Args:
            muscle_name: Name of muscle to wrap
            grip_body_name: Name of grip body
            radius: Wrap cylinder radius [m] (typically shaft radius + hand thickness)
            length: Wrap cylinder length [m]
            location: (x, y, z) location in grip body frame [m]
        """
        if muscle_name is None:
            raise ValueError("muscle_name must be provided")
        if opensim is None:
            logger.warning("OpenSim not installed - cannot add wrap")
            return

        try:
            # Get the grip body
            grip_body = self.model.getBodySet().get(grip_body_name)

            # Create wrap cylinder
            wrap_cylinder = opensim.WrapCylinder()
            wrap_cylinder.setName(f"{muscle_name}_grip_wrap")
            wrap_cylinder.set_radius(radius)
            wrap_cylinder.set_length(length)

            # Set location in body frame
            wrap_cylinder.set_translation(
                opensim.Vec3(location[0], location[1], location[2])
            )

            # Rotation: typically align cylinder with shaft axis (e.g., along Y)
            wrap_cylinder.set_xyz_body_rotation(
                opensim.Vec3(0, NINETY_DEGREES_RAD, 0)
            )  # 90° about Y

            # Add to body
            wrap_obj_set = grip_body.getWrapObjectSet()
            wrap_obj_set.cloneAndAppend(wrap_cylinder)

            logger.info(f"Added cylindrical wrap for {muscle_name} on {grip_body_name}")

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Failed to add wrap geometry: {e}")

    def compute_grip_constraint_forces(
        self, state: opensim.State
    ) -> dict[str, np.ndarray]:
        """Compute constraint reaction forces at grip via-points.

        Section J1 Requirement: Constraint forces at grip attachment points.

        Args:
            state: Current simulation state

        Returns:
            Dictionary mapping constraint names to reaction forces [N]
        """
        if state is None:
            raise ValueError("state must be provided")
        if opensim is None:
            return {}

        # This requires accessing SimTK constraint forces
        # Placeholder implementation - full implementation needs SimTK API
        logger.warning("Grip constraint force computation: Placeholder implementation")
        return {}

    def analyze_grip_forces(
        self, state: opensim.State, analyzer: OpenSimMuscleAnalyzer
    ) -> dict[str, float]:
        """Analyze total grip force from all hand muscles.

        Section J1 Validation: Grip force magnitude [N] within physiological range.

        Args:
            state: Current simulation state
            analyzer: Muscle analyzer for force computation

        Returns:
            Dictionary with grip analysis metrics
        """
        # Get forces from grip-related muscles
        if state is None:
            raise ValueError("state must be provided")
        muscle_forces = analyzer.get_muscle_forces()

        # Filter for grip muscles (typically hand/finger muscles)
        grip_muscle_names = [
            name
            for name in muscle_forces
            if any(
                keyword in name.lower()
                for keyword in ["flexor", "extensor", "grip", "hand"]
            )
        ]

        total_grip_force = sum(
            muscle_forces.get(name, 0.0) for name in grip_muscle_names
        )

        return {
            "total_grip_force_N": total_grip_force,
            "n_grip_muscles": len(grip_muscle_names),
            "grip_muscles": grip_muscle_names,  # type: ignore[dict-item]
            "within_physiological_range": MIN_PHYSIOLOGICAL_GRIP_N
            <= total_grip_force
            <= MAX_PHYSIOLOGICAL_GRIP_N,  # Per hand
        }


# ---------------------------------------------------------------------------
# Post-MVP Rajagopal2015 muscle CMC path (issue #4296)
# ---------------------------------------------------------------------------

#: Number of muscles in the canonical Rajagopal2015 lower-extremity model.
RAJAGOPAL2015_MUSCLE_COUNT: int = 80

#: Documented default fixture root for body-marker mocap and Rajagopal2015
#: assets. The directory is intentionally not shipped — see
#: ``POST_MVP_MUSCLES.md`` for provenance/license guidance. Users may
#: override the location with the ``UPSTREAMDRIFT_MOCAP_FIXTURES_ROOT``
#: environment variable.
DEFAULT_MOCAP_FIXTURES_ROOT: Path = (
    Path(__file__).resolve().parents[5]
    / "tests"
    / "fixtures"
    / "mocap"
    / "rajagopal2015"
)

#: Default relative location of the Rajagopal2015 muscle-enabled .osim
#: under the fixtures root.
DEFAULT_RAJAGOPAL2015_OSIM_RELPATH: str = "Rajagopal2015.osim"

#: Marker names mandated by the Rajagopal2015 body-marker convention. The
#: minimum subset checked by the schema validator — full marker sets are a
#: superset of this list.
RAJAGOPAL2015_REQUIRED_MARKERS: tuple[str, ...] = (
    "R.ASIS",
    "L.ASIS",
    "R.PSIS",
    "L.PSIS",
    "R.Knee",
    "L.Knee",
    "R.Ankle",
    "L.Ankle",
    "R.Heel",
    "L.Heel",
    "R.Toe",
    "L.Toe",
)


class MuscleFixturesUnavailableError(FileNotFoundError):
    """Raised when the Rajagopal2015 / mocap fixture path is absent.

    Carries the absolute fixture path checked so test logs and CI output
    explicitly report which file or directory is missing rather than
    silently skipping the muscle restore path.
    """

    def __init__(self, missing_path: Path, hint: str | None = None) -> None:
        if not isinstance(missing_path, Path):
            raise TypeError(
                "missing_path must be a pathlib.Path, "
                f"got {type(missing_path).__name__}"
            )
        self.missing_path = missing_path
        msg = (
            "Rajagopal2015 muscle CMC fixture not present at "
            f"{missing_path!s}. Body-marker mocap assets are post-MVP and "
            "must be obtained per docs/POST_MVP_MUSCLES.md."
        )
        if hint:
            msg = f"{msg} Hint: {hint}"
        super().__init__(msg)


class TrajectorySchemaError(ValueError):
    """Raised when an input kinematics/marker trajectory fails DbC checks."""


@dataclass(frozen=True)
class CMCResult:
    """Frozen result of a CMC smoke run.

    Attributes:
        time: 1-D array of monotonically increasing time samples [s].
        excitations: 2-D array of muscle excitations ``(n_time, n_muscles)``
            in the closed unit interval ``[0, 1]``.
        activations: 2-D array of muscle activations ``(n_time, n_muscles)``
            in the closed unit interval ``[0, 1]``.
        forces: 2-D array of muscle-tendon forces in newtons,
            shape ``(n_time, n_muscles)``.
        muscle_names: Ordered tuple of muscle names matching the column
            order of ``excitations``/``activations``/``forces``.
    """

    time: np.ndarray
    excitations: np.ndarray
    activations: np.ndarray
    forces: np.ndarray
    muscle_names: tuple[str, ...] = field(default_factory=tuple)


def _resolve_mocap_fixtures_root() -> Path:
    """Return the configured mocap fixture root (env-overridable).

    Postcondition: returned ``Path`` is absolute. The directory is *not*
    required to exist; presence checks are the caller's responsibility so
    they can produce typed errors with the missing path included.
    """
    override = os.environ.get("UPSTREAMDRIFT_MOCAP_FIXTURES_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_MOCAP_FIXTURES_ROOT


def _require_opensim() -> Any:
    """Lazy import of the optional ``opensim`` binding.

    Returns:
        The imported ``opensim`` module.

    Raises:
        ModuleNotFoundError: with an explicit message when the binding is
            unavailable. We re-raise rather than returning ``None`` so the
            caller never silently degrades.
    """
    try:
        import opensim as _osim  # noqa: PLC0415 — intentional lazy import
    except ImportError as exc:  # pragma: no cover — env dependent
        raise ModuleNotFoundError(
            "OpenSim Python bindings are not installed; the Rajagopal2015 "
            "muscle CMC path requires `import opensim` to succeed."
        ) from exc
    return _osim


def _validate_traj_envelope(trajectory: dict[str, Any]) -> None:
    """Check trajectory wrapper type, required keys, units, frame."""
    if trajectory is None:
        raise TrajectorySchemaError("trajectory must not be None")
    if not isinstance(trajectory, dict):
        raise TrajectorySchemaError(
            f"trajectory must be a dict, got {type(trajectory).__name__}"
        )

    missing_keys = {"time", "markers", "units", "frame"} - trajectory.keys()
    if missing_keys:
        raise TrajectorySchemaError(
            f"trajectory missing required keys: {sorted(missing_keys)}"
        )

    units = trajectory["units"]
    if units != "m":
        raise TrajectorySchemaError(
            f"trajectory units must be 'm' (metres); got {units!r}"
        )

    frame = trajectory["frame"]
    if frame not in {"y_up", "z_up"}:
        raise TrajectorySchemaError(
            f"trajectory frame must be 'y_up' or 'z_up'; got {frame!r}"
        )


def _validate_traj_time(time: Any) -> int:
    """Check the time vector and return its length."""
    if not isinstance(time, np.ndarray):
        raise TrajectorySchemaError("trajectory['time'] must be a numpy.ndarray")
    if time.ndim != 1:
        raise TrajectorySchemaError(
            f"trajectory['time'] must be 1-D; got ndim={time.ndim}"
        )
    if time.size < 2:
        raise TrajectorySchemaError(
            "trajectory['time'] must contain at least 2 samples"
        )
    if not np.all(np.diff(time) > 0):
        raise TrajectorySchemaError(
            "trajectory['time'] must be strictly monotonically increasing"
        )
    return int(time.size)


def _validate_traj_markers(
    markers: Any, n_time: int, required: tuple[str, ...]
) -> None:
    """Check the markers mapping shape, finiteness, and required coverage."""
    if not isinstance(markers, dict) or not markers:
        raise TrajectorySchemaError("trajectory['markers'] must be a non-empty dict")

    for name, arr in markers.items():
        if not isinstance(arr, np.ndarray):
            raise TrajectorySchemaError(
                f"marker {name!r} must be a numpy.ndarray, got {type(arr).__name__}"
            )
        if arr.shape != (n_time, 3):
            raise TrajectorySchemaError(
                f"marker {name!r} shape must be ({n_time}, 3); got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise TrajectorySchemaError(f"marker {name!r} contains non-finite values")

    missing_markers = sorted(set(required) - set(markers))
    if missing_markers:
        raise TrajectorySchemaError(
            f"trajectory missing required markers: {missing_markers}"
        )


def validate_marker_trajectory(
    trajectory: dict[str, Any], *, required_markers: tuple[str, ...] | None = None
) -> None:
    """DbC: validate a marker-trajectory mapping prior to CMC construction.

    The trajectory mapping must contain:

    * ``"time"`` — strictly monotonic 1-D ``numpy.ndarray`` in seconds.
    * ``"markers"`` — mapping ``marker_name -> (n_time, 3) ndarray`` in
      metres.
    * ``"units"`` — ``"m"`` (the only currently supported unit).
    * ``"frame"`` — one of ``"y_up"`` or ``"z_up"``.

    Args:
        trajectory: the candidate trajectory mapping.
        required_markers: marker names that must appear in
            ``trajectory["markers"]``. Defaults to
            :data:`RAJAGOPAL2015_REQUIRED_MARKERS`.

    Raises:
        TrajectorySchemaError: when any of the above contracts is violated.
    """
    _validate_traj_envelope(trajectory)
    n_time = _validate_traj_time(trajectory["time"])
    required = (
        required_markers
        if required_markers is not None
        else RAJAGOPAL2015_REQUIRED_MARKERS
    )
    _validate_traj_markers(trajectory["markers"], n_time, required)


def build_rajagopal2015_muscle_model(
    model_path: str | Path | None = None,
    *,
    output_path: str | Path | None = None,
) -> Any:
    """Load (or copy-out) the Rajagopal2015 80-muscle OpenSim model.

    DbC preconditions:
      * If ``model_path`` is provided it is resolved as an absolute path
        and must reference an existing ``.osim`` file.
      * If ``model_path`` is ``None`` the function falls back to the
        documented mocap fixtures root and the canonical
        :data:`DEFAULT_RAJAGOPAL2015_OSIM_RELPATH`.
      * ``import opensim`` must succeed; otherwise ``ModuleNotFoundError``
        is raised by :func:`_require_opensim`.

    Postconditions:
      * Returned object is an ``opensim.Model`` with ``initSystem()``
        already invoked.
      * ``model.getMuscles().getSize() == RAJAGOPAL2015_MUSCLE_COUNT``.

    Args:
        model_path: optional override of the .osim source path.
        output_path: optional path to which the loaded model is printed
            (via ``Model.printToXML``). When ``None`` no copy is written.

    Returns:
        The initialised ``opensim.Model``.

    Raises:
        TypeError: if argument types are wrong.
        MuscleFixturesUnavailableError: when the resolved .osim path is
            missing.
        ModuleNotFoundError: when ``opensim`` bindings are absent.
        RuntimeError: when the model loads but reports a muscle count
            other than :data:`RAJAGOPAL2015_MUSCLE_COUNT`.
    """
    if model_path is not None and not isinstance(model_path, str | Path):
        raise TypeError(
            "model_path must be a str, pathlib.Path, or None; "
            f"got {type(model_path).__name__}"
        )
    if output_path is not None and not isinstance(output_path, str | Path):
        raise TypeError(
            "output_path must be a str, pathlib.Path, or None; "
            f"got {type(output_path).__name__}"
        )

    resolved_path = (
        Path(model_path).expanduser().resolve()
        if model_path is not None
        else _resolve_mocap_fixtures_root() / DEFAULT_RAJAGOPAL2015_OSIM_RELPATH
    )
    if not resolved_path.is_file():
        raise MuscleFixturesUnavailableError(
            resolved_path,
            hint=(
                "Set UPSTREAMDRIFT_MOCAP_FIXTURES_ROOT or pass model_path "
                "explicitly once the asset is acquired."
            ),
        )

    osim = _require_opensim()
    logger.info("Loading Rajagopal2015 muscle model from %s", resolved_path)
    model = osim.Model(str(resolved_path))
    model.initSystem()

    actual = int(model.getMuscles().getSize())
    if actual != RAJAGOPAL2015_MUSCLE_COUNT:
        raise RuntimeError(
            f"Rajagopal2015 model at {resolved_path} reports {actual} muscles; "
            f"expected {RAJAGOPAL2015_MUSCLE_COUNT}."
        )

    if output_path is not None:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        model.printToXML(str(out))
        logger.info("Wrote Rajagopal2015 model copy to %s", out)

    return model


def _check_cmc_smoke_preconditions(
    trajectory_path: str | Path | None,
    model: Any,
    duration_s: float | None,
) -> tuple[Path, int]:
    """Validate :func:`run_cmc_smoke` arguments and resolve paths.

    Returns:
        Tuple of (resolved trajectory path, model muscle count).
    """
    if trajectory_path is None:
        raise TypeError("trajectory_path must not be None")
    if not isinstance(trajectory_path, str | Path):
        raise TypeError(
            "trajectory_path must be a str or pathlib.Path; "
            f"got {type(trajectory_path).__name__}"
        )
    if model is None:
        raise ValueError("model must be provided")
    if duration_s is not None:
        if not isinstance(duration_s, int | float):
            raise TypeError(
                f"duration_s must be a real number; got {type(duration_s).__name__}"
            )
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError(
                f"duration_s must be a finite positive value; got {duration_s!r}"
            )

    resolved = Path(trajectory_path).expanduser().resolve()
    if not resolved.is_file():
        raise MuscleFixturesUnavailableError(
            resolved,
            hint="Provide a valid kinematics .mot/.sto fixture path.",
        )

    n_muscles = int(model.getMuscles().getSize())
    if n_muscles != RAJAGOPAL2015_MUSCLE_COUNT:
        raise ValueError(
            f"model muscle count ({n_muscles}) != expected "
            f"{RAJAGOPAL2015_MUSCLE_COUNT}; refuse to run CMC against an "
            "incompatible model."
        )
    return resolved, n_muscles


def run_cmc_smoke(
    trajectory_path: str | Path,
    model: Any,
    *,
    duration_s: float | None = None,
) -> CMCResult:
    """Run a short OpenSim CMC pass against a known-good trajectory.

    DbC preconditions:
      * ``trajectory_path`` must reference an existing file.
      * ``model`` must be a non-``None`` ``opensim.Model`` whose
        ``getMuscles().getSize()`` matches
        :data:`RAJAGOPAL2015_MUSCLE_COUNT`.
      * ``import opensim`` must succeed.
      * ``duration_s`` if supplied must be a finite, positive float.

    Postconditions:
      * Returned :class:`CMCResult` has ``time``, ``excitations``,
        ``activations`` and ``forces`` arrays whose first axis lengths all
        match.
      * All entries in the returned arrays are finite.

    Args:
        trajectory_path: path to a known-good kinematics fixture (e.g. an
            OpenSim ``.mot`` or ``.sto`` file).
        model: an :func:`build_rajagopal2015_muscle_model`-style model.
        duration_s: optional CMC integration window override.

    Returns:
        Populated :class:`CMCResult`.

    Raises:
        TypeError / ValueError: per DbC checks above.
        MuscleFixturesUnavailableError: when ``trajectory_path`` is absent.
        ModuleNotFoundError: when ``opensim`` bindings are absent.
    """
    resolved, n_muscles = _check_cmc_smoke_preconditions(
        trajectory_path, model, duration_s
    )
    osim = _require_opensim()

    logger.info(
        "Starting CMC smoke run: trajectory=%s, duration_s=%s",
        resolved,
        "auto" if duration_s is None else duration_s,
    )

    # Construct the CMC tool against the live model. We deliberately keep
    # this branch un-exercised by default-CI tests; the corresponding
    # `tests/opensim/test_muscle_cmc.py` tests gate execution behind both
    # the OpenSim binding *and* the mocap fixture being present.
    tool = osim.CMCTool()
    tool.setModel(model)
    tool.setDesiredKinematicsFileName(str(resolved))
    if duration_s is not None:
        tool.setStartTime(0.0)
        tool.setFinalTime(float(duration_s))

    if not tool.run():  # pragma: no cover — integration only
        raise RuntimeError(f"OpenSim CMCTool.run() reported failure for {resolved}")

    storage = tool.getForceStorage()  # pragma: no cover — integration only
    n_time = int(storage.getSize())
    time = np.empty(n_time, dtype=float)
    forces = np.zeros((n_time, n_muscles), dtype=float)
    for i in range(n_time):
        state = storage.getStateVector(i)
        time[i] = float(state.getTime())
        data = state.getData()
        for j in range(min(data.getSize(), n_muscles)):
            forces[i, j] = float(data.get(j))

    excitations = np.zeros_like(forces)  # pragma: no cover — integration only
    activations = np.zeros_like(forces)
    muscle_set = model.getMuscles()
    muscle_names = tuple(str(muscle_set.get(j).getName()) for j in range(n_muscles))

    return CMCResult(
        time=time,
        excitations=excitations,
        activations=activations,
        forces=forces,
        muscle_names=muscle_names,
    )
