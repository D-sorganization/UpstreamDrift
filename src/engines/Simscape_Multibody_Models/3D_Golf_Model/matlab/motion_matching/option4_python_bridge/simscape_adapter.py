"""SimscapeAdapter — Python wrapper around the MATLAB Simscape forward sim.

Implements the Option-4 bridge described in
``option4_python_bridge/INTERFACES.md`` and ``APPROACH.md``. Only the surface
strictly required by issue #4077 is implemented here:

    SimscapeAdapter()                         — lazy MATLAB engine wrapper
    .start()                                  — start matlab.engine + addpath
    .close()                                  — quit matlab.engine
    .simulate_with_coefficients(theta)        — forward sim through MATLAB
    .target_from_xlsx(path, sheet)            — load_club_target_excel via Engine
    .compute_cost(theta, target, opts=None)   — compute_cost.m via Engine
    .get_polynomial_bounds()                  — build_coefficient_bounds via Engine

The full PhysicsEngine protocol (step/reset/compute_mass_matrix/...) is
deliberately deferred to issues #036–#040; the headline motion-matching
methods this issue requires are in scope.

Threading: not thread-safe. One adapter per process. Use a process pool for
concurrency (see SimscapeAdapterPool — issue #038).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.motion_matching.body_target import BodyTarget
from src.shared.python.motion_matching.multi_source_target import MultiSourceTarget

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataclasses mirroring the MATLAB sim_out / target structs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SimOut:
    """Flat numpy view of one Simscape simulation run.

    Mirrors the canonical struct returned by ``simulate_with_coefficients.m``.
    All time-indexed arrays share ``N`` rows along axis 0.

    Invariants (DbC):
        - all arrays share ``N`` along axis 0
        - ``time`` is monotonic non-decreasing, ``time[0] == 0``
        - ``club_quat`` rows are unit-norm to within 1e-3
        - ``solver_status`` ∈ {"success", "warning", "failed"}
    """

    time: np.ndarray  # (N,)        float64, seconds
    grip: np.ndarray  # (N, 3)      float64, metres (alias for r_butt)
    clubhead: np.ndarray  # (N, 3)  float64, metres
    club_quat: np.ndarray  # (N, 4) float64, [w x y z], unit-norm
    q: np.ndarray  # (N, n_joints)  joint angles (rad)
    qd: np.ndarray  # (N, n_joints) joint angular velocities (rad/s)
    tau: np.ndarray  # (N, n_joints) applied torques (N·m)
    omega: np.ndarray  # (N, n_joints) alias for qd, kept for cost-function clarity
    joint_names: tuple[str, ...]
    solver_status: str
    impact_idx: int  # index of max clubhead speed


@dataclass(frozen=True)
class ClubTarget:
    """Measured club 6-DOF trajectory (CLUB_IK_SPEC.md schema)."""

    time: np.ndarray  # (N,) seconds
    grip: np.ndarray  # (N, 3) metres (alias of butt)
    clubhead: np.ndarray  # (N, 3) metres
    club_quat: np.ndarray  # (N, 4) [w x y z]
    impact_idx: int


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class SimulationError(RuntimeError):
    """Raised when a Simscape simulation fails on the MATLAB side."""

    def __init__(
        self,
        message: str,
        *,
        matlab_error_id: str = "",
        matlab_traceback: str = "",
    ) -> None:
        super().__init__(message)
        self.matlab_error_id = matlab_error_id
        self.matlab_traceback = matlab_traceback


class EngineStartupError(SimulationError):
    """Raised when matlab.engine cannot start or the import fails."""


# --------------------------------------------------------------------------- #
# Helper: locate the motion_matching root so the engine can addpath() it
# --------------------------------------------------------------------------- #

_THIS = Path(__file__).resolve()
# .../matlab/motion_matching/option4_python_bridge/simscape_adapter.py
MOTION_MATCHING_ROOT = _THIS.parent.parent
MATLAB_ROOT = MOTION_MATCHING_ROOT.parent  # .../matlab
SUITE_MATLAB_FUNCTIONS = MATLAB_ROOT / "src" / "functions"


# --------------------------------------------------------------------------- #
# SimscapeAdapter
# --------------------------------------------------------------------------- #


class SimscapeAdapter:
    """Thin wrapper around ``matlab.engine.start_matlab()`` for Option 4.

    Lifecycle::

        adapter = SimscapeAdapter()           # cheap, no MATLAB started yet
        adapter.start()                       # ~10–30 s; license checkout
        sim_out = adapter.simulate_with_coefficients(theta)
        adapter.close()

    Use as a context manager to guarantee teardown::

        with SimscapeAdapter() as a:
            sim_out = a.simulate_with_coefficients(theta)

    Notes:
        - Not thread-safe. One adapter per process.
        - The MATLAB engine is started lazily on the first call that needs
          it (or explicitly via ``.start()``).
    """

    def __init__(
        self,
        *,
        rng_seed: int = 42,
        startup_args: str = "-nodesktop -nosplash",
    ) -> None:
        if not isinstance(rng_seed, int) or rng_seed < 0:
            raise ValueError("rng_seed must be a non-negative int")
        self._engine: Any | None = None
        self._rng_seed = rng_seed
        self._startup_args = startup_args
        self._motion_matching_added = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @property
    def engine(self) -> Any:
        """Return the live MATLAB engine, starting it if necessary."""
        if self._engine is None:
            self.start()
        return self._engine

    def start(self) -> None:
        """Start matlab.engine and addpath the motion_matching tree.

        Idempotent: a second call with the engine already alive is a no-op.

        Raises:
            EngineStartupError: matlab.engine cannot be imported, or
                ``start_matlab`` failed (license, install, or DLL issue).
        """
        if self._engine is not None:
            return
        try:
            import matlab.engine  # noqa: F401  (imported for the side effect of locating the package)

            from matlab import engine as _engine_mod
        except ImportError as e:  # pragma: no cover - exercised on no-MATLAB hosts
            raise EngineStartupError(
                "MATLAB Engine for Python could not be imported. Install "
                "with `python -m pip install matlabengine` and confirm the "
                "Python minor version is on the MathWorks compatibility "
                "matrix for your installed MATLAB release. See "
                "option4_python_bridge/INSTALLATION.md."
            ) from e

        try:
            self._engine = _engine_mod.start_matlab(self._startup_args)
        except Exception as e:  # pragma: no cover - depends on local MATLAB
            raise EngineStartupError(f"matlab.engine.start_matlab failed: {e}") from e

        # Add the shared motion_matching folder and the dataset_generator
        # folder so we can call simulate_with_coefficients, compute_cost,
        # build_coefficient_bounds, getPolynomialParameterInfo, etc.
        try:
            self._engine.addpath(
                self._engine.genpath(str(MOTION_MATCHING_ROOT)), nargout=0
            )
            if SUITE_MATLAB_FUNCTIONS.exists():
                self._engine.addpath(
                    self._engine.genpath(str(SUITE_MATLAB_FUNCTIONS)), nargout=0
                )
        except Exception as e:  # pragma: no cover
            self.close()
            raise EngineStartupError(f"addpath(motion_matching) failed: {e}") from e
        self._motion_matching_added = True
        logger.info("SimscapeAdapter: MATLAB engine up; motion_matching on path")

    def close(self) -> None:
        """Quit the MATLAB engine. Idempotent."""
        if self._engine is None:
            return
        try:
            self._engine.quit()
        except Exception:  # pragma: no cover - shutdown best-effort
            logger.exception("SimscapeAdapter.close(): engine.quit raised")
        finally:
            self._engine = None
            self._motion_matching_added = False

    def __enter__(self) -> SimscapeAdapter:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # Headline methods
    # ------------------------------------------------------------------ #

    def simulate_with_coefficients(self, theta: np.ndarray) -> SimOut:
        """Run one Simscape forward simulation with polynomial torques.

        Calls ``simulate_with_coefficients.m`` on the MATLAB side.

        Args:
            theta: 1-D float array, length ``n_joints * 7``, finite,
                ordered ``[A B C D E F G]`` per joint per
                ``COST_FUNCTION_SPEC.md``.

        Returns:
            ``SimOut`` with the canonical fields from
            ``simulate_with_coefficients.m``.

        Raises:
            ValueError: theta is not finite or shape is wrong.
            SimulationError: any MATLAB-side failure.
        """
        theta_np = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta_np.size == 0 or not np.all(np.isfinite(theta_np)):
            raise ValueError("theta must be a non-empty finite 1-D array")
        if theta_np.size % 7 != 0:
            raise ValueError(f"theta length {theta_np.size} is not a multiple of 7")

        eng = self.engine
        try:
            import matlab  # local import to avoid hard dependency at module import
        except ImportError as e:  # pragma: no cover
            raise EngineStartupError("matlab module unavailable") from e

        theta_m = matlab.double(theta_np.reshape(-1, 1).tolist())
        try:
            sim_out_m = eng.simulate_with_coefficients(theta_m, nargout=1)
        except Exception as e:
            raise _wrap_matlab_error(e, "simulate_with_coefficients") from e

        return _sim_out_from_matlab(sim_out_m)

    def target_from_xlsx(
        self,
        xlsx_path: str | os.PathLike[str],
        sheet_name: str,
    ) -> ClubTarget:
        """Load a measured club 6-DOF trajectory from a Wiffle xlsx.

        Calls ``load_club_target_excel.m`` via the MATLAB engine and
        marshals the result into a ``ClubTarget`` dataclass.

        Args:
            xlsx_path: filesystem path to the xlsx file.
            sheet_name: sheet name (e.g. ``"TW_ProV1"``).

        Raises:
            FileNotFoundError: xlsx_path does not exist.
            SimulationError: the MATLAB loader raised.
        """
        path = Path(os.fspath(xlsx_path))
        if not path.exists():
            raise FileNotFoundError(f"xlsx not found: {path}")

        eng = self.engine
        try:
            target_m = eng.load_club_target_excel(str(path), str(sheet_name), nargout=1)
        except Exception as e:
            raise _wrap_matlab_error(e, "load_club_target_excel") from e
        return _club_target_from_matlab(target_m)

    def compute_cost(
        self,
        theta: np.ndarray,
        target: MultiSourceTarget | ClubTarget,
        opts: dict[str, Any] | None = None,
    ) -> float:
        """Evaluate the canonical cost function for one theta against a target.

        Delegates to ``compute_cost.m`` on the MATLAB side, using
        ``simulate_with_coefficients`` as the forward callback.

        Args:
            theta: coefficient vector (see ``simulate_with_coefficients``).
            target: measured trajectory.
            opts: optional overrides on top of ``default_cost_options()``.

        Returns:
            The scalar cost ``J`` as a Python float.

        Raises:
            SimulationError: cost evaluation failed on the MATLAB side.
        """
        theta_np = np.asarray(theta, dtype=np.float64).reshape(-1)
        eng = self.engine
        try:
            import matlab
        except ImportError as e:  # pragma: no cover
            raise EngineStartupError("matlab module unavailable") from e

        theta_m = matlab.double(theta_np.reshape(-1, 1).tolist())

        import os

        tmp_file = None
        try:
            if hasattr(target, "club"):  # MultiSourceTarget
                target_m = _club_target_to_matlab(target.club, matlab)
                if hasattr(target, "body") and target.body is not None:
                    tmp_file = _body_target_to_json_file(target.body)
                    body_m = eng.load_body_target_json(tmp_file, nargout=1)
                    target_m["body"] = body_m
            else:
                target_m = _club_target_to_matlab(target, matlab)

            # default opts come from MATLAB side; override fields if requested.
            cost_opts = eng.default_cost_options(nargout=1)
            if opts:
                for key, value in opts.items():
                    cost_opts[key] = value
            sim_handle = eng.eval("@simulate_with_coefficients", nargout=1)
            J, _terms = eng.compute_cost(
                theta_m, target_m, sim_handle, cost_opts, nargout=2
            )
        except Exception as e:
            raise _wrap_matlab_error(e, "compute_cost") from e
        finally:
            if tmp_file is not None and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass

        return float(J)

    # Per-joint bound magnitudes from build_coefficient_bounds.m.
    # Coefficient ordering [A B C D E F G] (t^6, t^5, t^4, t^3, t^2, t^1, 1).
    _PER_JOINT_BOUND = np.array(
        [1000.0, 1000.0, 500.0, 500.0, 100.0, 100.0, 25.0],
        dtype=np.float64,
    )

    def get_polynomial_bounds(self, n_joints: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lb, ub)`` matching ``build_coefficient_bounds.m``.

        Each is a 1-D float array of length ``n_joints * 7``. Mirrors the
        MATLAB function so callers do not depend on a private/private-folder
        function being on the engine's path.
        """
        if not isinstance(n_joints, int) or n_joints <= 0:
            raise ValueError("n_joints must be a positive int")
        ub = np.tile(self._PER_JOINT_BOUND, n_joints)
        lb = -ub
        return lb, ub

    def get_n_joints(self, default: int | None = None) -> int:
        """Return ``n_joints`` from ``getPolynomialParameterInfo``.

        Args:
            default: fallback ``n_joints`` to return when MATLAB cannot
                resolve the parameter info (typical when
                ``PolynomialInputValues.mat`` is not on the MATLAB path).
                When ``None`` and the MATLAB call fails, the error is
                surfaced as :class:`SimulationError`.
        """
        eng = self.engine
        try:
            info = eng.getPolynomialParameterInfo(nargout=1)
        except Exception as e:
            if default is not None:
                logger.warning(
                    "getPolynomialParameterInfo failed (%s); using default n_joints=%d",
                    e,
                    default,
                )
                return int(default)
            raise _wrap_matlab_error(e, "getPolynomialParameterInfo") from e
        try:
            total = int(info["total_params"])
        except Exception:  # noqa: BLE001
            joint_names = info["joint_names"]
            total = len(joint_names) * 7
        return total // 7


# --------------------------------------------------------------------------- #
# Marshalling helpers
# --------------------------------------------------------------------------- #


def _np(x: Any) -> np.ndarray:
    """Coerce a MATLAB array (matlab.double) to numpy float64."""
    arr = np.array(x, dtype=np.float64)
    return arr


def _sim_out_from_matlab(sim_out_m: Any) -> SimOut:
    """Convert a MATLAB struct (returned by simulate_with_coefficients.m) to
    the Python ``SimOut`` dataclass.
    """
    # MATLAB structs surface as dict-like in matlab.engine.
    time = _np(sim_out_m["time"]).reshape(-1)
    # The MATLAB struct uses r_butt / r_clubhead / q_club; expose as
    # grip / clubhead / club_quat for cost-function-friendly naming.
    grip = _np(sim_out_m["r_butt"]).reshape(-1, 3)
    clubhead = _np(sim_out_m["r_clubhead"]).reshape(-1, 3)
    club_quat = _np(sim_out_m["q_club"]).reshape(-1, 4)
    q = _np(sim_out_m["q"])
    qd = _np(sim_out_m["qd"])
    tau = _np(sim_out_m["tau"])
    omega = _np(sim_out_m["omega"])
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    if qd.ndim == 1:
        qd = qd.reshape(-1, 1)
    if tau.ndim == 1:
        tau = tau.reshape(-1, 1)
    if omega.ndim == 1:
        omega = omega.reshape(-1, 1)

    joint_names_raw = sim_out_m.get("joint_names", [])
    joint_names = tuple(
        str(n) for n in (joint_names_raw if joint_names_raw is not None else ())
    )

    solver_status = str(sim_out_m.get("solver_status", "success"))

    # impact_idx may be absent on the MATLAB side; derive from clubhead speed.
    impact_idx = sim_out_m.get("impact_idx", None)
    if impact_idx is None:
        if clubhead.shape[0] >= 2:
            speed = np.linalg.norm(np.diff(clubhead, axis=0), axis=1)
            impact_idx = int(np.argmax(speed))
        else:
            impact_idx = 0
    else:
        impact_idx = int(impact_idx) - 1  # MATLAB 1-based to Python 0-based

    return SimOut(
        time=time,
        grip=grip,
        clubhead=clubhead,
        club_quat=club_quat,
        q=q,
        qd=qd,
        tau=tau,
        omega=omega,
        joint_names=joint_names,
        solver_status=solver_status,
        impact_idx=impact_idx,
    )


def _club_target_from_matlab(target_m: Any) -> ClubTarget:
    """Convert a MATLAB target struct to ``ClubTarget``."""
    time = _np(target_m["time"]).reshape(-1)
    # The MATLAB schema names the grip slot `butt` (rigid grip = mid-hands).
    grip_field = "grip" if "grip" in target_m else "butt"
    grip = _np(target_m[grip_field]).reshape(-1, 3)
    clubhead = _np(target_m["clubhead"]).reshape(-1, 3)
    club_quat = _np(target_m["club_quat"]).reshape(-1, 4)
    impact_idx = int(target_m.get("impact_idx", 1)) - 1
    return ClubTarget(
        time=time,
        grip=grip,
        clubhead=clubhead,
        club_quat=club_quat,
        impact_idx=impact_idx,
    )


def _body_target_to_json_file(body: BodyTarget) -> str:
    """Serialize a BodyTarget to a temporary body_target_json_v1 file."""
    import json
    import tempfile

    events = [
        {"label": ev.label, "frame": int(ev.frame), "time_s": float(ev.time_s)}
        for ev in body.events
    ]
    src = body.source
    source = {
        "filename": src.filename,
        "format": src.format,
        "subject_id": src.subject_id,
        "trial_id": src.trial_id,
        "sha256": src.sha256,
    }

    payload = {
        "schema": "body_target_json_v1",
        "time_s": body.time.tolist(),
        "marker_names": list(body.marker_names),
        "marker_xyz": body.marker_xyz.tolist(),
        "impact_idx": int(body.impact_idx),
        "events": events,
        "source": source,
        "coordinate_frame": body.coordinate_frame,
    }

    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w", encoding="utf-8"
    ) as fh:
        json.dump(payload, fh)
        return fh.name


def _club_target_to_matlab(target: ClubTarget, matlab: Any) -> Any:
    """Convert a Python ``ClubTarget`` into a MATLAB struct dict."""
    return {
        "time": matlab.double(target.time.reshape(-1, 1).tolist()),
        "butt": matlab.double(target.butt.tolist()),
        "grip": matlab.double(target.butt.tolist()),
        "clubhead": matlab.double(target.clubhead.tolist()),
        "club_quat": matlab.double(target.club_quat.tolist()),
        "impact_idx": float(target.impact_idx + 1),  # 1-based for MATLAB
    }


def _wrap_matlab_error(e: Exception, where: str) -> SimulationError:
    """Wrap a matlab.engine error into a SimulationError."""
    msg = f"{where} failed: {e}"
    matlab_error_id = ""
    matlab_traceback = ""
    # matlab.engine.MatlabExecutionError exposes `.args[0]` as the message
    if hasattr(e, "args") and e.args:
        matlab_traceback = str(e.args[0])
    return SimulationError(
        msg,
        matlab_error_id=matlab_error_id,
        matlab_traceback=matlab_traceback,
    )
