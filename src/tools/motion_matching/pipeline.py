"""Command construction and receipt summaries for the Motion Matching tool.

The matching pipeline lives in the evidence driver
``docs/development/full_body_models/evidence/ground_support/run_ground_support.py``
and the document builder ``docs/development/full_body_models/build_anthropometric_spec.py``;
this module knows how to call them for a capture and a club and how to read
back the receipt, so the GUI (``gui.py``) and any script share one path
through the pipeline. Pure functions, no Qt, tested without running the
pipeline. Epic #10113, child #10106.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.motion_matching.execution import (
    assets,
    downswing as _downswing,
    driver as _driver,
    mjx_export as _mjx_export,
    spec_builder as _spec_builder,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FULL_BODY = REPO_ROOT / "docs/development/full_body_models"

BUILDER = Path(_spec_builder.__file__).resolve()
DRIVER_SCRIPT = Path(_driver.__file__).resolve()
DOWNSWING_SCRIPT = Path(_downswing.__file__).resolve()
EXPORT_MJX_SCRIPT = Path(_mjx_export.__file__).resolve()


def resolve_native_spec() -> Path:
    return assets.get_native_geometry_spec()


def resolve_osim_model() -> Path:
    return assets.get_opensim_model()


def resolve_candidate_spec() -> Path:
    return assets.get_candidate_geometry_spec()


try:
    NATIVE = assets.get_native_geometry_spec()
except FileNotFoundError:
    NATIVE = (
        REPO_ROOT
        / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
    )

try:
    OSIM = assets.get_opensim_model()
except FileNotFoundError:
    OSIM = REPO_ROOT / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"

try:
    CANDIDATE = assets.get_candidate_geometry_spec()
except FileNotFoundError:
    CANDIDATE = FULL_BODY / "evidence/native_candidates/returned81_candidate.json"

CAPTURES = ("driver", "iron")
CLUBS = ("driver", "iron7")
CLUB_FOR_CAPTURE = {"driver": "driver", "iron": "iron7"}
BACKENDS = ("mujoco", "drake", "pinocchio", "opensim", "pink")
IK_BACKENDS = ("lm", "mujoco-minimize")
TRACKING_BACKENDS = ("kkt", "mj-inverse")
STEP_MODES = ("physical", "projection")
SOLVERS = ("quadprog",)


def available_engines() -> list[str]:
    """Registered physics engines available to the motion-matching plant."""
    try:
        from src.shared.python.motion_matching.pipeline.plant import (
            available_engines as _avail,
        )

        return _avail()
    except (ImportError, RuntimeError, TypeError, AttributeError, KeyError):
        return ["mujoco", "drake", "pinocchio"]


def extract_five_metrics_and_acceptance(
    summary: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Extract standard five kinematic metrics and determine acceptance verdict.

    Returns:
        tuple of (metrics_dict, acceptance_verdict)
    """
    metrics = {
        "full_capture_ik_rms_mm": summary.get("full_capture_ik_rms_mm"),
        "address_marker_rms_mm": summary.get("address_marker_rms_mm"),
        "backswing_root_error_max_mm": summary.get("backswing_root_error_max_mm"),
        "whole_run_root_rms_mm": summary.get("whole_run_root_rms_mm"),
        "inside_support_polygon_fraction": summary.get(
            "inside_support_polygon_fraction"
        ),
    }
    is_qual = summary.get("is_qualified")
    converged = summary.get("all_frames_converged")
    if is_qual is True and (converged is True or converged is None):
        verdict = "PASSED"
    elif is_qual is False or converged is False:
        verdict = "REJECTED"
    else:
        verdict = "UNCLASSIFIED"
    return metrics, verdict


@dataclass(frozen=True)
class MatchRequest:
    """One matching run: which capture, which club, which subject."""

    capture: str
    club: str
    stature_m: float = 1.71
    mass_kg: float = 78.0
    trunk_scale: float = 1.15
    arm_scale: float = 1.10
    shoulder_scale: float = 1.0
    output_name: str | None = None
    free_wrists: bool = False
    bound_wrists: bool = False
    fit_closure: bool = False
    zmp_filter: bool = False
    shooting_fit: int = 0
    shooting_gain: float = 0.7
    cutoff_hz: float | None = None
    backend: str = "mujoco"
    ik_backend: str = "lm"
    tracking: str = "kkt"
    step_mode: str = "physical"
    solver: str = "quadprog"
    output_root: Path | str | None = None
    native_path: Path | str | None = None
    osim_path: Path | str | None = None
    candidate_path: Path | str | None = None

    def __post_init__(self) -> None:
        if self.capture not in CAPTURES:
            raise ValueError(f"Capture must be one of {CAPTURES}")
        if self.club not in CLUBS:
            raise ValueError(f"Club must be one of {CLUBS}")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if self.ik_backend not in IK_BACKENDS:
            raise ValueError(f"ik_backend must be one of {IK_BACKENDS}")
        if self.tracking not in TRACKING_BACKENDS:
            raise ValueError(f"tracking must be one of {TRACKING_BACKENDS}")
        if self.backend != "mujoco" and self.ik_backend != "lm":
            raise ValueError("Native IK backends require backend=mujoco")
        if self.backend != "mujoco" and self.tracking != "kkt":
            raise ValueError("mj-inverse tracking requires backend=mujoco")
        if self.backend == "pink" and self.ik_backend != "lm":
            raise ValueError("Pink backend cannot use native MuJoCo IK")
        if self.backend == "pink" and self.tracking != "kkt":
            raise ValueError("Pink backend cannot use mj-inverse tracking")
        if self.step_mode not in STEP_MODES:
            raise ValueError(f"step_mode must be one of {STEP_MODES}")
        if self.solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}")
        for value in (
            self.stature_m,
            self.mass_kg,
            self.trunk_scale,
            self.arm_scale,
            self.shoulder_scale,
        ):
            if not value > 0:
                raise ValueError("Subject values and scale factors must be positive")
        if self.free_wrists and self.bound_wrists:
            raise ValueError("Cannot specify both free_wrists and bound_wrists")
        if not isinstance(self.shooting_fit, int) or self.shooting_fit < 0:
            raise ValueError("shooting_fit iterations must be a non-negative integer")
        if not (0.0 <= self.shooting_gain <= 1.0):
            raise ValueError("shooting_gain must be in [0, 1]")
        if self.cutoff_hz is not None and not (0.0 < self.cutoff_hz < 180.0):
            raise ValueError("cutoff_hz must be positive and below Nyquist (180 Hz)")

    @property
    def document_name(self) -> str:
        return f"full_body_spec_anthro_{self.club}"

    @property
    def run_name(self) -> str:
        return self.output_name or f"anthro_{self.capture}"

    @property
    def document_output_dir(self) -> Path:
        return assets.resolve_output_root(self.output_root)

    @property
    def document_path(self) -> Path:
        return self.document_output_dir / f"{self.document_name}.json"

    @property
    def output_dir(self) -> Path:
        root = self.document_output_dir
        if self.output_root is not None:
            return root / self.run_name
        if root.joinpath("evidence/ground_support").is_dir() or root == FULL_BODY:
            return root / "evidence/ground_support" / self.run_name
        return root / self.run_name


@dataclass(frozen=True)
class ExperimentRequest:
    """Settings for a downswing dynamics experiment."""

    run: Path | str
    name: str
    cutoff_hz: float | None = None
    omega: float = 120.0
    zeta: float = 1.0
    feedforward: float = 1.0
    legs_omega: float | None = None
    balance: bool = True
    root_regulation: tuple[float, float] | None = None
    transition_velocity: float | None = None
    friction: tuple[float, float] | None = None
    stiffness: float | None = None
    dissipation: float | None = None
    reference: Path | str | None = None
    dt: float | None = None
    duration: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Experiment name must be a non-empty string")
        if self.cutoff_hz is not None and not (0.0 < self.cutoff_hz < 180.0):
            raise ValueError(
                "Cutoff frequency must be positive and below Nyquist (180 Hz)"
            )
        if not (0.0 <= self.feedforward <= 1.0):
            raise ValueError("Feedforward gain must be in [0, 1]")
        if self.omega <= 0:
            raise ValueError("Omega must be positive")
        if self.zeta < 0:
            raise ValueError("Zeta must be non-negative")
        if self.legs_omega is not None and self.legs_omega <= 0:
            raise ValueError("Legs omega must be positive")
        if self.transition_velocity is not None and self.transition_velocity <= 0:
            raise ValueError("Transition velocity must be positive")
        if self.stiffness is not None and self.stiffness <= 0:
            raise ValueError("Stiffness must be positive")
        if self.dissipation is not None and self.dissipation < 0:
            raise ValueError("Dissipation must be non-negative")
        if self.dt is not None and self.dt <= 0:
            raise ValueError("Time step dt must be positive")
        if self.duration is not None and self.duration <= 0:
            raise ValueError("Duration must be positive")


def build_command(request: MatchRequest) -> list[str]:
    """Command line that writes the anthropometric document for ``request``."""
    native = request.native_path or resolve_native_spec()
    osim = request.osim_path or resolve_osim_model()
    candidate = request.candidate_path or resolve_candidate_spec()
    return [
        sys.executable,
        str(BUILDER),
        "--native",
        str(native),
        "--osim",
        str(osim),
        "--native-candidate",
        str(candidate),
        "--stature",
        str(request.stature_m),
        "--mass",
        str(request.mass_kg),
        "--trunk-scale",
        str(request.trunk_scale),
        "--arm-scale",
        str(request.arm_scale),
        "--shoulder-scale",
        str(request.shoulder_scale),
        "--club",
        request.club,
        "--output",
        str(request.document_output_dir),
    ]


def match_command(request: MatchRequest) -> list[str]:
    """Command line that matches the capture on the built document."""
    cmd = [
        sys.executable,
        str(DRIVER_SCRIPT),
        "--spec",
        str(request.document_path),
        "--skip-hip-calibration",
        "--static-seeds",
        "--capture",
        request.capture,
        "--out",
        str(request.output_dir),
    ]
    if request.backend != "mujoco":
        cmd.extend(["--backend", request.backend])
    if request.ik_backend != "lm":
        cmd.extend(["--ik-backend", request.ik_backend])
    if request.tracking != "kkt":
        cmd.extend(["--tracking", request.tracking])
    if request.step_mode != "physical":
        cmd.extend(["--pink-step-mode", request.step_mode])
    if request.solver != "quadprog":
        cmd.extend(["--pink-solver", request.solver])
    if request.free_wrists:
        cmd.append("--free-wrists")
    if request.bound_wrists:
        cmd.append("--bound-wrists")
    if request.fit_closure:
        cmd.append("--fit-closure")
    if request.zmp_filter:
        cmd.append("--zmp-filter")
    if request.shooting_fit > 0:
        cmd.extend(["--shooting-fit", str(request.shooting_fit)])
        cmd.extend(["--shooting-gain", str(request.shooting_gain)])
    elif request.shooting_gain != 0.7:
        cmd.extend(["--shooting-gain", str(request.shooting_gain)])
    return cmd


def experiment_command(request: ExperimentRequest) -> list[str]:
    """Command line for running a downswing experiment."""
    cmd = [
        sys.executable,
        str(DOWNSWING_SCRIPT),
        "--run",
        str(request.run),
        "--name",
        request.name,
        "--omega",
        str(request.omega),
        "--zeta",
        str(request.zeta),
        "--feedforward",
        str(request.feedforward),
    ]
    if request.cutoff_hz is not None:
        cmd.extend(["--cutoff-hz", str(request.cutoff_hz)])
    if request.legs_omega is not None:
        cmd.extend(["--legs-omega", str(request.legs_omega)])
    if not request.balance:
        cmd.append("--no-balance")
    if request.root_regulation is not None:
        cmd.extend(
            [
                "--root-regulation",
                str(request.root_regulation[0]),
                str(request.root_regulation[1]),
            ]
        )
    if request.transition_velocity is not None:
        cmd.extend(["--transition-velocity", str(request.transition_velocity)])
    if request.friction is not None:
        cmd.extend(["--friction", str(request.friction[0]), str(request.friction[1])])
    if request.stiffness is not None:
        cmd.extend(["--stiffness", str(request.stiffness)])
    if request.dissipation is not None:
        cmd.extend(["--dissipation", str(request.dissipation)])
    if request.reference is not None:
        cmd.extend(["--reference", str(request.reference)])
    if request.dt is not None:
        cmd.extend(["--dt", str(request.dt)])
    if request.duration is not None:
        cmd.extend(["--duration", str(request.duration)])
    return cmd


def export_mjx_command(run: Path | str) -> list[str]:
    """Command line that exports the MJX package for ``run``."""
    return [
        sys.executable,
        str(EXPORT_MJX_SCRIPT),
        "--run",
        str(run),
    ]


def validate_reference_command(
    run: Path | str, reference: Path | str, name: str = "mjx_validation"
) -> list[str]:
    """Command line that validates an optimised reference in the shared plant."""
    return [
        sys.executable,
        str(DOWNSWING_SCRIPT),
        "--run",
        str(run),
        "--name",
        name,
        "--reference",
        str(reference),
    ]


def read_experiment_summary(run_dir: Path | str, name: str) -> dict[str, Any]:
    """Read the downswing experiment receipt summary.

    Precondition: ``run_dir`` contains ``downswing_{name}.json``.
    Postcondition: Returns a dictionary containing headline metrics.
    """
    path = Path(run_dir) / f"downswing_{name}.json"
    if not path.exists():
        raise ValueError(f"No experiment receipt found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "name": name,
        "root_error_max_mm": _mm(data.get("root_error_max_m")),
        "root_error_timeline_m": data.get("root_error_timeline_m", {}),
        "marker_rms_to_1_5s_m": data.get("marker_rms_to_1_5s_m"),
        "marker_rms_to_1_5s_mm": _mm(data.get("marker_rms_to_1_5s_m")),
        "marker_rms_mm": _mm(data.get("marker_rms_m")),
        "inside_support_polygon_fraction": data.get("inside_support_polygon_fraction"),
        "peak_joint_torque_n_m": data.get("peak_joint_torque_n_m"),
    }


def summarise_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """The numbers a user reads first, from a ground-support receipt.

    Precondition: the receipt carries ``address``, ``ik`` and ``dynamics``
    blocks as written by the driver. Postcondition: every value is a plain
    float, bool or string; missing optional blocks read as ``None``.
    """
    for key in ("address", "ik", "dynamics"):
        if key not in receipt:
            raise ValueError(f"Receipt lacks the {key} block")
    backend = receipt.get("backend", "mujoco")
    constrained = receipt["ik"].get("constrained_ik") or {}
    all_frames_converged = constrained.get("all_frames_converged", True)
    is_qualified = constrained.get("is_qualified", True)
    address = receipt["address"].get("calibrated", {})
    com = address.get("centre_of_mass", {})
    dynamics = receipt["dynamics"]
    backswing = dynamics.get("backswing_to_1s", {})
    summary: dict[str, Any] = {
        "backend": backend,
        "is_qualified": is_qualified,
        "all_frames_converged": all_frames_converged,
        "capture": receipt.get("capture"),
        "club": (receipt.get("club") or {}).get("name"),
        "address_marker_rms_mm": _mm(address.get("marker_rms_m")),
        "full_capture_ik_rms_mm": _mm(receipt["ik"].get("marker_rms_m")),
        "backswing_root_error_max_mm": _mm(backswing.get("root_error_max_m")),
        "whole_run_root_rms_mm": _mm(dynamics.get("root_tracking_rms_m")),
        "inside_support_polygon_fraction": dynamics.get(
            "inside_support_polygon_fraction"
        ),
        "com_inside_polygon_at_address": com.get("inside_support_polygon"),
        "range_of_motion_flags_ik": sorted(
            (receipt["ik"].get("range_of_motion_flags") or {}).keys()
        ),
        "range_of_motion_flags_simulation": sorted(
            (dynamics.get("range_of_motion_flags") or {}).keys()
        ),
    }
    if "step_mode" in constrained:
        summary["step_mode"] = constrained["step_mode"]
    if (
        "first_failed_frame" in constrained
        and constrained["first_failed_frame"] is not None
    ):
        summary["first_failed_frame"] = constrained["first_failed_frame"]
    return summary


def _mm(value: Any) -> float | None:
    return None if value is None else round(float(value) * 1e3, 1)


def read_summary(output_dir: Path) -> dict[str, Any]:
    """Summary of the receipt in ``output_dir`` (ValueError when absent)."""
    receipt = output_dir / "receipt.json"
    if not receipt.exists():
        raise ValueError(f"No receipt in {output_dir}")
    return summarise_receipt(json.loads(receipt.read_text(encoding="utf-8")))


def artefacts(output_dir: Path) -> Sequence[Path]:
    """Playback GIFs of a finished run, in display order."""
    return tuple(
        p
        for p in (output_dir / "ik_playback.gif", output_dir / "tracking_playback.gif")
        if p.exists()
    )


def list_runs(ledger_path: Path | None = None) -> Sequence[Any]:
    """Return all classified runs from the matched-swing run ledger."""
    from src.shared.python.motion_matching.ledger import default_ledger_path, scan
    from src.shared.python.motion_matching.ledger_schema import Ledger

    path = ledger_path or default_ledger_path()
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Ledger.model_validate(data).rows
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return scan().rows
