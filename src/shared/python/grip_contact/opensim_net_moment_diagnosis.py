"""OpenSim ``net_moment_com`` parity diagnosis (#13560).

Decomposes the club-COM net moment into hand torques and ``r × F`` lever arms so
OpenSim lanes can see whether the gap is patch torque, friction mapping, or lever
geometry rather than net force alone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.grip_contact.club_dynamics import ClubDynamics
from src.shared.python.grip_contact.contact_run import ContactRun
from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.parity import (
    DETERMINATE_QUANTITIES,
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    GripKineticsSeries,
    QuantityError,
    parity_errors,
)

DIAGNOSIS_DURATION_S: float = 0.04
DIAGNOSIS_ISSUE: int = 13560


@dataclass(frozen=True)
class NetMomentComponents:
    """Per-sample contributors to :meth:`GripKineticsSeries.net_moment_com_nm`."""

    torque_l_nm: np.ndarray
    torque_r_nm: np.ndarray
    r_cross_f_l_nm: np.ndarray
    r_cross_f_r_nm: np.ndarray

    def total_nm(self) -> np.ndarray:
        """Sum of all contributors, shape ``(n, 3)``."""
        return (
            self.torque_l_nm
            + self.torque_r_nm
            + self.r_cross_f_l_nm
            + self.r_cross_f_r_nm
        )


def _club_com_offset_l(
    series: GripKineticsSeries,
    club: ClubDynamics | None,
    interface: GripInterface | None,
) -> np.ndarray:
    if club is not None and interface is not None:
        return np.asarray(club.com_m, float) - np.asarray(
            interface.left.position_m, float
        )
    if "club_com_offset_l_m" in series.metadata:
        return np.asarray(series.metadata["club_com_offset_l_m"], float)
    return np.array(
        (0.02501411692651008, 0.8679856230031948, 0.004910663712740029), float
    )


def club_com_world_m(
    series: GripKineticsSeries,
    club: ClubDynamics | None = None,
    interface: GripInterface | None = None,
) -> np.ndarray:
    """Club centre of mass in world coordinates, shape ``(n, 3)``."""
    dp_l = _club_com_offset_l(series, club, interface)
    return series.grip_point_m["L"] + np.einsum("nij,j->ni", series.club_rotation, dp_l)


def decompose_net_moment_components(
    series: GripKineticsSeries,
    club: ClubDynamics | None = None,
    interface: GripInterface | None = None,
) -> NetMomentComponents:
    """Split net moment about the club COM into torque and lever-arm parts."""
    p_com = club_com_world_m(series, club, interface)
    r_l = series.grip_point_m["L"] - p_com
    r_r = series.grip_point_m["R"] - p_com
    f_l = series.force_on_club_n["L"]
    f_r = series.force_on_club_n["R"]
    return NetMomentComponents(
        torque_l_nm=np.asarray(series.torque_on_club_nm["L"], float),
        torque_r_nm=np.asarray(series.torque_on_club_nm["R"], float),
        r_cross_f_l_nm=np.cross(r_l, f_l),
        r_cross_f_r_nm=np.cross(r_r, f_r),
    )


def _quantity_error_dict(err: QuantityError) -> dict[str, float | bool]:
    return err.to_dict()


def _residual_arrays(
    reference: ContactRun,
    candidate: ContactRun,
    t_end_s: float,
    club: ClubDynamics | None,
    interface: GripInterface | None,
) -> dict[str, np.ndarray]:
    ref_series = reference.series
    cand_series = candidate.series
    t = ref_series.time_s
    keep = np.ones(t.size, bool) if t_end_s is None else t <= t_end_s + 1e-12
    ref_m = ref_series.net_moment_com_nm(club, interface)[keep]
    cand_m = cand_series.net_moment_com_nm(club, interface)[keep]
    ref_parts = decompose_net_moment_components(ref_series, club, interface)
    cand_parts = decompose_net_moment_components(cand_series, club, interface)
    residual = cand_m - ref_m
    part_residuals: dict[str, np.ndarray] = {}
    for name in ("torque_l", "torque_r", "r_cross_f_l", "r_cross_f_r"):
        ref_arr = getattr(ref_parts, f"{name}_nm")[keep]
        cand_arr = getattr(cand_parts, f"{name}_nm")[keep]
        part_residuals[f"delta_{name}_nm"] = cand_arr - ref_arr
    return {
        "time_s": t[keep],
        "reference_net_moment_com_nm": ref_m,
        "candidate_net_moment_com_nm": cand_m,
        "residual_net_moment_com_nm": residual,
        **part_residuals,
    }


def trace_peak_summary(traces: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Summarise where the residual magnitude peaks and which term dominates."""
    residual = np.asarray(traces["residual_net_moment_com_nm"], float)
    mag = np.linalg.norm(residual, axis=1)
    peak_idx = int(np.argmax(mag))
    peak_mag = float(mag[peak_idx])
    part_keys = (
        "delta_torque_l_nm",
        "delta_torque_r_nm",
        "delta_r_cross_f_l_nm",
        "delta_r_cross_f_r_nm",
    )
    part_peaks = {
        key: float(np.linalg.norm(np.asarray(traces[key], float)[peak_idx]))
        for key in part_keys
        if key in traces
    }
    dominant = (
        max(part_peaks.items(), key=lambda item: item[1])[0] if part_peaks else ""
    )
    ref_peak = float(
        np.linalg.norm(traces["reference_net_moment_com_nm"], axis=1).max()
    )
    return {
        "peak_sample_index": peak_idx,
        "peak_time_s": float(traces["time_s"][peak_idx]),
        "peak_residual_nm": peak_mag,
        "reference_peak_nm": ref_peak,
        "residual_over_reference_peak": (
            peak_mag / ref_peak if ref_peak > 0.0 else float("inf")
        ),
        "part_residual_magnitude_at_peak_nm": part_peaks,
        "dominant_part_at_peak": dominant,
    }


def _serialise_trace_arrays(
    traces: Mapping[str, np.ndarray],
    *,
    max_samples: int = 64,
) -> dict[str, list[list[float]] | list[float]]:
    """Down-sample traces for JSON (always includes endpoints)."""
    n = int(np.asarray(traces["time_s"]).size)
    if n <= max_samples:
        idx = np.arange(n, dtype=int)
    else:
        idx = np.unique(np.linspace(0, n - 1, num=max_samples, dtype=int))
    out: dict[str, list[list[float]] | list[float]] = {}
    for key, arr in traces.items():
        sliced = np.asarray(arr, float)[idx]
        if sliced.ndim == 1:
            out[key] = [float(x) for x in sliced]
        else:
            out[key] = sliced.tolist()
    return out


def diagnosis_case_from_runs(
    *,
    style: GripStyle,
    motion: str,
    duration_s: float,
    reference: ContactRun,
    candidate: ContactRun,
    club: ClubDynamics,
    interface: GripInterface,
    include_traces: bool = True,
) -> dict[str, Any]:
    """One OpenSim-vs-Pinocchio case with errors and optional down-sampled traces."""
    if motion not in ("hold", "swing"):
        raise ValueError(f"motion must be hold or swing, got {motion!r}")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    errors = parity_errors(
        candidate, reference, t_end_s=duration_s, club=club, interface=interface
    )
    traces = _residual_arrays(reference, candidate, duration_s, club, interface)
    case: dict[str, Any] = {
        "style": GripStyle(style).value,
        "motion": motion,
        "duration_s": float(duration_s),
        "determinate_errors": {
            name: _quantity_error_dict(errors[name])
            for name in DETERMINATE_QUANTITIES
            if name in errors
        },
        "trace_summary": trace_peak_summary(traces),
    }
    if include_traces:
        case["traces"] = _serialise_trace_arrays(traces)
    return case


def xfail_baseline_cases() -> tuple[dict[str, Any], ...]:
    """Measured 0.04 s gaps recorded in ``test_grip_style_parity_report`` xfails."""
    return (
        {
            "style": GripStyle.OVERLAP.value,
            "motion": "hold",
            "duration_s": DIAGNOSIS_DURATION_S,
            "determinate_errors": {
                "net_moment_com": {
                    "peak_error": 0.05451,
                    "rms_error": 0.02512,
                    "passes": False,
                },
                "net_force": {"passes": True},
            },
            "source": "xfail_baseline",
        },
        {
            "style": GripStyle.INTERLOCK.value,
            "motion": "hold",
            "duration_s": DIAGNOSIS_DURATION_S,
            "determinate_errors": {
                "net_moment_com": {
                    "peak_error": 0.05451,
                    "rms_error": 0.02512,
                    "passes": False,
                },
                "net_force": {"passes": True},
            },
            "source": "xfail_baseline",
        },
        {
            "style": GripStyle.OVERLAP.value,
            "motion": "swing",
            "duration_s": DIAGNOSIS_DURATION_S,
            "determinate_errors": {
                "net_moment_com": {
                    "peak_error": 0.00391,
                    "rms_error": 0.04059,
                    "passes": False,
                },
                "net_force": {"passes": True},
            },
            "source": "xfail_baseline",
        },
        {
            "style": GripStyle.INTERLOCK.value,
            "motion": "swing",
            "duration_s": DIAGNOSIS_DURATION_S,
            "determinate_errors": {
                "net_moment_com": {
                    "peak_error": 0.00391,
                    "rms_error": 0.04059,
                    "passes": False,
                },
                "net_force": {"passes": True},
            },
            "source": "xfail_baseline",
        },
        {
            "style": GripStyle.TEN_FINGER.value,
            "motion": "swing",
            "duration_s": DIAGNOSIS_DURATION_S,
            "determinate_errors": {
                "net_moment_com": {
                    "peak_error": 0.02066,
                    "rms_error": 0.04104,
                    "passes": False,
                    "note": (
                        "RMS ~19.5 mN·m against ~0.475 N·m reference peak; "
                        "net_force passes (#13548)"
                    ),
                },
                "net_force": {
                    "peak_error": 0.00706,
                    "rms_error": 0.01011,
                    "passes": True,
                },
            },
            "source": "xfail_baseline",
        },
    )


def diagnosis_payload(
    cases: Sequence[Mapping[str, Any]],
    *,
    duration_s: float = DIAGNOSIS_DURATION_S,
    generated_by: str,
    opensim_available: bool,
) -> dict[str, Any]:
    """Top-level JSON document for the #13560 diagnosis artifact."""
    if not cases:
        raise ValueError("cases must be non-empty")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    return {
        "issue": DIAGNOSIS_ISSUE,
        "duration_s": float(duration_s),
        "reference_engine": "pinocchio",
        "candidate_engine": "opensim",
        "peak_tolerance": float(PEAK_TOLERANCE),
        "rms_tolerance": float(RMS_TOLERANCE),
        "hypothesis": (
            "Since #13308 the Pinocchio reference uses the shared line-contact law; "
            "remaining OpenSim gaps on net_moment_com are friction and patch-torque "
            "mapping (tau_n = 0, surface lever), not the capsule law or net_force."
        ),
        "generated_by": generated_by,
        "opensim_available": bool(opensim_available),
        "repro": [
            "PYTHONPATH=. python3 scripts/grip/opensim_net_moment_diagnosis.py",
            "PYTHONPATH=. python3 scripts/grip/grip_style_parity_report.py "
            "--duration-s 0.04 --out /tmp/grip_parity",
            "python3 -m pytest tests/unit/grip_contact/test_grip_style_parity_report.py "
            "-k 'opensim-overlap-hold' -m 'requires_opensim and slow' --timeout=600",
        ],
        "cases": [dict(case) for case in cases],
    }


def validate_diagnosis_payload(payload: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when ``payload`` is not a well-formed diagnosis doc."""
    for key in (
        "issue",
        "duration_s",
        "reference_engine",
        "candidate_engine",
        "cases",
        "repro",
    ):
        if key not in payload:
            raise ValueError(f"diagnosis payload missing {key!r}")
    if int(payload["issue"]) != DIAGNOSIS_ISSUE:
        raise ValueError("issue must be 13560")
    cases = payload["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")
    for case in cases:
        if "style" not in case or "motion" not in case:
            raise ValueError("each case needs style and motion")
        errs = case.get("determinate_errors")
        if not isinstance(errs, dict) or "net_moment_com" not in errs:
            raise ValueError("each case needs determinate_errors.net_moment_com")


def write_diagnosis_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Validate and write ``payload`` to ``path``."""
    validate_diagnosis_payload(payload)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if not text.endswith("\n"):
        text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def components_match_series(
    series: GripKineticsSeries,
    club: ClubDynamics | None = None,
    interface: GripInterface | None = None,
    *,
    atol: float = 1e-9,
) -> bool:
    """True when decomposition sums to :meth:`GripKineticsSeries.net_moment_com_nm`."""
    parts = decompose_net_moment_components(series, club, interface)
    total = parts.total_nm()
    direct = series.net_moment_com_nm(club, interface)
    return bool(np.allclose(total, direct, atol=atol))
