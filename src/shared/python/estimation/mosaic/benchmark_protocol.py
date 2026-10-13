"""Frozen benchmark protocol, tolerances and promotion metrics for MOSAIC.

Governs the canonical synthetic planar benchmark fixture, defining its
experimental parameters, baseline priors, anchors, and frozen acceptance
thresholds. Any loosening of these tolerances requires empirical evidence
and a ``Tolerance-Change-Evidence`` commit trailer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any

from src.shared.python.core.contracts import ensure, require

logger = logging.getLogger(__name__)

PROTOCOL_CONFIG_PATH = Path("src/config/mosaic_benchmark_protocol.json")

# Authoritative frozen tolerances from epic specification
FROZEN_GEOMETRY_MAX_M: float = 1e-4
FROZEN_KINEMATIC_RMS_MAX_RAD: float = 1e-3
FROZEN_TORQUE_RMS_RATIO_MAX: float = 0.25
FROZEN_STRUCTURAL_RANK: int = 9
FROZEN_UNOBSERVABLE_PARAMETERS: tuple[str, ...] = ("m_0",)
FROZEN_OPEN_LOOP_REPLAY_RMS_MAX_RAD: float = 0.035
FROZEN_OPEN_LOOP_HORIZON_S: float = 0.75


@dataclass(frozen=True)
class FixtureConfig:
    """Fixed-seed synthetic experiment definition."""

    n_trials: int = 2
    n_nodes: int = 150
    sample_rate_hz: float = 200.0
    marker_noise_m: float = 5e-4
    data_seed: int = 8
    prior_seed: int = 2
    geometry_scale: float = 1.05
    prior_relative_spread: float = 0.10


@dataclass(frozen=True)
class ToleranceConfig:
    """Frozen acceptance tolerances for MOSAIC conformance."""

    geometry_error_max_m: float = FROZEN_GEOMETRY_MAX_M
    kinematic_rms_max_rad: float = FROZEN_KINEMATIC_RMS_MAX_RAD
    torque_rms_ratio_max: float = FROZEN_TORQUE_RMS_RATIO_MAX
    structural_rank: int = FROZEN_STRUCTURAL_RANK
    unobservable_parameters: tuple[str, ...] = FROZEN_UNOBSERVABLE_PARAMETERS
    open_loop_replay_rms_max_rad: float = FROZEN_OPEN_LOOP_REPLAY_RMS_MAX_RAD
    open_loop_horizon_s: float = FROZEN_OPEN_LOOP_HORIZON_S


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Frozen protocol specification and tolerance container."""

    fixture: FixtureConfig
    tolerances: ToleranceConfig
    anchors: dict[str, Any]
    promotion_metrics: dict[str, Any]
    evidence_trailer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert protocol to a plain dictionary."""
        data: dict[str, Any] = {
            "fixture": asdict(self.fixture),
            "tolerances": asdict(self.tolerances),
            "anchors": dict(self.anchors),
            "promotion_metrics": dict(self.promotion_metrics),
        }
        # ensure unobservable_parameters is serialized as a list
        data["tolerances"]["unobservable_parameters"] = list(
            self.tolerances.unobservable_parameters
        )
        if self.evidence_trailer is not None:
            data["evidence_trailer"] = self.evidence_trailer
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkProtocol:
        """Construct BenchmarkProtocol from a dictionary."""
        require(isinstance(data, dict), "data must be a dict")
        require("fixture" in data, "missing 'fixture' in protocol data")
        require("tolerances" in data, "missing 'tolerances' in protocol data")

        fixture_dict = data["fixture"]
        fixture = FixtureConfig(
            n_trials=int(fixture_dict.get("n_trials", 2)),
            n_nodes=int(fixture_dict.get("n_nodes", 150)),
            sample_rate_hz=float(fixture_dict.get("sample_rate_hz", 200.0)),
            marker_noise_m=float(fixture_dict.get("marker_noise_m", 5e-4)),
            data_seed=int(fixture_dict.get("data_seed", 8)),
            prior_seed=int(fixture_dict.get("prior_seed", 2)),
            geometry_scale=float(fixture_dict.get("geometry_scale", 1.05)),
            prior_relative_spread=float(
                fixture_dict.get("prior_relative_spread", 0.10)
            ),
        )

        tol_dict = data["tolerances"]
        unobs = tol_dict.get("unobservable_parameters", ("m_0",))
        tolerances = ToleranceConfig(
            geometry_error_max_m=float(
                tol_dict.get("geometry_error_max_m", FROZEN_GEOMETRY_MAX_M)
            ),
            kinematic_rms_max_rad=float(
                tol_dict.get("kinematic_rms_max_rad", FROZEN_KINEMATIC_RMS_MAX_RAD)
            ),
            torque_rms_ratio_max=float(
                tol_dict.get("torque_rms_ratio_max", FROZEN_TORQUE_RMS_RATIO_MAX)
            ),
            structural_rank=int(
                tol_dict.get("structural_rank", FROZEN_STRUCTURAL_RANK)
            ),
            unobservable_parameters=tuple(str(p) for p in unobs),
            open_loop_replay_rms_max_rad=float(
                tol_dict.get(
                    "open_loop_replay_rms_max_rad",
                    FROZEN_OPEN_LOOP_REPLAY_RMS_MAX_RAD,
                )
            ),
            open_loop_horizon_s=float(
                tol_dict.get("open_loop_horizon_s", FROZEN_OPEN_LOOP_HORIZON_S)
            ),
        )

        anchors = data.get("anchors", {})
        promotion = data.get("promotion_metrics", {})
        evidence_trailer = data.get("evidence_trailer")

        return cls(
            fixture=fixture,
            tolerances=tolerances,
            anchors=anchors,
            promotion_metrics=promotion,
            evidence_trailer=evidence_trailer,
        )


def load_benchmark_protocol(path: Path | None = None) -> BenchmarkProtocol:
    """Load benchmark protocol from disk or return default frozen protocol.

    Precondition: path exists if provided.
    Postcondition: returns a valid BenchmarkProtocol.
    """
    resolved_path = path or PROTOCOL_CONFIG_PATH
    if not resolved_path.exists():
        # Fallback to repo root if path is relative and called from subfolder
        repo_root = Path(__file__).resolve().parents[5]
        candidate = repo_root / resolved_path
        if candidate.exists():
            resolved_path = candidate

    require(
        resolved_path.exists(),
        f"benchmark protocol file not found at {resolved_path}",
    )
    data = json.loads(resolved_path.read_text(encoding="utf-8"))
    protocol = BenchmarkProtocol.from_dict(data)
    ensure(isinstance(protocol, BenchmarkProtocol), "protocol must be loaded")
    return protocol


def verify_protocol_integrity(
    protocol: BenchmarkProtocol, trailer: str | None = None
) -> bool:
    """Verify that tolerances have not been loosened without explicit evidence.

    Raises AssertionError if any tolerance is loosened without a valid
    ``Tolerance-Change-Evidence`` trailer.
    """
    tol = protocol.tolerances
    effective_trailer = trailer or protocol.evidence_trailer or ""
    has_trailer = "Tolerance-Change-Evidence" in effective_trailer

    is_widened = False
    reasons: list[str] = []

    if tol.geometry_error_max_m > FROZEN_GEOMETRY_MAX_M + 1e-12:
        is_widened = True
        reasons.append(
            f"geometry_error_max_m {tol.geometry_error_max_m} > {FROZEN_GEOMETRY_MAX_M}"
        )
    if tol.kinematic_rms_max_rad > FROZEN_KINEMATIC_RMS_MAX_RAD + 1e-12:
        is_widened = True
        reasons.append(
            f"kinematic_rms_max_rad {tol.kinematic_rms_max_rad} > {FROZEN_KINEMATIC_RMS_MAX_RAD}"
        )
    if tol.torque_rms_ratio_max > FROZEN_TORQUE_RMS_RATIO_MAX + 1e-12:
        is_widened = True
        reasons.append(
            f"torque_rms_ratio_max {tol.torque_rms_ratio_max} > {FROZEN_TORQUE_RMS_RATIO_MAX}"
        )
    if tol.structural_rank < FROZEN_STRUCTURAL_RANK:
        is_widened = True
        reasons.append(
            f"structural_rank {tol.structural_rank} < {FROZEN_STRUCTURAL_RANK}"
        )
    if tol.open_loop_replay_rms_max_rad > FROZEN_OPEN_LOOP_REPLAY_RMS_MAX_RAD + 1e-12:
        is_widened = True
        reasons.append(
            f"open_loop_replay_rms_max_rad {tol.open_loop_replay_rms_max_rad} > {FROZEN_OPEN_LOOP_REPLAY_RMS_MAX_RAD}"
        )
    if tol.open_loop_horizon_s < FROZEN_OPEN_LOOP_HORIZON_S - 1e-12:
        is_widened = True
        reasons.append(
            f"open_loop_horizon_s {tol.open_loop_horizon_s} < {FROZEN_OPEN_LOOP_HORIZON_S}"
        )
    if tol.unobservable_parameters != FROZEN_UNOBSERVABLE_PARAMETERS:
        is_widened = True
        reasons.append(
            f"unobservable_parameters {tol.unobservable_parameters} != {FROZEN_UNOBSERVABLE_PARAMETERS}"
        )

    if is_widened and not has_trailer:
        detail = ", ".join(reasons)
        raise AssertionError(
            f"Tolerance-Change-Evidence trailer required. Loosened tolerances: {detail}"
        )

    return True
