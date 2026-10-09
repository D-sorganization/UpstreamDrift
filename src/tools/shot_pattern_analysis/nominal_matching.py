"""Match zero-error curved shots to one physical target by forward simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol, TypeVar

import numpy as np
from scipy.optimize import least_squares

from .core import AnalysisConfig
from .matched_config import MatchedConfig
from .physics import ShotOutcome

ConfigT = TypeVar("ConfigT", AnalysisConfig, MatchedConfig)
ConfigContra = TypeVar(
    "ConfigContra", AnalysisConfig, MatchedConfig, contravariant=True
)


class LandingPhysics(Protocol[ConfigContra]):
    """The public endpoint needed to calibrate either scientific adapter."""

    def simulate(
        self,
        *,
        face_deg: float,
        path_deg: float,
        config: ConfigContra,
        sample_trajectory: bool = False,
        nominal_face_deg: float = 0.0,
    ) -> ShotOutcome: ...


class NominalMatchError(RuntimeError):
    """Requested raw landing is unreachable within the stated delivery bounds."""


@dataclass(frozen=True)
class MatchBounds:
    """Closed face/speed search box for a nominal delivery."""

    face_min_deg: float
    face_max_deg: float
    speed_min_mps: float
    speed_max_mps: float

    def __post_init__(self) -> None:
        values = (
            self.face_min_deg,
            self.face_max_deg,
            self.speed_min_mps,
            self.speed_max_mps,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or self.face_min_deg >= self.face_max_deg
            or self.speed_min_mps <= 0
            or self.speed_min_mps >= self.speed_max_mps
            or max(abs(self.face_min_deg), abs(self.face_max_deg)) >= 15
        ):
            raise ValueError("matching bounds must be finite, ordered, and physical")


@dataclass(frozen=True)
class MatchedNominal:
    """One physically simulated zero-error delivery at the common target."""

    face_deg: float
    path_deg: float
    speed_mps: float
    outcome: ShotOutcome
    residual_m: float
    n_evaluations: int


def match_nominal_landing(
    physics: LandingPhysics[ConfigT],
    *,
    config: ConfigT,
    target_x_m: float,
    reference_face_deg: float,
    face_to_path_deg: float,
    bounds: MatchBounds | None = None,
    tolerance_m: float = 0.05,
) -> MatchedNominal:
    """Solve face and speed while keeping face minus path fixed.

    The simulated *raw* endpoint must reach (target_x_m, 0). There is no
    postflight rotation, translation, or per-shot aim adjustment. The nominal
    face is passed as its own reference so shaft-rotation coupling responds to
    later face errors, not to a change in the zero-error club setup.
    """
    if not math.isfinite(target_x_m) or target_x_m <= 0:
        raise ValueError("target_x_m must be finite and positive")
    if not all(math.isfinite(v) for v in (reference_face_deg, face_to_path_deg)):
        raise ValueError("reference face and face-to-path must be finite")
    if not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("tolerance_m must be finite and positive")
    if bounds is None:
        bounds = MatchBounds(
            reference_face_deg - 3.0,
            reference_face_deg + 3.0,
            config.club_speed_mps * 0.85,
            config.club_speed_mps * 1.15,
        )
    low = np.array([bounds.face_min_deg, bounds.speed_min_mps], dtype=float)
    high = np.array([bounds.face_max_deg, bounds.speed_max_mps], dtype=float)
    initial = np.clip([reference_face_deg, config.club_speed_mps], low, high)

    def simulate(candidate: np.ndarray) -> ShotOutcome:
        face, speed = (float(v) for v in candidate)
        outcome = physics.simulate(
            face_deg=face,
            path_deg=face - face_to_path_deg,
            config=replace(config, club_speed_mps=speed),
            sample_trajectory=False,
            nominal_face_deg=face,
        )
        if not all(math.isfinite(v) for v in (outcome.carry_x_m, outcome.carry_y_m)):
            raise RuntimeError(
                "matching physics produced nonfinite landing coordinates"
            )
        return outcome

    def residuals(candidate: np.ndarray) -> np.ndarray:
        landing = simulate(candidate)
        return np.array(
            [landing.carry_x_m - target_x_m, landing.carry_y_m], dtype=float
        )

    result = least_squares(
        residuals,
        initial,
        bounds=(low, high),
        max_nfev=80,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    final = simulate(result.x)
    residual = math.hypot(final.carry_x_m - target_x_m, final.carry_y_m)
    if not result.success or residual > tolerance_m:
        raise NominalMatchError(
            f"nominal landing unreachable within bounds: residual={residual:.6f} m"
        )
    face, speed = (float(v) for v in result.x)
    return MatchedNominal(
        face_deg=face,
        path_deg=face - face_to_path_deg,
        speed_mps=speed,
        outcome=final,
        residual_m=residual,
        n_evaluations=int(result.nfev),
    )
