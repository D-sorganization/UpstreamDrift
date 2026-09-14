r"""Global degree-six polynomial effort profile and forward simulation controller (OS-5).

Standardizes on a well-conditioned degree-six descending monomial basis
$f(t) = c_0 t^6 + c_1 t^5 + \dots + c_6$ matching OpenSim's native
`PolynomialFunction` and Simscape descending power conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

DEGREE_6: int = 6
COEFFS_COUNT: int = DEGREE_6 + 1  # 7 coefficients: c0..c6


@dataclass(frozen=True)
class Degree6PolynomialCoefficients:
    r"""Degree-6 polynomial effort coefficients for one actuator.

    By convention, coefficients are stored in descending power order:
    $f(t) = c_0 t^6 + c_1 t^5 + c_2 t^4 + c_3 t^3 + c_4 t^2 + c_5 t + c_6$.
    This directly matches OpenSim `PolynomialFunction` and Simscape native order.
    """

    actuator_name: str
    coefficients: tuple[float, float, float, float, float, float, float]
    ordering: str = "descending"
    units: str = "N*m"
    time_origin_s: float = 0.0
    duration_s: float = 0.85

    def __post_init__(self) -> None:
        """Validate DbC parameter preconditions."""
        if len(self.coefficients) != COEFFS_COUNT:
            raise ValueError(
                f"coefficients must have exactly {COEFFS_COUNT} coefficients, "
                f"got {len(self.coefficients)}"
            )
        if not all(np.isfinite(c) for c in self.coefficients):
            raise ValueError("All polynomial coefficients must be finite")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration_s must be positive and finite")
        if not np.isfinite(self.time_origin_s) or self.time_origin_s < 0:
            raise ValueError("time_origin_s must be non-negative and finite")
        if self.ordering not in ("descending", "ascending"):
            raise ValueError(f"Unknown ordering: {self.ordering!r}")

    @property
    def degree(self) -> int:
        """Polynomial degree."""
        return DEGREE_6

    def evaluate(self, t: float) -> float:
        r"""Evaluate continuous effort $\tau(t)$ via Horner's method."""
        if not np.isfinite(t):
            raise ValueError(f"Evaluation time t must be finite, got {t!r}")
        dt = t - self.time_origin_s
        coeffs = self.to_descending()
        val = coeffs[0]
        for c in coeffs[1:]:
            val = val * dt + c
        return float(val)

    def evaluate_rate(self, t: float) -> float:
        r"""Evaluate continuous effort derivative $d\tau/dt(t)$."""
        if not np.isfinite(t):
            raise ValueError(f"Evaluation time t must be finite, got {t!r}")
        dt = t - self.time_origin_s
        desc = self.to_descending()
        # Derivative powers: 6*c0, 5*c1, 4*c2, 3*c3, 2*c4, c5
        val = 6.0 * desc[0]
        for p, c in zip((5.0, 4.0, 3.0, 2.0, 1.0), desc[1:6], strict=True):
            val = val * dt + p * c
        return float(val)

    def to_descending(self) -> tuple[float, float, float, float, float, float, float]:
        """Return coefficients in descending power order $c_0..c_6$."""
        if self.ordering == "descending":
            return self.coefficients
        return self.coefficients[::-1]  # type: ignore[return-value]

    def to_ascending(self) -> tuple[float, float, float, float, float, float, float]:
        """Return coefficients in ascending power order $c_6..c_0$."""
        if self.ordering == "ascending":
            return self.coefficients
        return self.coefficients[::-1]  # type: ignore[return-value]

    def to_simscape_vector(self) -> NDArray[np.float64]:
        """Convert to Simscape descending power 1D array."""
        return np.array(self.to_descending(), dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """Serialize coefficient object to dictionary."""
        return {
            "actuator_name": self.actuator_name,
            "coefficients": list(self.to_descending()),
            "ordering": "descending",
            "units": self.units,
            "time_origin_s": self.time_origin_s,
            "duration_s": self.duration_s,
            "degree": self.degree,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Degree6PolynomialCoefficients:
        """Reconstruct coefficient object from dictionary."""
        raw_coeffs = data["coefficients"]
        coeffs_tuple = tuple(float(x) for x in raw_coeffs)
        if len(coeffs_tuple) != COEFFS_COUNT:
            raise ValueError(
                f"Expected {COEFFS_COUNT} coefficients, got {len(coeffs_tuple)}"
            )
        return cls(
            actuator_name=str(data["actuator_name"]),
            coefficients=coeffs_tuple,  # type: ignore[arg-type]
            ordering=str(data.get("ordering", "descending")),
            units=str(data.get("units", "N*m")),
            time_origin_s=float(data.get("time_origin_s", 0.0)),
            duration_s=float(data.get("duration_s", 0.85)),
        )


@dataclass
class PolynomialTorqueProfile:
    """Full-body degree-6 polynomial torque profile across all actuators."""

    actuator_names: tuple[str, ...]
    profiles: dict[str, Degree6PolynomialCoefficients]
    duration_s: float = 0.85
    basis: str = "monomial_descending_degree6"
    fit_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def evaluate_all(self, t: float) -> dict[str, float]:
        """Evaluate efforts for all actuators at time t."""
        return {name: self.profiles[name].evaluate(t) for name in self.actuator_names}

    def evaluate_matrix(self, times: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate efforts on a time array, shape (len(times), len(actuators))."""
        out = np.zeros((len(times), len(self.actuator_names)), dtype=np.float64)
        for j, name in enumerate(self.actuator_names):
            poly = self.profiles[name]
            desc = poly.to_descending()
            out[:, j] = np.polyval(desc, times - poly.time_origin_s)
        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile to dictionary."""
        return {
            "basis": self.basis,
            "degree": DEGREE_6,
            "duration_s": self.duration_s,
            "actuator_names": list(self.actuator_names),
            "profiles": {
                name: self.profiles[name].to_dict() for name in self.actuator_names
            },
            "fit_metrics": self.fit_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolynomialTorqueProfile:
        """Reconstruct profile from dictionary."""
        actuator_names = tuple(data["actuator_names"])
        raw_profiles = data["profiles"]
        profiles = {
            name: Degree6PolynomialCoefficients.from_dict(raw_profiles[name])
            for name in actuator_names
        }
        return cls(
            actuator_names=actuator_names,
            profiles=profiles,
            duration_s=float(data.get("duration_s", 0.85)),
            basis=str(data.get("basis", "monomial_descending_degree6")),
            fit_metrics=data.get("fit_metrics", {}),
        )

    def save_json(self, path: Path | str) -> None:
        """Save profile to JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> PolynomialTorqueProfile:
        """Load profile from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def fit_degree6_from_discrete_controls(
    times: Sequence[float] | NDArray[Any],
    controls: Sequence[Sequence[float]] | NDArray[Any],
    actuator_names: Sequence[str],
    duration_s: float,
) -> PolynomialTorqueProfile:
    """Fit degree-6 polynomials to discrete control trajectories using least squares."""
    t_arr = np.asarray(times, dtype=np.float64)
    u_arr = np.asarray(controls, dtype=np.float64)
    if u_arr.ndim == 1:
        u_arr = u_arr.reshape(-1, 1)

    if len(t_arr) != u_arr.shape[0]:
        raise ValueError(
            f"Times length {len(t_arr)} does not match controls rows {u_arr.shape[0]}"
        )
    if u_arr.shape[1] != len(actuator_names):
        raise ValueError(
            f"Controls columns {u_arr.shape[1]} does not match actuators {len(actuator_names)}"
        )
    if duration_s <= 0 or not np.isfinite(duration_s):
        raise ValueError("duration_s must be positive and finite")

    profiles: dict[str, Degree6PolynomialCoefficients] = {}
    fit_metrics: dict[str, dict[str, float]] = {}

    for idx, name in enumerate(actuator_names):
        u_col = u_arr[:, idx]
        coeffs_desc = np.polyfit(t_arr, u_col, deg=DEGREE_6)
        coeffs_tuple = tuple(float(c) for c in coeffs_desc)
        poly_obj = Degree6PolynomialCoefficients(
            actuator_name=name,
            coefficients=coeffs_tuple,  # type: ignore[arg-type]
            duration_s=duration_s,
        )
        profiles[name] = poly_obj

        # Metrics
        y_fit = np.polyval(coeffs_desc, t_arr)
        residuals = u_col - y_fit
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((u_col - np.mean(u_col)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-14 else 1.0
        max_err = float(np.max(np.abs(residuals)))
        rms_err = float(np.sqrt(np.mean(residuals**2)))

        fit_metrics[name] = {
            "r_squared": float(r2),
            "max_abs_error": float(max_err),
            "rms_error": float(rms_err),
            "ss_res": float(ss_res),
        }

    return PolynomialTorqueProfile(
        actuator_names=tuple(actuator_names),
        profiles=profiles,
        duration_s=duration_s,
        fit_metrics=fit_metrics,
    )


def check_effort_and_rate_bounds(
    profile: PolynomialTorqueProfile,
    max_effort: float = 1000.0,
    max_rate: float = 10000.0,
    num_samples: int = 100,
) -> tuple[bool, dict[str, dict[str, float]]]:
    """Check whether profile satisfies physical torque and torque rate bounds."""
    t_eval = np.linspace(0.0, profile.duration_s, num_samples)
    violations: dict[str, dict[str, float]] = {}

    for name in profile.actuator_names:
        poly = profile.profiles[name]
        efforts = np.array([poly.evaluate(t) for t in t_eval])
        rates = np.array([poly.evaluate_rate(t) for t in t_eval])

        peak_effort = float(np.max(np.abs(efforts)))
        peak_rate = float(np.max(np.abs(rates)))

        if peak_effort > max_effort or peak_rate > max_rate:
            violations[name] = {
                "peak_effort": peak_effort,
                "max_effort_limit": max_effort,
                "peak_rate": peak_rate,
                "max_rate_limit": max_rate,
            }

    return (len(violations) == 0, violations)


def create_polynomial_prescribed_controller(
    profile: PolynomialTorqueProfile,
    model: Any,
) -> Any:
    """Create an opensim.PrescribedController with PolynomialFunction per actuator."""
    import opensim

    controller = opensim.PrescribedController()
    force_set = model.getForceSet()

    for name in profile.actuator_names:
        clean_name = name.split("/")[-1] if "/" in name else name
        actuator = None
        if force_set.contains(clean_name):
            actuator = force_set.get(clean_name)
        elif force_set.contains(name):
            actuator = force_set.get(name)

        if actuator is None:
            logger.warning("Actuator %s not found in model ForceSet; skipping", name)
            continue

        poly_coeffs = profile.profiles[name].to_descending()
        simtk_vec = opensim.Vector(COEFFS_COUNT, 0.0)
        for i, val in enumerate(poly_coeffs):
            simtk_vec.set(i, float(val))

        poly_func = opensim.PolynomialFunction(simtk_vec)
        controller.addActuator(opensim.Actuator.safeDownCast(actuator))
        controller.prescribeControlForActuator(actuator.getName(), poly_func)

    return controller


def load_controls_from_sto(
    path: Path | str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Load control trajectories from an OpenSim Storage (STO) file in pure Python."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"STO file not found: {path}")

    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    header_end = -1
    for idx, line in enumerate(lines):
        if line.strip() == "endheader":
            header_end = idx
            break

    if header_end == -1 or header_end + 1 >= len(lines):
        raise ValueError(f"Invalid STO file format: missing 'endheader' in {path}")

    col_line = lines[header_end + 1].strip()
    columns = [c for c in col_line.replace("\t", " ").split(" ") if c]
    if not columns or columns[0].lower() != "time":
        raise ValueError(f"First STO column must be 'time', got {columns[:1]}")

    actuator_names = columns[1:]
    data_rows: list[list[float]] = []
    for line in lines[header_end + 2 :]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = [float(p) for p in stripped.replace("\t", " ").split(" ") if p]
        if len(parts) == len(columns):
            data_rows.append(parts)

    data_arr = np.array(data_rows, dtype=np.float64)
    times = data_arr[:, 0]
    controls = data_arr[:, 1:]
    return times, controls, actuator_names
