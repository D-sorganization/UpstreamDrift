"""Measure physical error when mathematically inactive sensitivity columns grow."""

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy

from src.shared.python.motion_matching.forward_sensitivity import (
    integrate_sensitivities,
)


def main() -> None:
    clock = np.linspace(0.0, 1.0, 101)
    records = []
    for columns in (1, 27, 81):

        def linearize(t: float, state: np.ndarray, count: int = columns) -> tuple:
            return (
                np.array([200 * np.cos(200 * t)]),
                np.zeros((1, 1)),
                np.zeros((1, count)),
            )

        result = integrate_sensitivities(
            np.zeros(1),
            clock,
            linearize,
            columns,
            rtol=1e-6,
            atol=1e-9,
            max_step=1,
            max_evaluations=20000,
        )
        records.append(
            {
                "columns": columns,
                "max_physical_error": float(
                    np.max(np.abs(result.integration.state[:, 0] - np.sin(200 * clock)))
                ),
                "max_sensitivity_abs": float(
                    np.max(np.abs(result.state_parameter_jacobian))
                ),
                "evaluations": result.integration.evaluations,
                "physical_state": result.integration.state[:, 0].tolist(),
            }
        )
    report = {
        "qualification": "manufactured padding diagnostic; not native dynamics or fit acceptance",
        "equation": "x_dot=200*cos(200*t), x(0)=0; all parameter derivatives identically zero",
        "rtol": 1e-6,
        "atol": 1e-9,
        "max_step": 1,
        "max_evaluations": 20000,
        "time": clock.tolist(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": records,
    }
    Path(__file__).with_name("report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
