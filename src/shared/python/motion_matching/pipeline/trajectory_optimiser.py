"""Selectable trajectory optimisation stage for the matching pipeline (#11051).

Provides backend registration, validation, and execution for post-tracking
trajectory optimisation (e.g. MJX knot optimisation). Does not import
MuJoCo or JAX at module level to preserve shared library boundaries.
"""

from __future__ import annotations

from dataclasses import asdict
import importlib
import json
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.pipeline.constants import DEFAULT_MJX_ITERATIONS
from src.shared.python.workspace.installed_journeys import DependencyUnavailableError

TRAJECTORY_OPTIMISERS: frozenset[str] = frozenset({"none", "mjx-knots"})


@precondition(lambda name: isinstance(name, str), "name must be a string")
@postcondition(
    lambda result: result in TRAJECTORY_OPTIMISERS,
    "trajectory_optimiser must be supported",
)
def validate_trajectory_optimiser(name: str) -> str:
    """Validate selectable trajectory optimiser backend names."""
    key = name.strip().lower()
    if key not in TRAJECTORY_OPTIMISERS:
        known = ", ".join(sorted(TRAJECTORY_OPTIMISERS))
        raise ValueError(
            f"Unknown trajectory optimiser {name!r}; expected one of [{known}]"
        )
    return key


@precondition(lambda name, *_, **__: isinstance(name, str), "name must be a string")
def run_trajectory_optimiser(
    name: str,
    out_dir: Path,
    *,
    iterations: int = DEFAULT_MJX_ITERATIONS,
    method: Literal["adam", "lbfgs"] = "adam",
    neural_mode: str = "classical",
    allow_fallback: bool = True,
    proposal_initializer: Any = None,
) -> dict[str, Any] | None:
    """Run the selected trajectory optimiser on a pipeline output directory.

    'none' returns None and touches no files.
    'mjx-knots' exports the run's MJX package (the run's receipt.json must
    already be written), optimises the knot corrections, writes
    mjx_optimised_reference.npz and mjx_optimisation_receipt.json, and
    returns a summary dict. In 'preview' neural mode, starts from an
    injected proposal or falls back to classical start (unless fallback disabled).

    Raises DependencyUnavailableError if jax or mujoco.mjx is not installed.
    """
    backend = validate_trajectory_optimiser(name)
    if backend == "none":
        return None
    for module in ("jax", "mujoco.mjx"):
        try:
            importlib.import_module(module)
        except ImportError as err:
            raise DependencyUnavailableError(
                f"trajectory optimiser 'mjx-knots' needs {module}, which is not "
                f"importable ({err}); install the mjx extra or pass "
                "'--trajectory-optimiser none'."
            ) from err

    from src.engines.physics_engines.mujoco.python.motion_matching.mjx_knot_optimiser import (
        KnotOptimisationSettings,
        load_mjx_package,
        _prepare,
        optimisation_receipt,
        optimise_reference,
    )
    from src.shared.python.motion_matching.execution.mjx_export import (
        export_mjx_package,
    )
    from src.shared.python.motion_matching.proposal_initializer import (
        StubProposalInitializer,
        validate_proposal,
    )
    from src.tools.motion_matching.neural_cli import NeuralRunRejectedError

    run = Path(out_dir)
    export_mjx_package(run)
    package = load_mjx_package(run)
    settings = KnotOptimisationSettings(iterations=iterations, method=method)

    init_delta: np.ndarray | None = None
    neural_proposal_receipt: dict[str, Any] | None = None

    if neural_mode == "preview":
        initializer = proposal_initializer or StubProposalInitializer()
        prop_delta, source_id, prop_sha = initializer.propose(package)
        problem = _prepare(package, settings)
        is_valid, reason = validate_proposal(prop_delta, problem.delta_shape)
        if is_valid:
            init_delta = prop_delta
            neural_proposal_receipt = {
                "source": source_id,
                "sha256": prop_sha,
                "accepted": True,
                "fallback_reason": None,
            }
        else:
            if not allow_fallback:
                raise NeuralRunRejectedError(f"rejected neural proposal: {reason}")
            init_delta = None
            neural_proposal_receipt = {
                "source": source_id,
                "sha256": prop_sha,
                "accepted": False,
                "fallback_reason": reason,
            }

    t0 = time.perf_counter()
    result = optimise_reference(package, settings, init_delta=init_delta)
    np.savez(
        run / "mjx_optimised_reference.npz",
        q=result.q_best,
        delta_knots=result.delta_best,
        knot_spacing_s=settings.knot_spacing_s,
    )
    receipt = optimisation_receipt(
        result,
        asdict(settings),
        package=package,
        run=run,
        elapsed_s=round(time.perf_counter() - t0, 1),
    )
    if neural_proposal_receipt is not None:
        receipt["neural"] = {"proposal": neural_proposal_receipt}
    (run / "mjx_optimisation_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    summary: dict[str, Any] = {
        "name": backend,
        "method": settings.method,
        "iterations": iterations,
        "port_check_replay_marker_rms_m": result.port_check_replay_marker_rms_m,
        "best_replay_marker_rms_m": result.best_replay_marker_rms_m,
        "stop_reason": result.stop_reason,
        "receipt": "mjx_optimisation_receipt.json",
    }
    if neural_proposal_receipt is not None:
        summary["neural_proposal"] = neural_proposal_receipt
    return summary
