"""
Example: Choose Your Engine Demo

Demonstrates how to iterate over available physics engines (MuJoCo, Pinocchio,
Drake, OpenSim) through the unified EngineManager interface, load a simple
humanoid model in each, run a short simulation loop, and compare basic outputs.

This example is the companion to:
    docs/tutorials/choose_your_engine.md

Usage::

    python3 examples/choose_engine_demo.py

If an engine is not installed it is skipped gracefully.  Install engines with:
    pip install mujoco          # MuJoCo
    conda install pinocchio     # Pinocchio (conda-forge)
    pip install drake           # Drake
    # OpenSim: see https://github.com/opensim-org/opensim-core
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Bundled humanoid model included with the repository
HUMANOID_URDF = (
    Path(__file__).resolve().parents[1] / "src/shared/urdf/simple_humanoid.urdf"
)

# Engines to attempt in order of recommendation
ENGINES_TO_TRY: list[str] = ["mujoco", "pinocchio", "drake", "opensim"]

# Number of simulation steps to run for each engine
SIM_STEPS: int = 200


def run_demo() -> None:
    """Load each available engine, run a short sim loop, and report results."""
    try:
        from src.shared.python.engine_manager import (
            EngineManager,  # type: ignore[import]
        )
    except ImportError as exc:
        logger.error(
            "Could not import EngineManager — is the package installed? (%s)", exc
        )
        sys.exit(1)

    manager = EngineManager()
    results: dict[str, str] = {}

    for engine_name in ENGINES_TO_TRY:
        logger.info("--- Trying engine: %s ---", engine_name)
        try:
            engine = manager.load_engine(engine_name)
        except ImportError as exc:
            logger.warning("  %s not installed, skipping. (%s)", engine_name, exc)
            results[engine_name] = "not installed"
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("  Failed to load %s: %s", engine_name, exc, exc_info=True)
            results[engine_name] = f"load error: {exc}"
            continue

        model_path = HUMANOID_URDF
        if not model_path.exists():
            logger.warning(
                "  Bundled URDF not found at %s; skipping %s", model_path, engine_name
            )
            results[engine_name] = "model file missing"
            continue

        try:
            engine.load_from_path(str(model_path))
            engine.reset()

            for _ in range(SIM_STEPS):
                engine.step()

            t = engine.get_time()
            q = engine.get_configuration()
            logger.info(
                "  %s: completed %d steps, t=%.4f s, q[0]=%.6f",
                engine_name,
                SIM_STEPS,
                t,
                q[0] if len(q) > 0 else float("nan"),
            )
            results[engine_name] = f"OK  t={t:.4f}s"
        except NotImplementedError as exc:
            logger.warning(
                "  %s has unimplemented adapter methods: %s", engine_name, exc
            )
            results[engine_name] = f"partial (NotImplementedError: {exc})"
        except Exception as exc:  # noqa: BLE001
            logger.error("  %s simulation error: %s", engine_name, exc, exc_info=True)
            results[engine_name] = f"error: {exc}"

    # Summary table
    print("\n" + "=" * 50)
    print("Engine Demo Summary")
    print("=" * 50)
    for name, status in results.items():
        print(f"  {name:<12}  {status}")
    print("=" * 50)
    print(
        "\nSee docs/tutorials/choose_your_engine.md for guidance on selecting "
        "the right engine for your use case."
    )


if __name__ == "__main__":
    run_demo()
