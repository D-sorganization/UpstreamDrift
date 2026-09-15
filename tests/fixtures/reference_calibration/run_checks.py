"""Exercise worker-only records in their real isolated provider environment."""

from pathlib import Path
import runpy

root = Path(__file__).resolve().parents[3]
worker = root / "src/tools/capture_rig/reference_calibration/worker.py"
bootstrap = runpy.run_path(str(worker), run_name="reference_worker_bootstrap")
bootstrap["_configure_provider"]()

# Match the worker's import order, independently of root conftest's deliberate
# local-Sidekick cache. This does not change imports in the parent test process.
import shared.python.sidekick.lab.mocap.reference_placements  # noqa: E402, F401
import pytest  # noqa: E402

fixture = Path(__file__).parent
raise SystemExit(
    pytest.main(
        [
            str(fixture / "session_checks.py"),
            str(fixture / "solver_checks.py"),
            str(fixture / "reuse_checks.py"),
            "--confcutdir",
            str(fixture),
            "-c",
            str(fixture / "pytest.ini"),
            "-q",
            "--no-cov",
        ]
    )
)
