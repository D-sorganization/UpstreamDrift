"""Exercise the Tools session export with the pinned Tools family resolved first.

Same arrangement as ``tests/fixtures/reference_calibration/run_checks.py``: the
root test process deliberately keeps ``shared.python.sidekick.lab.mocap``
unresolvable (UpstreamDrift's own Sidekick is cached first), so the ready path
of the bridge is checked here, in a fresh process wired like the launcher
bootstrap and the capture-rig worker.
"""

from pathlib import Path
import runpy

root = Path(__file__).resolve().parents[3]
worker = root / "src/tools/capture_rig/reference_calibration/worker.py"
bootstrap = runpy.run_path(str(worker), run_name="reference_worker_bootstrap")
bootstrap["_configure_provider"]()

import shared.python.sidekick.lab.mocap  # noqa: E402, F401
import pytest  # noqa: E402

fixture = Path(__file__).parent
raise SystemExit(
    pytest.main(
        [
            str(fixture / "export_checks.py"),
            "--confcutdir",
            str(fixture),
            "-c",
            str(fixture / "pytest.ini"),
            "-q",
            "--no-cov",
        ]
    )
)
