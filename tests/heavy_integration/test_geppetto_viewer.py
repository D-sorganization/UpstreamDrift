"""Opt-in real-runtime contracts for the Gepetto viewer adapter."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.live_simulation,
    pytest.mark.requires_pinocchio,
    pytest.mark.requires_gl,
]


_REAL_PROBE = textwrap.dedent(
    """
    import os
    import pathlib
    import shutil
    import signal
    import subprocess
    import tempfile
    import time
    import uuid

    probe_deadline = time.monotonic() + 45.0

    def wait_for(predicate, description, timeout=20.0):
        deadline = min(probe_deadline, time.monotonic() + timeout)
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.25)
        raise RuntimeError(f"Timed out waiting for {description}")

    def stop_owned(process):
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    root = pathlib.Path(tempfile.mkdtemp(prefix="gepetto-viewer-probe-"))
    runtime = root / "runtime"
    runtime.mkdir(mode=0o700)
    env = os.environ.copy()
    env.update({"XDG_RUNTIME_DIR": str(runtime), "WAYLAND_DISPLAY": "wayland-gepetto"})
    weston = xwayland = gui = None
    try:
        weston_path = shutil.which("weston")
        xwayland_path = shutil.which("Xwayland")
        gui_path = shutil.which("gepetto-gui")
        if not weston_path or not xwayland_path or not gui_path:
            raise RuntimeError("weston, Xwayland, and gepetto-gui are required")
        if pathlib.Path(gui_path).stat().st_size == 0:
            raise RuntimeError(f"gepetto-gui is truncated: {gui_path}")

        weston = subprocess.Popen(
            [
                weston_path,
                "--backend=headless",
                "--renderer=pixman",
                "--socket=wayland-gepetto",
                "--no-config",
                f"--log={root / 'weston.log'}",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        wait_for(
            lambda: (runtime / "wayland-gepetto").exists(),
            "Weston Wayland socket",
        )

        display_number = next(
            n
            for n in range(100, 200)
            if not pathlib.Path(f"/tmp/.X11-unix/X{n}").exists()
        )
        display = f":{display_number}"
        xenv = env | {"DISPLAY": display}
        xwayland = subprocess.Popen(
            [xwayland_path, display, "-rootless", "-terminate"],
            env=xenv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        wait_for(
            lambda: pathlib.Path(f"/tmp/.X11-unix/X{display_number}").exists(),
            "Xwayland display socket",
        )

        gui = subprocess.Popen(
            [gui_path, "--log", str(root / "gepetto.log")],
            env=xenv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        import gepetto.corbaserver

        client = None
        last_error = None
        deadline = min(probe_deadline, time.monotonic() + 30.0)
        while time.monotonic() < deadline:
            if gui.poll() is not None:
                raise RuntimeError(f"gepetto-gui exited with {gui.returncode}")
            try:
                client = gepetto.corbaserver.Client()
                break
            except Exception as exc:  # CORBA exception classes vary by omniORB build.
                last_error = exc
                time.sleep(0.25)
        if client is None:
            raise RuntimeError(f"Timed out connecting to gepetto-gui: {last_error}")

        import pinocchio as pin
        from src.engines.physics_engines.pinocchio.python.dtack.viz import GeppettoViewer

        model = pin.buildSampleModelManipulator()
        visual_model = pin.buildSampleGeometryModelManipulator(model)
        foreign = f"world/foreign_{uuid.uuid4().hex}"
        viewer = GeppettoViewer()
        try:
            viewer.load_model(model, visual_model)
            client.gui.createGroup(foreign)
            q0 = pin.neutral(model)
            viewer.display(q0)
            geometry_name = visual_model.geometryObjects[0].name
            node = f"world/{viewer._root_node_name}/visuals/{geometry_name}"
            wait_for(lambda: client.gui.nodeExists(node), "loaded visual geometry")
            before = tuple(client.gui.getNodeGlobalTransform(node))
            q1 = q0.copy()
            q1[0] += 0.25
            viewer.display(q1)
            after = tuple(client.gui.getNodeGlobalTransform(node))
            if before == after:
                raise AssertionError("visual transform did not change after display(q)")
            owned_root = viewer._root_node_name
            viewer.close()
            viewer.close()
            if client.gui.nodeExists(f"world/{owned_root}"):
                raise AssertionError("close() left the adapter scene root")
            if not client.gui.nodeExists(foreign):
                raise AssertionError("close() removed a foreign scene node")
        finally:
            viewer.close()
    finally:
        stop_owned(gui)
        stop_owned(xwayland)
        stop_owned(weston)
        shutil.rmtree(root, ignore_errors=True)
    """
)


def test_geppetto_viewer_real_model_dispatch_and_cleanup() -> None:
    """Exercise real CORBA scene setup, pose dispatch, and owned cleanup."""
    pytest.importorskip("pinocchio")
    pytest.importorskip("gepetto.corbaserver")
    repo_root = Path(__file__).resolve().parents[2]
    process = subprocess.Popen(
        [sys.executable, "-c", _REAL_PROBE],
        cwd=repo_root,
        capture_output=True,
        text=True,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=75)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate(timeout=10)
        pytest.fail(
            "real Gepetto qualification exceeded its bounded timeout\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    assert process.returncode == 0, (
        f"real Gepetto qualification failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
