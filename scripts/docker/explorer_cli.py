#!/usr/bin/env python3
"""Model-explorer container CLI (UpstreamDrift #12878, workstream D).

One code path behind the ``explorer-headless`` / ``explorer-gui`` compose
services (``docker/explorer/entrypoint.sh`` dispatches here):

``probe-gl``
    Prove the requested ``MUJOCO_GL`` backend can create an offscreen
    context. Fails fast with an actionable message instead of a segfault
    deep inside a render call.
``engines``
    JSON report of every physics engine and the GUI toolkit: import status,
    version and a tiny functional smoke (load a model, step it).
``export``
    Open a model, run it (or replay a results file) and write an MP4. There
    is exactly one frame source (:func:`iter_frames`) and one video writer
    (:func:`write_mp4`).
``app-screenshot``
    Construct the real PyQt6 Model Explorer window, optionally load a model,
    and save a screenshot of it. Works with ``QT_QPA_PLATFORM=offscreen``
    (no display) or ``xcb`` (Xvfb/VNC or WSLg).

Exit codes: 0 ok, 64 usage/environment (missing display or GL), 65 bad input.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

EXIT_USAGE = 64
EXIT_BAD_INPUT = 65

HEADLESS_GL_BACKENDS = ("egl", "osmesa")
CANONICAL_MODEL_NAME = "canonical-humanoid"
DEFAULT_FPS = 30
DEFAULT_DURATION_S = 2.0
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
RESULT_SUFFIXES = (".npz", ".npy", ".csv")


class ExplorerError(Exception):
    """A violated precondition; the message is shown to the operator as-is."""


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def require_gl_backend(env: Mapping[str, str]) -> str:
    """Return the headless GL backend named by ``MUJOCO_GL``.

    Raises:
        ExplorerError: if the variable is unset or not a headless backend.
    """
    backend = env.get("MUJOCO_GL", "")
    if not backend:
        raise ExplorerError(
            "MUJOCO_GL is not set; set MUJOCO_GL=egl (GPU/Mesa) or "
            "MUJOCO_GL=osmesa (pure software) for headless rendering"
        )
    if backend not in HEADLESS_GL_BACKENDS:
        raise ExplorerError(
            f"MUJOCO_GL={backend!r} cannot render without a window; "
            "use egl or osmesa inside the container"
        )
    return backend


def _require_even_positive(width: int, height: int) -> None:
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ExplorerError(
            f"width and height must be positive even integers for H.264 "
            f"(got {width}x{height})"
        )


def _require_timing(duration_s: float, fps: int) -> None:
    if not duration_s > 0:
        raise ExplorerError(f"duration must be > 0 seconds (got {duration_s})")
    if fps <= 0:
        raise ExplorerError(f"fps must be a positive integer (got {fps})")


def probe_gl() -> str | None:
    """Try to create a tiny offscreen render context; return an error or None."""
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>"
        )
        renderer = mujoco.Renderer(model, 16, 16)
        try:
            renderer.update_scene(mujoco.MjData(model))
            renderer.render()
        finally:
            renderer.close()
    except Exception as exc:  # noqa: BLE001 - any GL failure is a preflight result
        return f"{type(exc).__name__}: {exc}"
    return None


# ---------------------------------------------------------------------------
# Model + trajectory
# ---------------------------------------------------------------------------


def load_mujoco_model(model_arg: str) -> Any:
    """Compile a MuJoCo model from a path, or the built-in canonical humanoid.

    Raises:
        ExplorerError: if the file is missing or does not compile.
    """
    import mujoco

    try:
        if model_arg == CANONICAL_MODEL_NAME:
            from src.engines.physics_engines.mujoco._golf_swing_canonical_xml import (
                CANONICAL_GOLF_HUMANOID_XML,
            )

            return mujoco.MjModel.from_xml_string(CANONICAL_GOLF_HUMANOID_XML)
        path = Path(model_arg)
        if not path.is_file():
            raise ExplorerError(f"model file not found: {model_arg}")
        return mujoco.MjModel.from_xml_path(str(path))
    except ExplorerError:
        raise
    except Exception as exc:  # noqa: BLE001 - mujoco raises bare ValueError/Exception
        raise ExplorerError(f"could not compile model {model_arg}: {exc}") from exc


def load_trajectory(path: Path, nq: int) -> Any:
    """Load a ``(frames, nq)`` joint-position array from npz/npy/csv.

    Raises:
        ExplorerError: on a missing file, unknown format, wrong width or NaN/inf.
    """
    import numpy as np

    if not path.is_file():
        raise ExplorerError(f"results file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as archive:
            if "qpos" not in archive:
                raise ExplorerError(f"{path.name} has no 'qpos' array")
            qpos = np.asarray(archive["qpos"], dtype=float)
    elif suffix == ".npy":
        qpos = np.asarray(np.load(path), dtype=float)
    elif suffix == ".csv":
        qpos = np.atleast_2d(np.loadtxt(path, delimiter=","))
    else:
        raise ExplorerError(
            f"unsupported results format {suffix!r}; use one of {RESULT_SUFFIXES}"
        )
    if qpos.ndim != 2 or qpos.shape[1] != nq or qpos.shape[0] < 1:
        raise ExplorerError(
            f"results must be shaped (frames, nq={nq}); got {qpos.shape}"
        )
    if not np.isfinite(qpos).all():
        raise ExplorerError(f"results {path.name} contain non-finite values")
    return qpos


def simulate_trajectory(model: Any, duration_s: float, fps: int) -> Any:
    """Run the model passively and sample ``qpos`` at ``fps``.

    Postcondition: returns ``(round(duration_s * fps), nq)`` finite samples.
    """
    import mujoco
    import numpy as np

    _require_timing(duration_s, fps)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    frames = max(1, round(duration_s * fps))
    samples = np.empty((frames, model.nq))
    for index in range(frames):
        samples[index] = data.qpos
        target = (index + 1) / fps
        while data.time < target:
            previous = data.time
            mujoco.mj_step(model, data)
            if data.time <= previous:
                raise ExplorerError(
                    "simulation became unstable (MuJoCo reset the state at "
                    f"t={previous:.4f}s); use --results to replay a trajectory "
                    "or a model with a stable passive pose"
                )
    if not np.isfinite(samples).all():
        raise ExplorerError("simulation diverged (non-finite qpos); check the model")
    return samples


# ---------------------------------------------------------------------------
# Frame source + writer (the single export path)
# ---------------------------------------------------------------------------


def iter_frames(model: Any, qpos: Any, width: int, height: int) -> Iterator[Any]:
    """Yield one RGB ``(height, width, 3)`` uint8 frame per ``qpos`` row."""
    import mujoco

    _require_even_positive(width, height)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = model.stat.center
    camera.distance = max(float(model.stat.extent) * 2.2, 0.5)
    camera.azimuth, camera.elevation = 135.0, -15.0
    renderer = mujoco.Renderer(model, height, width)
    try:
        for row in qpos:
            data.qpos[:] = row
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            yield renderer.render().copy()
    finally:
        renderer.close()


def write_mp4(frames: Iterator[Any], out: Path, fps: int) -> int:
    """Encode frames to H.264 MP4 at ``out``; return the frame count."""
    import imageio.v2 as imageio

    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    writer = imageio.get_writer(
        str(out),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        quality=7,
    )
    try:
        for frame in frames:
            writer.append_data(frame)
            count += 1
    finally:
        writer.close()
    if count == 0:
        raise ExplorerError("no frames were rendered; refusing to leave an empty MP4")
    return count


def verify_mp4(out: Path) -> dict[str, int]:
    """Re-open the MP4 and report decoded frame count and size."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(out))
    try:
        width, height = reader.get_meta_data()["size"]
        frames = sum(1 for _ in reader)
    finally:
        reader.close()
    return {"verified_frames": frames, "width": int(width), "height": int(height)}


def export_video(  # noqa: PLR0913 - explicit export contract, all keyword-only
    *,
    model_arg: str,
    out: Path,
    results: Path | None,
    duration_s: float,
    fps: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Open a model, run or replay it, write an MP4 and verify it.

    Returns a summary dict (frames, verified_frames, size, bytes, source).
    """
    if out.suffix.lower() != ".mp4":
        raise ExplorerError(f"output must be an .mp4 file (got {out.name})")
    _require_even_positive(width, height)
    _require_timing(duration_s, fps)
    model = load_mujoco_model(model_arg)
    if results is not None:
        qpos, source = load_trajectory(results, model.nq), "results"
    else:
        qpos, source = simulate_trajectory(model, duration_s, fps), "simulation"
    frames = write_mp4(iter_frames(model, qpos, width, height), out, fps)
    summary: dict[str, Any] = {
        "out": str(out),
        "model": model_arg,
        "source": source,
        "fps": fps,
        "frames": frames,
        "bytes": out.stat().st_size,
    }
    summary.update(verify_mp4(out))
    if summary["verified_frames"] != frames:
        raise ExplorerError(
            f"MP4 verification failed: wrote {frames} frames, "
            f"decoded {summary['verified_frames']}"
        )
    return summary


# ---------------------------------------------------------------------------
# Engine report
# ---------------------------------------------------------------------------


def _smoke_mujoco(module: Any) -> str:
    model = module.MjModel.from_xml_string(
        "<mujoco><worldbody><body><joint/><geom size='.1'/></body></worldbody></mujoco>"
    )
    data = module.MjData(model)
    module.mj_step(model, data)
    return f"mujoco {module.__version__}"


def _smoke_pinocchio(module: Any) -> str:
    model = module.buildSampleModelHumanoid()
    data = model.createData()
    module.forwardKinematics(model, data, module.neutral(model))
    return f"pinocchio {module.__version__} (nq={model.nq})"


def _smoke_drake(module: Any) -> str:
    from importlib.metadata import version

    from pydrake.multibody.plant import MultibodyPlant

    MultibodyPlant(0.001)
    return f"drake {version('drake')} (module {module.__name__})"


def _smoke_opensim(module: Any) -> str:
    model = module.Model()
    model.initSystem()
    return f"opensim {module.GetVersion()}"


def _smoke_pyqt6(module: Any) -> str:
    return f"Qt {module.QT_VERSION_STR}"


# name -> (module to import, smoke callable or None). Smokes are tiny on purpose.
ENGINE_PROBES: dict[str, tuple[str, Callable[[Any], str] | None]] = {
    "mujoco": ("mujoco", _smoke_mujoco),
    "pinocchio": ("pinocchio", _smoke_pinocchio),
    "drake": ("pydrake.common", _smoke_drake),
    "opensim": ("opensim", _smoke_opensim),
    "myosuite": ("myosuite", None),
    "pyqt6": ("PyQt6.QtCore", _smoke_pyqt6),
}


def engine_report() -> dict[str, dict[str, str]]:
    """Probe every engine; never raises. Status is ok, missing or broken."""
    report: dict[str, dict[str, str]] = {}
    for name, (module_name, smoke) in ENGINE_PROBES.items():
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            report[name] = {"status": "missing", "detail": f"{module_name}: {exc}"}
            continue
        except Exception as exc:  # noqa: BLE001 - a broken install is a report row
            report[name] = {
                "status": "broken",
                "detail": f"{module_name}: {type(exc).__name__}: {exc}",
            }
            continue
        try:
            detail = smoke(module) if smoke else getattr(module, "__version__", "")
        except Exception as exc:  # noqa: BLE001 - smoke failure is a report row
            report[name] = {
                "status": "broken",
                "detail": f"smoke failed: {type(exc).__name__}: {exc}",
            }
            continue
        report[name] = {"status": "ok", "detail": str(detail) or module_name}
    return report


# ---------------------------------------------------------------------------
# App screenshot
# ---------------------------------------------------------------------------


def app_screenshot(
    out: Path, model: Path | None, settle_ms: int = 2500
) -> dict[str, Any]:
    """Construct the real Model Explorer window and save a screenshot of it."""
    if not (os.environ.get("QT_QPA_PLATFORM") or os.environ.get("DISPLAY")):
        raise ExplorerError(
            "no display: set QT_QPA_PLATFORM=offscreen or provide DISPLAY"
        )
    if model is not None and not model.is_file():
        raise ExplorerError(f"model file not found: {model}")
    try:
        import mujoco  # noqa: F401 - must precede Qt imports (see launch_model_explorer)
        from PyQt6.QtCore import QEventLoop, QTimer
        from PyQt6.QtWidgets import QApplication

        from src.tools.model_explorer.main_window import URDFGeneratorWindow
    except ImportError as exc:
        raise ExplorerError(f"GUI stack is not importable: {exc}") from exc

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = URDFGeneratorWindow()
    if model is not None and not window.load_model(model):
        raise ExplorerError(f"Model Explorer could not load {model}")
    window.show()
    loop = QEventLoop()
    QTimer.singleShot(settle_ms, loop.quit)
    loop.exec()
    app.processEvents()
    image = window.grab().toImage()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(out)):
        raise ExplorerError(f"could not write screenshot to {out}")
    colours = {
        image.pixel(x, y)
        for x in range(0, image.width(), 9)
        for y in range(0, image.height(), 9)
    }
    window.close()
    return {
        "out": str(out),
        "width": image.width(),
        "height": image.height(),
        "distinct_colours": len(colours),
        "platform": app.platformName(),
        "model": str(model) if model else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probe-gl", help="check the MUJOCO_GL backend can render")
    sub.add_parser("engines", help="JSON engine availability + smoke report")

    export = sub.add_parser("export", help="open a model and export an MP4")
    export.add_argument(
        "--model",
        default=CANONICAL_MODEL_NAME,
        help=f"MJCF/URDF path, or '{CANONICAL_MODEL_NAME}' (default)",
    )
    export.add_argument("--out", required=True, type=Path)
    export.add_argument("--results", type=Path, help="npz(qpos)/npy/csv to replay")
    export.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    export.add_argument("--fps", type=int, default=DEFAULT_FPS)
    export.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    export.add_argument("--height", type=int, default=DEFAULT_HEIGHT)

    shot = sub.add_parser("app-screenshot", help="screenshot the real Qt app")
    shot.add_argument("--out", required=True, type=Path)
    shot.add_argument("--model", type=Path)
    shot.add_argument("--settle-ms", type=int, default=2500)
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.command == "probe-gl":
        backend = require_gl_backend(os.environ)
        error = probe_gl()
        if error:
            raise ExplorerError(
                f"MUJOCO_GL={backend} cannot create a context ({error}); "
                "try MUJOCO_GL=osmesa, or on a GPU host pass --gpus all"
            )
        print(f"GL ok: MUJOCO_GL={backend}")
        return 0
    if args.command == "engines":
        print(json.dumps(engine_report(), indent=2))
        return 0
    if args.command == "export":
        if sys.platform.startswith("linux"):
            require_gl_backend(os.environ)
        summary = export_video(
            model_arg=args.model,
            out=args.out,
            results=args.results,
            duration_s=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
        print(json.dumps(summary))
        return 0
    print(json.dumps(app_screenshot(args.out, args.model, args.settle_ms)))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; maps :class:`ExplorerError` to a clear stderr message."""
    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except ExplorerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return (
            EXIT_USAGE
            if "MUJOCO_GL" in str(exc) or "display" in str(exc)
            else (EXIT_BAD_INPUT)
        )


if __name__ == "__main__":
    raise SystemExit(main())
