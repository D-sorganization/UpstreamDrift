"""Reproducible CPU-only capture playback benchmark; no cameras or Qt required.

Run with ``python -m scripts.benchmark_capture_responsiveness --output report.json``.
Timing is diagnostic, never a portable CI pass/fail threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np

from src.tools.capture_rig.layout_model import (
    Cell,
    LayoutSpec,
    SourceRef,
    Tile,
    compose,
)
from src.tools.capture_rig.player import VideoReader

SOURCE_SIZE = (1280, 720)
CANVAS_SIZE = (960, 540)
WARMUP = 5


def measure(operation: Callable[[int], object], samples: int) -> dict[str, Any]:
    """Warm a path, then return bounded wall-clock distribution in milliseconds."""
    for index in range(WARMUP):
        operation(index)
    elapsed = []
    for index in range(WARMUP, WARMUP + samples):
        started = perf_counter()
        operation(index)
        elapsed.append((perf_counter() - started) * 1000)
    return {
        "samples": samples,
        "median_ms": round(float(np.median(elapsed)), 3),
        "p95_ms": round(float(np.percentile(elapsed, 95)), 3),
        "max_ms": round(max(elapsed), 3),
    }


def run_benchmark(samples: int = 60) -> dict[str, Any]:
    """Measure one/three/six-view composition and sequential/duplicate decoding."""
    import cv2

    if not 1 <= samples <= 300:
        raise ValueError("samples must be between 1 and 300")
    rng = np.random.default_rng(9851)
    frame = rng.integers(0, 256, (SOURCE_SIZE[1], SOURCE_SIZE[0], 3), dtype=np.uint8)
    composition = []
    for views in (1, 3, 6):
        cols = min(views, 3)
        sources = [SourceRef("recorded", f"camera-{i}") for i in range(views)]
        spec = LayoutSpec(
            f"benchmark-{views}",
            rows=(views + cols - 1) // cols,
            cols=cols,
            tiles=tuple(
                Tile(source, Cell(i // cols, i % cols))
                for i, source in enumerate(sources)
            ),
            canvas=CANVAS_SIZE,
        )
        frames = {source.key: frame for source in sources}

        def draw(
            _index: int, layout: LayoutSpec = spec, inputs: dict[str, Any] = frames
        ) -> object:
            return compose(inputs, layout)

        composition.append({"views": views, **measure(draw, samples)})
    decode = {}
    with tempfile.TemporaryDirectory(prefix="capture-benchmark-") as directory:
        path = Path(directory) / "source.avi"
        writer = cv2.VideoWriter(
            str(path), cast(Any, cv2).VideoWriter_fourcc(*"MJPG"), 30, SOURCE_SIZE
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV MJPEG writer is unavailable")
        try:
            for _ in range(samples + WARMUP):
                writer.write(frame)
        finally:
            writer.release()
        for label, repeats in (("sequential", 1), ("duplicate", 2)):
            with VideoReader(path) as reader:

                def read(index: int, repeats: int = repeats) -> None:
                    for _ in range(repeats):
                        if reader.read(index) is None:
                            raise RuntimeError(
                                f"Fixture decode failed at frame {index}"
                            )

                decode[label] = measure(read, samples)
    player = Path(__file__).resolve().parents[1] / "src/tools/capture_rig/player.py"
    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
        },
        "fixture": {
            "source_size": SOURCE_SIZE,
            "canvas_size": CANVAS_SIZE,
            "codec": "MJPG",
            "seed": 9851,
            "warmup": WARMUP,
        },
        "player_sha256": hashlib.sha256(player.read_bytes()).hexdigest(),
        "composition": composition,
        "decode": decode,
        "limitations": "Synthetic noisy video; excludes camera capture, Qt painting, inference and storage contention. Compare on the same host.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_benchmark(args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
