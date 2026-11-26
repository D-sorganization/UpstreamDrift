"""CLI for headless grip build preview (#15126)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.shared.python.grip_build.builder import build_lead_fingers
from src.shared.python.grip_spec.types import GripSpec
from src.tools.grip_build_preview.export import write_gif, write_mp4, write_still_strip
from src.tools.grip_build_preview.preview import generate_preview_rgb_frames
from src.tools.grip_build_preview.settings import DEFAULT_PREVIEW_SETTINGS


def _load_spec(path: Path) -> GripSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("spec JSON must be an object")
    return GripSpec.from_dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay HRG grip build stages (GB-4 preview, lead fingers on GB-2)."
    )
    parser.add_argument("--spec", type=Path, required=True, help="GripSpec JSON path")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output clip (.mp4 / .gif) or still strip (.png)",
    )
    parser.add_argument(
        "--compare-spec",
        type=Path,
        default=None,
        help="Optional second spec for side-by-side still comparison",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--stills-cols", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = _load_spec(args.spec)
    built = build_lead_fingers(spec)
    frames = generate_preview_rgb_frames(spec, built, DEFAULT_PREVIEW_SETTINGS)
    if args.compare_spec is not None:
        other = _load_spec(args.compare_spec)
        other_built = build_lead_fingers(other)
        other_frames = generate_preview_rgb_frames(
            other, other_built, DEFAULT_PREVIEW_SETTINGS
        )
        paired = []
        for left, right in zip(frames, other_frames, strict=False):
            import numpy as np

            paired.append(np.concatenate([left, right], axis=1))
        frames = tuple(paired)
    suffix = args.out.suffix.lower()
    if suffix == ".gif":
        write_gif(args.out, frames, fps=args.fps)
    elif suffix == ".mp4":
        write_mp4(args.out, frames, fps=args.fps)
    elif suffix in {".png", ".jpg", ".jpeg"}:
        write_still_strip(args.out, frames, cols=args.stills_cols)
    else:
        raise SystemExit(f"unsupported output suffix {suffix!r}")
    sys.stdout.write(
        json.dumps(
            {
                "frames": len(frames),
                "status": built.status.value,
                "spec_sha256": built.receipt.spec_sha256,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
