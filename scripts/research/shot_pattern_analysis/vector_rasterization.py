"""Operational PDF/SVG vector rasterization helper for review workflows."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from PIL import Image

from scripts.research.shot_pattern_analysis.vector_inspection import inspect_vector

CONFIGURED_DENSITY_DPI: int = 144

_NODE_SHARP_SCRIPT: str = """
const sharp = require(process.argv[1]);
const inputPath = process.argv[2];
const outputPath = process.argv[3];
const maxW = parseInt(process.argv[4], 10);
const maxH = parseInt(process.argv[5], 10);

sharp(inputPath, { density: 144, limitInputPixels: 50000000 })
  .flatten({ background: '#ffffff' })
  .resize(maxW, maxH, { fit: 'inside' })
  .png()
  .toFile(outputPath)
  .then((info) => {
    const payload = {
      status: 'ok',
      node_version: process.version,
      sharp_versions: sharp.versions,
      output_info: info,
    };
    process.stdout.write(JSON.stringify(payload));
    process.exit(0);
  })
  .catch((err) => {
    process.stderr.write(String(err && err.stack ? err.stack : err));
    process.exit(1);
  });
"""


def _compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Precondition: path must be an existing regular file.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Target is not an existing file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@functools.lru_cache(maxsize=16)
def _get_pdftoppm_version(binary_path: str) -> dict[str, str]:
    """Extract pdftoppm version string if obtainable."""
    try:
        res = subprocess.run(
            [binary_path, "-v"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        combined = f"{res.stdout}\n{res.stderr}"
        match = re.search(r"pdftoppm version (\S+)", combined)
        if match:
            return {"pdftoppm": match.group(1)}
        return {"pdftoppm": "unknown"}
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return {"pdftoppm": "unavailable"}


def _calculate_svg_target_dimensions(viewport: dict[str, Any]) -> tuple[int, int]:
    """Derive max pixel dimensions from SVG viewport at 144 pixels/inch.

    Scales:
    - pt: 144 / 72 = 2.0 px/pt
    - px or unitless (CSS px): 144 / 96 = 1.5 px/px
    - in: 144.0 px/in
    - mm: 144.0 / 25.4 px/mm
    - cm: 144.0 / 2.54 px/cm

    Precondition: viewport must contain positive finite width and height with
    supported physical or CSS units.
    """
    if not viewport or "width" not in viewport or "height" not in viewport:
        raise ValueError(
            "Unsupported SVG qualification: missing or incomplete viewport width/height"
        )

    try:
        width_val = float(viewport["width"])
        height_val = float(viewport["height"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric viewport dimensions: {exc}") from exc

    width_unit = str(viewport.get("width_unit", "")).lower()
    height_unit = str(viewport.get("height_unit", "")).lower()

    scale_map: dict[str, float] = {
        "pt": 2.0,
        "px": 1.5,
        "": 1.5,
        "in": 144.0,
        "mm": 144.0 / 25.4,
        "cm": 144.0 / 2.54,
    }

    if width_unit not in scale_map:
        raise ValueError(f"Unsupported SVG viewport width unit: {width_unit!r}")
    if height_unit not in scale_map:
        raise ValueError(f"Unsupported SVG viewport height unit: {height_unit!r}")

    max_w = int(round(width_val * scale_map[width_unit]))
    max_h = int(round(height_val * scale_map[height_unit]))

    if max_w <= 0 or max_h <= 0:
        raise ValueError(
            f"Calculated non-positive SVG target dimensions: width={max_w}, height={max_h}"
        )

    return max_w, max_h


def _prepare_render_command(
    file_path: Path,
    target_dir: Path,
    inspection: dict[str, Any],
    tool_paths: dict[str, str | Path] | None,
) -> tuple[list[str], dict[str, str], Any, dict[str, int] | None]:
    """Validate explicit executables and derive a native render command."""
    vector_format = str(inspection.get("format", "")).lower()
    if tool_paths is None:
        raise ValueError("An explicit tool_paths configuration is required")
    required = ("pdftoppm",) if vector_format == "pdf" else ("node", "sharp")
    tools: dict[str, str] = {}
    for key in required:
        if key not in tool_paths:
            raise ValueError(f"Missing required tool path: {key}")
        tool = Path(tool_paths[key]).resolve()
        if key == "sharp":
            valid = tool.is_dir() and (tool / "package.json").is_file()
        else:
            valid = tool.is_file() and os.access(tool, os.X_OK)
        if not valid:
            raise ValueError(f"Invalid tool path for {key}: {tool}")
        tools[key] = str(tool)

    stem = file_path.stem
    output_png = target_dir / f"{stem}.png"

    target_dims: dict[str, int] | None = None
    if vector_format == "pdf":
        original_viewport = inspection.get("dimensions", {}).get("media_box")
        output_stem = target_dir / stem
        cmd = [
            tools["pdftoppm"],
            "-f",
            "1",
            "-singlefile",
            "-r",
            str(CONFIGURED_DENSITY_DPI),
            "-png",
            str(file_path),
            str(output_stem),
        ]
    elif vector_format == "svg":
        original_viewport = inspection.get("viewport", {})
        max_w, max_h = _calculate_svg_target_dimensions(original_viewport)
        target_dims = {"max_width": max_w, "max_height": max_h}
        cmd = [
            tools["node"],
            "-e",
            _NODE_SHARP_SCRIPT,
            tools["sharp"],
            str(file_path),
            str(output_png),
            str(max_w),
            str(max_h),
        ]
    else:
        raise ValueError(f"Unsupported vector format: {vector_format}")

    return cmd, tools, original_viewport, target_dims


def _renderer_versions(
    vector_format: str,
    mutation: bool,
    proc: subprocess.CompletedProcess[str],
    tools: dict[str, str],
) -> dict[str, Any]:
    """Read successful renderer metadata without trusting malformed stdout."""
    renderer_versions: dict[str, Any] = {}
    if vector_format == "pdf" and not mutation and proc.returncode == 0:
        renderer_versions = _get_pdftoppm_version(tools["pdftoppm"])
    elif vector_format == "svg":
        try:
            node_payload = json.loads(proc.stdout)
            if isinstance(node_payload, dict):
                if "node_version" in node_payload:
                    renderer_versions["node"] = node_payload["node_version"]
                if "sharp_versions" in node_payload:
                    renderer_versions["sharp_versions"] = node_payload["sharp_versions"]
        except json.JSONDecodeError:
            pass

    return renderer_versions


def _launch_render(
    cmd: list[str],
) -> tuple[subprocess.CompletedProcess[str], str | None]:
    """Preserve launch failures as diagnostics for the caller to save."""
    launch_error: str | None = None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        launch_error = f"{type(exc).__name__}: {exc}"
        proc = subprocess.CompletedProcess(cmd, -1, stdout="", stderr=launch_error)
    return proc, launch_error


@dataclass(frozen=True)
class _RasterOutput:
    success: bool
    error: str | None
    actual_pixel_shape: dict[str, int] | None
    output_png_sha: str | None
    output_png_path_str: str | None


def _inspect_render_output(
    file_path: Path,
    output_png: Path,
    proc: subprocess.CompletedProcess[str],
    mutation: bool,
) -> _RasterOutput:
    """Validate the native output while retaining partial-failure evidence."""
    success = False
    error: str | None = None
    actual_pixel_shape: dict[str, int] | None = None
    output_png_sha: str | None = None
    output_png_path_str: str | None = None

    if mutation:
        error = f"Post-render SHA256 mutation detected for {file_path.name}"
    elif proc.returncode != 0:
        error = (
            f"Renderer exited with code {proc.returncode}: "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    elif not output_png.is_file() or output_png.stat().st_size == 0:
        error = f"Expected output PNG not found or empty: {output_png}"
    else:
        try:
            decoded_sha = _compute_sha256(output_png)
            with Image.open(output_png) as img:
                if img.format != "PNG":
                    raise ValueError(f"Expected PNG format, got {img.format}")
                actual_w, actual_h = img.size
                if actual_w <= 0 or actual_h <= 0:
                    raise ValueError(f"Non-positive dimensions: {img.size}")
                img.load()
            output_png_sha = _compute_sha256(output_png)
            if output_png_sha != decoded_sha:
                raise ValueError("Output PNG changed during decode/hash validation")
            success = True
            actual_pixel_shape = {"width": actual_w, "height": actual_h}
            output_png_path_str = str(output_png.resolve())
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            error = f"Output PNG is undecodable: {exc}"
            output_png_path_str = str(output_png.resolve())

    return _RasterOutput(
        success, error, actual_pixel_shape, output_png_sha, output_png_path_str
    )


def render_vector(
    path: Path | str,
    expected_sha256: str,
    output_dir: Path | str,
    tool_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Render single vector asset (.pdf or .svg) to PNG with strict hash verification.

    Preconditions:
    - path must exist, be regular file, and match expected_sha256 before read.
    - output_dir must be a new directory; existing directories are rejected.
    - tools must be available or specified via tool_paths.

    Postconditions:
    - input file SHA-256 remains unchanged after subprocess execution.
    - visual_approval is unconditionally False.
    - diagnostics dictionary is JSON-serializable.
    - partial failure directories and outputs are preserved for review.
    """
    file_path = Path(path)
    target_dir = Path(output_dir)

    if target_dir.exists():
        raise FileExistsError(f"Output directory already exists: {target_dir}")

    # Validate file format and metadata using canonical inspect_vector helper
    inspection = inspect_vector(file_path, expected_sha256)
    vector_format = str(inspection.get("format", "")).lower()

    # Re-verify pre-subprocess SHA-256
    pre_sha = _compute_sha256(file_path)
    if pre_sha.lower() != expected_sha256.lower():
        raise ValueError(
            f"Pre-render SHA256 mismatch for {file_path.name}: "
            f"expected {expected_sha256}, got {pre_sha}"
        )

    cmd, tools, original_viewport, target_dims = _prepare_render_command(
        file_path, target_dir, inspection, tool_paths
    )
    output_png = target_dir / f"{file_path.stem}.png"
    original_viewbox = inspection.get("view_box")

    target_dir.mkdir(parents=True, exist_ok=False)
    # Preserve diagnostics even when a native launch or post-render guard fails.
    proc, launch_error = _launch_render(cmd)
    try:
        post_sha: str | None = _compute_sha256(file_path)
    except OSError:
        post_sha = None
    mutation = post_sha is None or post_sha.lower() != expected_sha256.lower()

    renderer_versions = _renderer_versions(vector_format, mutation, proc, tools)

    output = _inspect_render_output(file_path, output_png, proc, mutation)

    record = {
        "success": output.success,
        "format": vector_format,
        "input_path": str(file_path.resolve()),
        "input_sha256": pre_sha,
        "post_input_sha256": post_sha,
        "inspection": inspection,
        "source_sha256": _compute_sha256(Path(__file__)),
        "launch_error": launch_error,
        "expected_sha256": expected_sha256,
        "output_dir": str(target_dir.resolve()),
        "output_png_path": output.output_png_path_str,
        "output_png_sha256": output.output_png_sha,
        "output_dimensions": output.actual_pixel_shape,
        "actual_pixel_shape": output.actual_pixel_shape,
        "configured_density": CONFIGURED_DENSITY_DPI,
        "original_viewport": original_viewport,
        "original_viewbox": original_viewbox,
        "target_max_dimensions": target_dims,
        "command": cmd,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "exit_code": proc.returncode,
        "renderer_versions": renderer_versions,
        "visual_approval": False,
        "claim": (
            "rasterization diagnostics only; visual_approval is false; "
            "does not prove visual acceptance, glyph visibility, or layout; "
            "no whole-publication claim"
        ),
        "error": output.error,
    }

    (target_dir / "raster_diagnostics.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    if mutation:
        raise ValueError(output.error)
    return record
