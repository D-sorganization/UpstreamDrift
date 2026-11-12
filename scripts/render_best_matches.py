"""Render the best MuJoCo and OpenSim matches headless from SHA-256-pinned inputs.

MATCH-3 (#14445). A missing or changed input stops the run before rendering.
Each model gets SUMMARY.md with fit and hashes (from the #14443 ledger); videos
go to --out, outside git; private inputs appear only as placeholders.

Usage: python3 scripts/render_best_matches.py --out DIR --mujoco-bundle B.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.contracts import postcondition, precondition  # noqa: E402

DEFAULT_LEDGER = (
    REPO_ROOT / "docs" / "development" / "model_matching" / "model_matching_status.json"
)
MODELS_DIR = REPO_ROOT / "docs" / "development" / "full_body_models"
V1_SPEC = MODELS_DIR / "full_body_spec_anthro_driver.json"
V2_SPEC = MODELS_DIR / "full_body_spec_anthro_driver_v2.json"
OS9_DIR = "docs/development/opensim_tour_matching/evidence/os9_address_registration"
GEOMETRY_DIR = "shared/models/opensim/opensim-models/Geometry"
HEADLESS_ENV = {
    "MUJOCO_GL": "egl",
    "MPLBACKEND": "Agg",
    "QT_QPA_PLATFORM": "offscreen",
    "SDL_VIDEODRIVER": "dummy",
    "NATIVE_VIEWER_XVFB": "1",
}
VIDEO_SUFFIXES = (".mp4", ".png")
BUNDLE_SHA256 = "bfa1a9a86cc84c8c7d2f1bce821750d10564d1aa5f59549d082c87e855cdfe91"
MODEL_SHA256 = "56c6c6f0e947ae92d2b6041135e049b68284d46c70fee135339ba3e3f447393e"
STATES_SHA256 = "f489d41b68e84df3addab2ade825ae0266e672e60ec3abd083275f9f8c76d3f2"
MUJOCO_FLAGS = (
    "--swing driver --club Driver --engines mujoco --size 1920x1080 --fps 120"
    " --speeds 1,0.5"
)
XVFB_SCREEN = "-screen 0 1920x1200x24"


@dataclass(frozen=True)
class PinnedInput:
    """One render input; ``repo_path`` is None for private inputs given on the CLI."""

    name: str
    repo_path: str | None
    sha256: str


@dataclass
class RenderTarget:
    """A best-match model to render and the ledger engine that describes it."""

    model_id: str
    ledger_engine: str
    inputs: tuple[PinnedInput, ...]
    trunk_v2: bool = False  # migrate the fit onto the v2 lumbar+thoracic trunk


TARGETS: tuple[RenderTarget, ...] = (
    RenderTarget(
        model_id="mujoco_driver",
        ledger_engine="mujoco",
        inputs=(PinnedInput("bundle", None, BUNDLE_SHA256),),
        trunk_v2=True,
    ),
    RenderTarget(
        model_id="opensim_os9",
        ledger_engine="opensim",
        inputs=(
            PinnedInput(
                "model",
                f"{OS9_DIR}/golf_humanoid_scaled_tour_markers_address_registered.osim",
                MODEL_SHA256,
            ),
            PinnedInput("states", f"{OS9_DIR}/ik_states_continuous.sto", STATES_SHA256),
        ),
    ),
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_inputs(target: RenderTarget, paths: dict[str, Path]) -> dict[str, str]:
    """Return ``{name: sha256}``; raise ValueError on a missing or changed input."""
    hashes: dict[str, str] = {}
    for pinned in target.inputs:
        path = paths.get(pinned.name)
        if path is None or not path.is_file():
            raise ValueError(f"{target.model_id}: input {pinned.name!r} is missing")
        actual = sha256_file(path)
        if actual != pinned.sha256:
            raise ValueError(
                f"{target.model_id}: {pinned.name} sha256 {actual} != pinned "
                f"{pinned.sha256}"
            )
        hashes[pinned.name] = actual
    return hashes


@precondition(lambda target, paths, out_dir, python: bool(python))
@postcondition(lambda result: len(result) > 0)
def build_command(
    target: RenderTarget, paths: dict[str, Path], out_dir: Path, python: str
) -> list[str]:
    """Headless renderer argv for ``target``; identical inputs give identical argv."""
    if target.model_id == "mujoco_driver":
        return [
            python,
            *("-m", "src.tools.native_viewer_export", "--bundle"),
            str(paths["bundle"]),
            *MUJOCO_FLAGS.split(),
            *("--out", str(out_dir)),
        ]
    return [
        *("xvfb-run", "-a", "-s", XVFB_SCREEN, python),
        str(REPO_ROOT / "scripts" / "render_osim_ik_states.py"),
        *("--model", str(paths["model"]), "--states", str(paths["states"])),
        *("--geometry", str(REPO_ROOT / GEOMETRY_DIR), "--out", str(out_dir)),
    ]


def load_ledger_entry(ledger_path: Path, engine: str) -> dict[str, Any]:
    """The #14443 ledger entry for ``engine``; ValueError when absent."""
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    for entry in ledger.get("engines", []):
        if entry.get("engine") == engine:
            return entry
    raise ValueError(f"ledger has no entry for engine {engine!r}")


def _metric(value: float | None, unit: str) -> str:
    return "not measured" if value is None else f"{value:.1f} {unit}"


def _hash_list(hashes: dict[str, str]) -> str:
    return "; ".join(f"{name} `{digest}`" for name, digest in sorted(hashes.items()))


def _reproduction(target: RenderTarget) -> str:
    placeholders = {p.name: Path(p.repo_path or f"<{p.name}>") for p in target.inputs}
    argv = build_command(target, placeholders, Path("<out>"), "python3")
    return " ".join(arg.replace(str(REPO_ROOT) + "/", "") for arg in argv)


def _trunk_lines(target: RenderTarget) -> list[str]:
    if not target.trunk_v2:
        return []
    return [
        "- Trunk: full-body-v2 lumbar + thoracic segments "
        f"(`{V2_SPEC.name}`); v1 fit coordinates migrated by TrunkMigrator "
        "(replay, not a refit; the marker RMS above is the v1 fit's)"
    ]


def render_summary(
    target: RenderTarget,
    entry: dict[str, Any],
    input_hashes: dict[str, str],
    output_hashes: dict[str, str],
    source_commit: str,
) -> str:
    """Markdown summary sheet: source, method, parameters, fit and limitations."""
    status = "verified" if entry["verified"] else "UNVERIFIED"
    lines = [
        f"# Best Match: {entry['model']}",
        "",
        f"- Model id: `{target.model_id}` (engine `{entry['engine']}`)",
        f"- Source commit: `{source_commit}`",
        f"- Fit date: {entry['date']}; evidence ({status}): {entry['evidence']}",
        f"- Method: {entry['approach']}",
        f"- Parameters: {entry['key_parameters']}",
        f"- Marker RMS: {_metric(entry['marker_rms_mm'], 'mm')}",
        f"- Joint-angle RMS: {_metric(entry['joint_angle_rms_deg'], 'deg')}",
        *_trunk_lines(target),
        f"- Verdict: {entry['verdict']}",
        f"- Limitations and blockers: {entry['blocker']}",
        f"- Inputs (SHA-256): {_hash_list(input_hashes)}",
        f"- Outputs (SHA-256): {_hash_list(output_hashes)}",
        "",
        "## Reproduction",
        "",
        f"`{_reproduction(target)}`",
        "",
        "An animation is not physical acceptance; see the verdict above.",
        "",
    ]
    return "\n".join(lines)


def _input_paths(target: RenderTarget, mujoco_bundle: Path | None) -> dict[str, Path]:
    return {
        p.name: REPO_ROOT / p.repo_path if p.repo_path else mujoco_bundle or Path()
        for p in target.inputs
    }


def _migrated_bundle(bundle: Path, out_dir: Path) -> Path:
    """A v2 same-input bundle from the pinned v1 bundle (lumbar+thoracic trunk)."""
    from src.shared.python.motion_matching.best_match_v2 import migrate_bundle

    out_dir.mkdir(parents=True, exist_ok=True)
    return migrate_bundle(bundle, V1_SPEC, V2_SPEC, out_dir / "bundle_v2.npz")


def _render(target: RenderTarget, args: argparse.Namespace, commit: str) -> None:
    paths = _input_paths(target, args.mujoco_bundle)
    input_hashes = verify_inputs(target, paths)
    out_dir = args.out / target.model_id
    if target.trunk_v2 and not args.dry_run:
        paths = {**paths, "bundle": _migrated_bundle(paths["bundle"], out_dir)}
    command = build_command(target, paths, out_dir, sys.executable)
    print(" ".join(command))  # noqa: T201
    if args.dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **HEADLESS_ENV, "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT}/src"}
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)  # noqa: S603
    outputs = {
        p.name: sha256_file(p)
        for p in sorted(out_dir.iterdir())
        if p.suffix in VIDEO_SUFFIXES
    }
    entry = load_ledger_entry(args.ledger, target.ledger_engine)
    summary = render_summary(target, entry, input_hashes, outputs, commit)
    (out_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mujoco-bundle", type=Path)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--only", choices=[t.model_id for t in TARGETS])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    git_head = ["git", "rev-parse", "HEAD"]
    commit = subprocess.check_output(git_head, cwd=REPO_ROOT, text=True).strip()  # noqa: S603
    for target in TARGETS:
        if args.only in (None, target.model_id):
            _render(target, args, commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
