"""Run the unchanged parent verifier in its exact preserved source environment."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .supplement_publication import sha256


def _repo_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


_WORKER = r"""
import importlib.util
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()

def _load_module(name: str, relative: str):
    path = (root / relative).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f'parent verifier cannot load {relative}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

pipeline = _load_module(
    'matched_supplement_pipeline',
    'src/tools/shot_pattern_analysis/matched_supplement_pipeline.py',
)
export_mod = _load_module(
    'matched_export',
    'src/tools/shot_pattern_analysis/matched_export.py',
)
matched_source_snapshot = export_mod.matched_source_snapshot
if Path(pipeline.__file__).resolve() != (
    root / 'src/tools/shot_pattern_analysis/matched_supplement_pipeline.py'
).resolve():
    raise ValueError('parent verifier import origin differs')
expected=json.loads(sys.stdin.read())
if matched_source_snapshot()!=expected:
    raise ValueError('parent execution/runtime/native differs')
result = ({"execution": matched_source_snapshot(), "report_sources": pipeline._report_source_hashes()}
          if sys.argv[3] == "probe" else pipeline.load_verified_supplement_artifacts(Path(sys.argv[2])))
if matched_source_snapshot()!=expected:
    raise ValueError('parent execution changed while validating')
print(json.dumps(result))
"""


def check_verifier_pin(binding: dict[str, Any], *, base: Path) -> tuple[Path, Path]:
    """Check immutable identity; relative checkout paths may be reconstructed."""
    root = (base / binding["source_root"]).resolve()
    if "origin_pin_relative_path" in binding:
        origin = (base / binding["origin_pin_relative_path"]).resolve()
        if not origin.is_file() or sha256(origin) != binding.get("origin_pin_sha256"):
            raise ValueError("parent origin pin changed")
        original = json.loads(origin.read_text())
        if (origin.parent / original["source_root"]).resolve() != root or any(
            binding.get(key) != value
            for key, value in original.items()
            if key != "source_root"
        ):
            raise ValueError("parent origin pin identity differs")
    executable = Path(binding["python_executable"]).absolute()
    if not executable.is_file() or sha256(executable) != binding["python_sha256"]:
        raise ValueError("parent Python executable differs")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != binding["source_commit"]:
        raise ValueError("parent source commit differs")
    source_map = binding["source_sha256"]
    study_paths = {
        _repo_relative(path, root)
        for folder in (
            "src/tools/shot_pattern_analysis",
            "scripts/research/shot_pattern_analysis",
        )
        for path in (root / folder).rglob("*.py")
    }
    if not study_paths.issubset(source_map):
        raise ValueError("parent pin omits study source")
    required = "src/tools/shot_pattern_analysis/matched_supplement_pipeline.py"
    if required not in source_map:
        raise ValueError("parent pin omits verifier source")
    for name, digest in source_map.items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or sha256(path) != digest
        ):
            raise ValueError("parent pinned source differs")
    if not isinstance(binding.get("execution"), dict) or not binding["execution"]:
        raise ValueError("parent execution pin missing")
    return root, executable


def _run_frozen(
    parent: Path, binding: dict[str, Any], *, base: Path, mode: str
) -> dict:
    """Use isolated imports and exact runtime checks, never current-source fallback."""
    root, executable = check_verifier_pin(binding, base=base)
    environment = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"}
    }
    environment.update(
        QT_QPA_PLATFORM="offscreen",
        MPLBACKEND="Agg",
        MUJOCO_GL="egl",
        SDL_VIDEODRIVER="dummy",
    )
    result = subprocess.run(
        [str(executable), "-I", "-c", _WORKER, str(root), str(parent.resolve()), mode],
        cwd=root,
        env=environment,
        input=json.dumps(binding["execution"]),
        capture_output=True,
        text=True,
        check=True,
    )
    check_verifier_pin(binding, base=base)
    return json.loads(result.stdout)


def verify_parent_in_frozen_source(
    parent: Path, binding: dict[str, Any], *, base: Path
) -> dict:
    """Run full original artifact validation in the pinned parent environment."""
    return _run_frozen(parent, binding, base=base, mode="verify")


def probe_parent_environment(binding: dict[str, Any], *, base: Path) -> dict:
    """Qualify sources/runtime without claiming any artifact verification."""
    return _run_frozen(base, binding, base=base, mode="probe")
