"""Run the unchanged parent verifier in its exact preserved source environment."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .supplement_publication import sha256

_WORKER = r"""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root))
from src.tools.shot_pattern_analysis import matched_supplement_pipeline as pipeline
from src.tools.shot_pattern_analysis.matched_export import matched_source_snapshot
if Path(pipeline.__file__).resolve()!=root/'src/tools/shot_pattern_analysis/matched_supplement_pipeline.py':
    raise ValueError('parent verifier import origin differs')
expected=json.loads(sys.stdin.read())
if matched_source_snapshot()!=expected:
    raise ValueError('parent execution/runtime/native differs')
result=pipeline.load_verified_supplement_artifacts(Path(sys.argv[2]))
if matched_source_snapshot()!=expected:
    raise ValueError('parent execution changed while validating')
print(json.dumps(result))
"""


def check_verifier_pin(binding: dict[str, Any], *, base: Path) -> tuple[Path, Path]:
    """Check immutable identity; relative checkout paths may be reconstructed."""
    root = (base / binding["source_root"]).resolve()
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
        str(path.relative_to(root))
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


def verify_parent_in_frozen_source(
    parent: Path, binding: dict[str, Any], *, base: Path
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
        [str(executable), "-I", "-c", _WORKER, str(root), str(parent.resolve())],
        cwd=root,
        env=environment,
        input=json.dumps(binding["execution"]),
        capture_output=True,
        text=True,
        check=True,
    )
    check_verifier_pin(binding, base=base)
    return json.loads(result.stdout)
