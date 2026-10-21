"""Classify pull-request file changes for optional CI lanes (#13571 / UD-CI-010a).

Precondition: ``changed_files`` entries are repository-relative POSIX paths.
Postcondition: ``should_run_opensim_lane`` is true on non-PR events, when the
``ci:opensim`` label is present, or when any changed file matches the pinned
OpenSim path list; otherwise it is false for doc-only diffs and unrelated code.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "scripts/config/ci_opensim_path_patterns.json"


class ChangeCategory(str, Enum):
    DOC = "doc"
    WORKFLOW = "workflow"
    SHARED_PHYSICS_CORE = "shared_physics_core"
    OPENSIM = "opensim"
    OTHER = "other"


@dataclass(frozen=True)
class OpenSimRunDecision:
    run: bool
    reason: str
    matched_paths: tuple[str, ...]


@dataclass(frozen=True)
class LaneConfig:
    label: str
    doc_only_globs: tuple[str, ...]
    path_globs: tuple[str, ...]


def _normalize_path(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise TypeError("path must be a non-empty string")
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def load_lane_config(config_path: Path | None = None) -> LaneConfig:
    """Load the pinned OpenSim lane path list."""
    path = config_path or _DEFAULT_CONFIG
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    label = str(raw.get("label", "")).strip()
    if not label:
        raise ValueError(f"{path} must define a non-empty label")
    doc_globs = tuple(str(item) for item in raw.get("doc_only_globs", ()))
    path_globs = tuple(str(item) for item in raw.get("path_globs", ()))
    if not path_globs:
        raise ValueError(f"{path} must define at least one path_glob")
    return LaneConfig(label=label, doc_only_globs=doc_globs, path_globs=path_globs)


def _glob_matches(pattern: str, path: str) -> bool:
    normalized = _normalize_path(path)
    pattern = pattern.replace("\\", "/")
    if pattern.endswith("/**"):
        prefix = pattern[: -len("/**")]
        return normalized == prefix or normalized.startswith(f"{prefix}/")
    return fnmatch.fnmatch(normalized, pattern)


def path_matches_lane(path: str, path_globs: Sequence[str]) -> bool:
    """Return whether ``path`` is covered by any lane glob."""
    return any(_glob_matches(pattern, path) for pattern in path_globs)


def is_doc_only_path(path: str, doc_globs: Sequence[str]) -> bool:
    """Return whether ``path`` counts as documentation-only for lane gating."""
    normalized = _normalize_path(path)
    return any(_glob_matches(pattern, normalized) for pattern in doc_globs)


def classify_path(path: str, config: LaneConfig) -> ChangeCategory:
    """Assign a single changed file to a coarse category (for unit tests)."""
    normalized = _normalize_path(path)
    if is_doc_only_path(normalized, config.doc_only_globs):
        return ChangeCategory.DOC
    if normalized.startswith(".github/workflows/"):
        return ChangeCategory.WORKFLOW
    if normalized.startswith("src/shared/python/physics/"):
        return ChangeCategory.SHARED_PHYSICS_CORE
    if path_matches_lane(normalized, config.path_globs):
        return ChangeCategory.OPENSIM
    return ChangeCategory.OTHER


def _matched_lane_paths(
    changed_files: Sequence[str], path_globs: Sequence[str]
) -> tuple[str, ...]:
    return tuple(path for path in changed_files if path_matches_lane(path, path_globs))


def should_run_opensim_lane(
    *,
    event_name: str,
    changed_files: Sequence[str],
    labels: Sequence[str],
    config: LaneConfig,
) -> OpenSimRunDecision:
    """Decide whether the real-OpenSim workflow should execute its heavy steps."""
    if event_name != "pull_request":
        return OpenSimRunDecision(True, event_name, ())

    label_set = {label.strip() for label in labels if label and label.strip()}
    if config.label in label_set:
        return OpenSimRunDecision(True, "label", ())

    normalized = tuple(_normalize_path(path) for path in changed_files if path.strip())
    if not normalized:
        return OpenSimRunDecision(False, "no_changes", ())

    if all(is_doc_only_path(path, config.doc_only_globs) for path in normalized):
        return OpenSimRunDecision(False, "doc_only", ())

    matched = _matched_lane_paths(normalized, config.path_globs)
    if matched:
        return OpenSimRunDecision(True, "path", matched)

    return OpenSimRunDecision(False, "outside_lane_paths", ())


def _parse_labels(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _write_github_output(outputs: Mapping[str, str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    path = Path(output_path or os.devnull)
    with path.open("a", encoding="utf-8") as handle:
        for key, value in outputs.items():
            handle.write(f"{key}={value}\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "lane",
        choices=("opensim",),
        help="Lane identifier (only opensim is implemented in UD-CI-002).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to the lane JSON config.",
    )
    parser.add_argument(
        "--event",
        default="pull_request",
        help="GitHub event_name (pull_request, push, schedule, workflow_dispatch).",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated pull request labels.",
    )
    parser.add_argument(
        "--changed-file",
        action="append",
        default=[],
        dest="changed_files",
        help="Repository-relative changed path (repeatable).",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Write run/reason/matched_paths outputs for GitHub Actions.",
    )
    args = parser.parse_args(argv)

    config = load_lane_config(args.config)
    decision = should_run_opensim_lane(
        event_name=args.event,
        changed_files=args.changed_files,
        labels=_parse_labels(args.labels),
        config=config,
    )
    if args.github_output:
        _write_github_output(
            {
                "run": "true" if decision.run else "false",
                "reason": decision.reason,
                "matched_paths": ";".join(decision.matched_paths),
            }
        )
    print(
        json.dumps(
            {
                "run": decision.run,
                "reason": decision.reason,
                "matched_paths": list(decision.matched_paths),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
