#!/usr/bin/env python3
"""Translate a Docker profile (or feature list) into pip invocations.

This is the bridge between :file:`docker/profiles.yaml`, the canonical
feature registry, and a running ``RUN`` step inside the
:file:`Dockerfile`. The Dockerfile passes either:

  * ``--build-arg PROFILE=research``  (a profile name), or
  * ``--build-arg FEATURES=mujoco,drake`` (a comma-separated list).

This script resolves either input to the union of pip commands needed
and runs them in order. Running the helper in the build means the
Dockerfile itself stays short and feature additions don't require
Dockerfile edits — only edits to ``profiles.yaml`` and the registry.

Usage
-----
::

    python install_features.py --profile research
    python install_features.py --features mujoco,drake
    python install_features.py --features "mujoco, pinocchio" --dry-run

When ``--dry-run`` is set, commands are printed but not executed —
useful for local inspection and for CI to log the per-profile install
plan before the build runs.

Exit codes
----------
* 0 — every install succeeded (or --dry-run)
* 2 — invalid input (unknown profile, unknown feature, cycle, etc.)
* >0 — first failed pip invocation's exit code

This script intentionally has zero non-stdlib dependencies — it
must run before ``pip install`` itself, on a minimal Python image.
``profiles.yaml`` is parsed with a hand-rolled mini-YAML reader so we
don't pull PyYAML into the builder.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal YAML loader.
#
# We only need to read a single fixed shape: top-level keys ``version``
# and ``profiles``, then per-profile ``description: str``,
# ``features: [str, ...]``, ``extends: str`` (optional), and
# ``max_size_mb: int``. This is enough to avoid the PyYAML dependency.
# If profiles.yaml ever grows more complex, switch to PyYAML at the
# cost of adding it to the builder stage.
# ---------------------------------------------------------------------------


def _parse_yaml(text: str) -> dict:
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    root: dict = {}
    profiles: dict = {}
    current: dict | None = None
    in_profiles = False

    for raw in lines:
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)

        if indent == 0:
            if stripped.startswith("version:"):
                root["version"] = int(stripped.split(":", 1)[1].strip())
            elif stripped.startswith("profiles:"):
                in_profiles = True
            else:
                # Unrecognized top-level key — ignore so profiles.yaml can
                # grow forward-compatibly.
                in_profiles = False
            continue

        if not in_profiles:
            continue

        if indent == 2 and stripped.endswith(":"):
            name = stripped[:-1].strip()
            current = {"features": []}
            profiles[name] = current
            continue

        if current is None:
            continue

        if indent >= 4 and ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()
            if key == "features":
                if value.startswith("[") and value.endswith("]"):
                    body = value[1:-1].strip()
                    items = [item.strip() for item in body.split(",") if item.strip()]
                    current["features"] = items
                else:
                    current["features"] = []
            elif key == "extends":
                current["extends"] = value
            elif key == "max_size_mb":
                current["max_size_mb"] = int(value)
            elif key == "description":
                current["description"] = value

    root["profiles"] = profiles
    return root


def _load_profiles(profiles_path: Path) -> dict:
    if not profiles_path.exists():
        raise FileNotFoundError(profiles_path)
    return _parse_yaml(profiles_path.read_text(encoding="utf-8"))


def _resolve_profile_features(profiles: dict, name: str) -> list[str]:
    """Return the deduplicated feature list for a profile, honoring ``extends``."""
    if name not in profiles:
        known = ", ".join(sorted(profiles))
        raise SystemExit(f"unknown profile {name!r}; known: {known}")

    seen: set[str] = set()
    chain: list[str] = []

    cursor: str | None = name
    while cursor is not None:
        if cursor in seen:
            raise SystemExit(f"profile cycle detected through {cursor!r}")
        seen.add(cursor)
        chain.append(cursor)
        cursor = profiles[cursor].get("extends")

    features: list[str] = []
    for profile_name in reversed(chain):
        for feature in profiles[profile_name].get("features", []):
            if feature not in features:
                features.append(feature)
    return features


# ---------------------------------------------------------------------------
# Feature → install command via the registry.
#
# We import the registry directly so the *single* source of truth lives in
# Python. To avoid running the registry's import-time DbC validator
# (which would also import probes that need numpy etc.) we read the
# features module in isolation by adding the repo to sys.path.
# ---------------------------------------------------------------------------


def _feature_install_argv(feature_name: str, repo_root: Path) -> list[list[str]]:
    """Return one or more argv lists describing how to install a feature."""
    sys.path.insert(0, str(repo_root))
    try:
        from src.shared.python.feature_registry.features import get_feature
    finally:
        sys.path.pop(0)

    feature = get_feature(feature_name)

    if feature.install_channel == "external":
        # Cannot be installed during a Docker build. Surface this as an
        # explicit no-op so the build log records the skip.
        return [["echo", f"[skip] feature {feature.name} is external-build only"]]

    if feature.install_channel == "conda":
        # Builder image is Debian + pip; conda is not present. Profile
        # consumers that include opensim/chrono need a conda-base image
        # (handled separately in a future profile). For now, skip.
        return [
            [
                "echo",
                f"[skip] feature {feature.name} requires conda — not available "
                "in this builder. Use the conda-base image variant.",
            ]
        ]

    if feature.install_channel == "pip-extra":
        if feature.pip_extra is None:
            # api / pendulum: install the package itself.
            return [["pip", "install", "--no-cache-dir", "."]]
        return [
            [
                "pip",
                "install",
                "--no-cache-dir",
                f".[{feature.pip_extra}]",
            ]
        ]

    if feature.install_channel == "pip":
        # Verbatim from the documented command (e.g. torch CUDA index URL).
        tokens = feature.install_command.split()
        if tokens[:2] == ["pip", "install"]:
            return [["pip", "install", "--no-cache-dir", *tokens[2:]]]
        return [tokens]

    raise SystemExit(
        f"unsupported install_channel {feature.install_channel!r} "
        f"for feature {feature_name!r}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install Docker-build features via pip."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--profile",
        help="Name of a profile defined in docker/profiles.yaml.",
    )
    group.add_argument(
        "--features",
        help="Comma-separated feature list, e.g. 'mujoco,drake'.",
    )
    parser.add_argument(
        "--profiles-file",
        default=None,
        help="Path to profiles.yaml. Defaults to <repo>/docker/profiles.yaml.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to the parent of this script's dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args(argv)


def _resolve_features(args: argparse.Namespace, repo_root: Path) -> list[str]:
    if args.features is not None:
        return [item.strip() for item in args.features.split(",") if item.strip()]

    profiles_file = (
        Path(args.profiles_file)
        if args.profiles_file is not None
        else repo_root / "docker" / "profiles.yaml"
    )
    data = _load_profiles(profiles_file)
    return _resolve_profile_features(data["profiles"], args.profile)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    repo_root = Path(
        args.repo_root
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )

    features = _resolve_features(args, repo_root)
    if not features:
        print("[install_features] no features resolved — nothing to do")
        return 0

    print(f"[install_features] resolved features: {', '.join(features)}")

    for feature in features:
        commands = _feature_install_argv(feature, repo_root)
        for argv_cmd in commands:
            print(f"[install_features] $ {' '.join(argv_cmd)}")
            if args.dry_run:
                continue
            env = os.environ.copy()
            result = subprocess.run(argv_cmd, cwd=str(repo_root), env=env)
            if result.returncode != 0:
                print(
                    f"[install_features] command failed with exit "
                    f"{result.returncode}: {' '.join(argv_cmd)}",
                    file=sys.stderr,
                )
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
