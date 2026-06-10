#!/usr/bin/env python3
"""Validate Docker build/deploy contracts that must not silently drift."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT_DOCKERFILES = (
    Path("Dockerfile"),
    Path("Dockerfile.modular"),
    Path("Dockerfile.heavy_test"),
)
PIP_PIN_RE = re.compile(r"\bpip==([0-9][0-9.]+)")


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def docker_contract_failures(root: Path = Path(".")) -> list[str]:
    failures: list[str] = []
    pin_by_file: dict[str, set[str]] = {}
    for rel_path in ROOT_DOCKERFILES:
        path = root / rel_path
        if not path.exists():
            continue
        pins = set(PIP_PIN_RE.findall(_read(path)))
        if pins:
            pin_by_file[str(rel_path)] = pins

    all_pins = {pin for pins in pin_by_file.values() for pin in pins}
    if len(all_pins) > 1:
        failures.append(f"Dockerfile pip pins diverge: {pin_by_file}")

    dockerfile = _read(root / "Dockerfile")
    if "ARG SKIP_AUDIT=false" not in dockerfile:
        failures.append("Dockerfile builder audit must default to enabled")
    for required in (
        "pip-audit==2.10.0",
        "check_pip_audit_waivers.py",
        "pip_audit_waivers.json",
    ):
        if required not in dockerfile:
            failures.append(f"Dockerfile builder audit is missing {required}")
    if "MUJOCO_GL=osmesa" not in dockerfile and 'MUJOCO_GL="osmesa"' not in dockerfile:
        failures.append("Dockerfile runtime image must set MUJOCO_GL=osmesa")
    if "urllib.request.urlopen('http://localhost:8001/health'" not in dockerfile:
        failures.append("Dockerfile healthcheck must use Python urllib, not curl")

    modular = _read(root / "Dockerfile.modular")
    first_dry_run = modular.find("install_features.py --profile")
    feature_registry_copy = modular.find("COPY src/shared/python/feature_registry/")
    engine_core_copy = modular.find("COPY src/shared/python/engine_core/")
    if feature_registry_copy == -1 or (
        first_dry_run != -1 and feature_registry_copy > first_dry_run
    ):
        failures.append(
            "Dockerfile.modular must copy feature_registry before profile dry-run"
        )
    if engine_core_copy == -1 or (
        first_dry_run != -1 and engine_core_copy > first_dry_run
    ):
        failures.append(
            "Dockerfile.modular must copy engine_core before profile dry-run"
        )

    heavy = _read(root / "Dockerfile.heavy_test")
    for package in ("PyQt6", "drake", "opensim", "myosuite"):
        if re.search(rf"pip install {package}\s+\|\|", heavy):
            failures.append(f"Dockerfile.heavy_test masks {package} install failure")
    if "pip install -e .[dev,test]" in heavy and "||" in heavy:
        failures.append("Dockerfile.heavy_test must not mask project install failure")

    compose = _read(root / "docker-compose.yml")
    if "condition: service_healthy" not in compose:
        failures.append("docker-compose frontend must wait for backend health")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    failures = docker_contract_failures(args.root)
    if failures:
        print("\n".join(failures))
        return 1
    print("Docker build contracts are coherent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
