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


# Images the explorer launcher must verify (and rebuild) before every run.
EXPLORER_REQUIRED_IMAGES = (
    "upstream-drift:explorer-headless",
    "upstream-drift:explorer-gui",
    "ud-sim:latest",
    "upstream-drift:full",
)
EXPLORER_STAGES = ("explorer-base", "explorer-headless", "explorer-gui")
# service -> env/volume fragments its block must contain (issue #12878 D).
EXPLORER_SERVICE_REQUIRES: dict[str, tuple[str, ...]] = {
    "explorer-headless": ("QT_QPA_PLATFORM=offscreen", "MUJOCO_GL="),
    "explorer-gui": (
        "QT_QPA_PLATFORM=xcb",
        "MUJOCO_GL=",
        "DISPLAY=:99",
        "NOVNC_PORT=6080",
        "VNC_PORT=5900",
    ),
    "explorer-gui-wslg": ("DISPLAY=", "/tmp/.X11-unix:/tmp/.X11-unix"),
}
# Display / GL prerequisites the entrypoint must refuse to start without (DbC).
EXPLORER_ENTRYPOINT_GUARDS = ("QT_QPA_PLATFORM", "MUJOCO_GL", "DISPLAY")


def _service_block(compose: str, name: str) -> str | None:
    """Return the text of one top-level compose service, or None if absent."""
    match = re.search(
        rf"^  {re.escape(name)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|^[A-Za-z]|\Z)",
        compose,
        re.MULTILINE | re.DOTALL,
    )
    return match.group(1) if match else None


def explorer_contract_failures(root: Path) -> list[str]:
    """Check the model-explorer Docker surface: stages, services, entrypoint."""
    failures: list[str] = []
    modular = _read(root / "Dockerfile.modular")
    for stage in EXPLORER_STAGES:
        if not re.search(rf"^FROM\s+\S+\s+AS\s+{stage}\s*$", modular, re.MULTILINE):
            failures.append(f"Dockerfile.modular is missing the {stage} stage")
    for stage in ("explorer-headless", "explorer-gui"):
        if not re.search(
            rf"^FROM\s+explorer-base\s+AS\s+{stage}\s*$", modular, re.MULTILINE
        ):
            failures.append(
                f"{stage} must derive from the shared explorer-base stage (DRY)"
            )

    compose = _read(root / "docker-compose.yml")
    for service, needles in EXPLORER_SERVICE_REQUIRES.items():
        block = _service_block(compose, service)
        if block is None:
            failures.append(f"docker-compose.yml is missing the {service} service")
            continue
        failures.extend(
            f"docker-compose {service} must set {needle.split('=')[0]} ({needle})"
            for needle in needles
            if needle not in block
        )
        for port in re.findall(r'-\s*"((?:[\d.]+:)?\d+:\d+)"', block):
            if not port.startswith("127.0.0.1:"):
                failures.append(
                    f"docker-compose {service} port {port} must bind loopback only"
                )

    for build_arg in ("TOOLS_GITLINK_SHA", "TOOLS_SOURCE_SHA256"):
        if build_arg not in compose:
            failures.append(
                f"docker-compose explorer build must pass {build_arg} "
                "(pinned Tools provenance, see scripts/docker/explorer.sh)"
            )
    if not re.search(r"labels:\s*\n\s+keep:", compose):
        failures.append(
            "docker-compose explorer images must carry label keep (host prune jobs "
            "spare it)"
        )
    launcher = root / "scripts" / "docker" / "explorer.sh"
    if not launcher.exists():
        failures.append("scripts/docker/explorer.sh launcher is missing")
    else:
        script = _read(launcher)
        for image in EXPLORER_REQUIRED_IMAGES:
            if image not in script or "docker image inspect" not in script:
                failures.append(
                    f"explorer.sh must health-check image {image} with "
                    "docker image inspect and rebuild it if missing"
                )

    entrypoint = root / "docker" / "explorer" / "entrypoint.sh"
    if not entrypoint.exists():
        failures.append("docker/explorer/entrypoint.sh is missing")
    else:
        script = _read(entrypoint)
        for variable in EXPLORER_ENTRYPOINT_GUARDS:
            if f'[ -n "${{{variable}:-}}" ]' not in script:
                failures.append(
                    f"docker/explorer/entrypoint.sh must fail fast without {variable}"
                )
    return failures


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
    feature_install = modular.rfind('install_features.py --profile "$PROFILE"')
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
    for launcher in ("launch_golf_suite.py", "launch_upstream_drift.py"):
        launcher_copy = modular.find(f"COPY {launcher} ./")
        if launcher_copy == -1 or (
            feature_install != -1 and launcher_copy > feature_install
        ):
            failures.append(
                f"Dockerfile.modular must copy {launcher} before feature install"
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
    failures.extend(explorer_contract_failures(root))
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
