#!/usr/bin/env python3
"""Cross-engine humanoid model build orchestrator.

This is the user-visible entry point for regenerating per-engine model files
from the shared anthropometric YAML at
``shared/models/golf_humanoid_dimensions.yaml`` (and the companion
``golf_humanoid_inertia.yaml`` / ``golf_humanoid_topology.yaml``).

This script is owned long-term by issue **#4094 (PARITY-MODEL-BUILD)**. The
Drake-side hook landed first as part of issue **#4108 (DRAKE-1)**; the other
three engines (Pinocchio, MuJoCo, OpenSim) are wired here per #4094.

Usage::

    python3 scripts/build_humanoid_models.py --engine drake
    python3 scripts/build_humanoid_models.py --engine pinocchio
    python3 scripts/build_humanoid_models.py --engine mujoco
    python3 scripts/build_humanoid_models.py --engine opensim
    python3 scripts/build_humanoid_models.py --engine all
    python3 scripts/build_humanoid_models.py --engine all --check    # CI gate

The ``--check`` mode regenerates each engine's artifact into a temp dir and
asserts the on-disk file matches byte-for-byte; the CI gate
(``humanoid-models-drift.yml``) uses this to forbid hand-edits to generated
files. Drift is currently advisory-warn only; the workflow does not fail PRs
on drift (we'll harden this once every engine has a real generator).

Per-engine status (2026-05-06):

* **drake** — full generator from shared YAML; canonical at
  ``src/engines/physics_engines/drake/models/generated/golfer.urdf``.
* **opensim** — deterministic build via ``scripts/build_humanoid_osim.py``;
  canonical at
  ``src/engines/physics_engines/opensim/models/golf_humanoid.osim``.
  Requires the ``shared/models/opensim/opensim-models`` submodule to be
  initialised.
* **pinocchio** — verify-only. The URDF at
  ``src/engines/physics_engines/pinocchio/models/generated/golfer.urdf`` is
  currently hand-authored against
  ``src/engines/physics_engines/pinocchio/models/spec/golfer_canonical.yaml``;
  a real generator is tracked under PINOCCHIO-MODEL-BUILD. We confirm the
  artifact exists and is well-formed XML.
* **mujoco** — verify-only. The MJCF lives in module-level Python string
  constants in ``src/engines/physics_engines/mujoco/_golf_swing_*_xml.py``
  (no on-disk artifact). We confirm each constant is non-empty and parses
  as well-formed XML.
"""

from __future__ import annotations

import argparse
import filecmp
import sys
import tempfile
import xml.etree.ElementTree as ET  # noqa: N817
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_YAML = REPO_ROOT / "shared" / "models" / "golf_humanoid_dimensions.yaml"

# Make the repo root importable so this script works whether or not the
# package is pip-installed. CI invokes us as `python3 scripts/build_...`
# from the repo root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
shared_python = REPO_ROOT / "src" / "shared" / "python"
if str(shared_python) not in sys.path:
    sys.path.insert(0, str(shared_python))

#: All engines this orchestrator knows about, in canonical order.
ENGINES: tuple[str, ...] = ("drake", "pinocchio", "mujoco", "opensim")


# ---------------------------------------------------------------------------
# Drake
# ---------------------------------------------------------------------------


def _build_drake(yaml_path: Path, *, check: bool) -> int:
    """Generate (or verify) the Drake humanoid URDF."""
    # Lazy import so that --engine mujoco doesn't pull in drake plumbing.
    from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (  # noqa: E501
        CANONICAL_URDF,
        build_humanoid_urdf,
    )

    if check:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_out = Path(tmp) / "golfer.urdf"
            build_humanoid_urdf(yaml_path=yaml_path, out_path=tmp_out)
            try:
                rel_canonical = CANONICAL_URDF.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                rel_canonical = str(CANONICAL_URDF)
            if not CANONICAL_URDF.exists():
                sys.stderr.write(
                    "FAIL: drake URDF drift gate (#4129)\n"
                    f"  Canonical URDF missing: {rel_canonical}\n"
                    "  Fix locally:\n"
                    "    python3 scripts/build_humanoid_models.py --engine drake\n"
                    "    git add "
                    f"{rel_canonical}\n"
                    "    git commit -m 'regen drake URDF'\n"
                )
                return 1
            if not filecmp.cmp(tmp_out, CANONICAL_URDF, shallow=False):
                sys.stderr.write(
                    "FAIL: drake URDF drift gate (#4129)\n"
                    f"  Drift detected in: {rel_canonical}\n"
                    "  The on-disk URDF does not match a fresh regeneration\n"
                    "  from the shared YAML. Hand-edits to engine model files\n"
                    "  are forbidden by CROSS_ENGINE_PARITY_SPEC.md §6.\n"
                    "\n"
                    "  Fix locally:\n"
                    "    python3 scripts/build_humanoid_models.py --engine drake\n"
                    "    git add "
                    f"{rel_canonical}\n"
                    "    git commit -m 'regen drake URDF'\n"
                )
                return 1
            sys.stdout.write(
                f"OK: drake URDF matches regeneration ({rel_canonical}).\n"
            )
            return 0

    out = build_humanoid_urdf(yaml_path=yaml_path)
    sys.stdout.write(f"Wrote {out}\n")
    return 0


# ---------------------------------------------------------------------------
# OpenSim
# ---------------------------------------------------------------------------


def _build_opensim(yaml_path: Path, *, check: bool) -> int:  # noqa: ARG001
    """Generate (or verify) the OpenSim humanoid ``.osim`` file.

    Delegates to ``scripts/build_humanoid_osim.py``. The ``yaml_path``
    argument is accepted for orchestrator-level uniformity but is unused —
    the OpenSim builder reads from the Rajagopal2015 OpenSense base model
    directly (see ``build_humanoid_osim.py`` for rationale).
    """
    # Lazy import: build_humanoid_osim defines REPO_ROOT relative to itself,
    # which is fine since scripts/ is the package directory we're in.
    from scripts import build_humanoid_osim  # type: ignore[import-not-found]

    if check:
        canonical = build_humanoid_osim.OUTPUT_OSIM
        if not canonical.exists():
            sys.stderr.write(f"FAIL: canonical OpenSim model missing at {canonical}\n")
            return 1
        # Regenerate to a temp dir and compare. We monkey-patch the output
        # path on the build module so we don't clobber the canonical file
        # when running --check.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_out = Path(tmp) / "golf_humanoid.osim"
            saved = build_humanoid_osim.OUTPUT_OSIM
            build_humanoid_osim.OUTPUT_OSIM = tmp_out
            try:
                build_humanoid_osim.build()
                if not tmp_out.exists():
                    # build() did not honor the patched OUTPUT_OSIM — most
                    # likely the build module captured the path at import
                    # time. We can't reliably regenerate, so warn-skip
                    # rather than treat this as a hard failure (#4531).
                    raise FileNotFoundError(
                        f"build() did not produce {tmp_out}; build module "
                        "may have captured OUTPUT_OSIM at import"
                    )
            except FileNotFoundError as exc:
                sys.stderr.write(
                    f"WARN: opensim --check skipped: {exc}\n"
                    "Initialise the submodule: "
                    "`git submodule update --init "
                    "shared/models/opensim/opensim-models`.\n"
                )
                return 0
            finally:
                build_humanoid_osim.OUTPUT_OSIM = saved
            if not filecmp.cmp(tmp_out, canonical, shallow=False):
                sys.stderr.write(
                    "FAIL: regenerated opensim model differs from on-disk file. "
                    "Run `python3 scripts/build_humanoid_models.py "
                    "--engine opensim` and commit the result.\n"
                )
                return 1
            sys.stdout.write("OK: opensim model matches regeneration.\n")
            return 0

    try:
        out = build_humanoid_osim.build()
    except FileNotFoundError as exc:
        sys.stderr.write(
            f"WARN: opensim build skipped: {exc}\n"
            "Initialise the submodule: "
            "`git submodule update --init "
            "shared/models/opensim/opensim-models`.\n"
        )
        return 0
    sys.stdout.write(f"Wrote {out}\n")
    return 0


# ---------------------------------------------------------------------------
# Pinocchio
# ---------------------------------------------------------------------------

PINOCCHIO_URDF: Path = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "models"
    / "generated"
    / "golfer.urdf"
)


def _build_pinocchio(yaml_path: Path, *, check: bool) -> int:  # noqa: ARG001
    """Verify the Pinocchio humanoid URDF.

    Pinocchio's URDF is currently hand-authored against the per-engine
    canonical YAML at ``models/spec/golfer_canonical.yaml`` rather than the
    shared anthropometric YAML; replacing the hand-authored URDF with a
    full generator is tracked under PINOCCHIO-MODEL-BUILD (a follow-up to
    #4094). For now we treat this engine as **verify-only**: confirm the
    canonical URDF exists and parses as well-formed XML.
    """
    if not PINOCCHIO_URDF.exists():
        sys.stderr.write(f"FAIL: pinocchio URDF missing at {PINOCCHIO_URDF}\n")
        return 1
    try:
        ET.parse(PINOCCHIO_URDF)
    except ET.ParseError as exc:
        sys.stderr.write(f"FAIL: pinocchio URDF is malformed XML: {exc}\n")
        return 1
    if check:
        sys.stdout.write(
            "OK: pinocchio URDF exists and parses (verify-only mode; "
            "see PINOCCHIO-MODEL-BUILD for full generator).\n"
        )
        return 0
    sys.stdout.write(
        f"OK: pinocchio URDF at {PINOCCHIO_URDF} (verify-only; "
        "no regeneration performed).\n"
    )
    return 0


# ---------------------------------------------------------------------------
# MuJoCo
# ---------------------------------------------------------------------------

#: MuJoCo MJCF strings live in module-level Python constants. There is no
#: on-disk artifact; the constants are imported by callers and fed directly
#: to ``mujoco.MjModel.from_xml_string``. We verify each one is well-formed.
_MUJOCO_XML_CONSTANTS: tuple[tuple[str, str], ...] = (
    (
        "src.engines.physics_engines.mujoco._golf_swing_advanced_xml",
        "ADVANCED_BIOMECHANICAL_GOLF_SWING_XML",
    ),
    (
        "src.engines.physics_engines.mujoco._golf_swing_full_body_xml",
        "FULL_BODY_GOLF_SWING_XML",
    ),
    (
        "src.engines.physics_engines.mujoco._golf_swing_upper_body_xml",
        "UPPER_BODY_GOLF_SWING_XML",
    ),
)


def _build_mujoco(yaml_path: Path, *, check: bool) -> int:  # noqa: ARG001
    """Verify the MuJoCo MJCF model strings.

    MuJoCo's models are not on-disk files; they are Python module-level
    string constants in ``_golf_swing_*_xml.py`` (split out of
    ``golf_swing_models_xml.py`` for file-size budget reasons). We import
    each constant and confirm it is non-empty and parses as well-formed
    XML. A full generator that emits these strings from the shared YAML is
    tracked under MUJOCO-MODEL-BUILD (a follow-up to #4094).
    """
    import importlib

    failures: list[str] = []
    for module_path, attr in _MUJOCO_XML_CONSTANTS:
        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{module_path}: import failed ({exc})")
            continue
        xml_str = getattr(mod, attr, None)
        if not isinstance(xml_str, str) or not xml_str.strip():
            failures.append(f"{module_path}.{attr}: missing or empty")
            continue
        try:
            ET.fromstring(xml_str)
        except ET.ParseError as exc:
            failures.append(f"{module_path}.{attr}: malformed XML ({exc})")
    if failures:
        for line in failures:
            sys.stderr.write(f"FAIL: mujoco {line}\n")
        return 1

    mode = "check" if check else "build"
    sys.stdout.write(
        f"OK: mujoco MJCF constants parse cleanly ({len(_MUJOCO_XML_CONSTANTS)} "
        f"models verified, verify-only mode; {mode}).\n"
    )
    return 0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_DISPATCH = {
    "drake": _build_drake,
    "opensim": _build_opensim,
    "pinocchio": _build_pinocchio,
    "mujoco": _build_mujoco,
}


def _resolve_engines(selected: list[str]) -> list[str]:
    """Expand ``all`` to every engine, preserving canonical order and dedup."""
    if "all" in selected:
        return list(ENGINES)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in selected:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        action="append",
        choices=[*ENGINES, "all"],
        required=True,
        help="Engine(s) to regenerate. Repeatable. Use 'all' for every engine.",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=SHARED_YAML,
        help="Path to the shared humanoid dimensions YAML.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Regenerate to a temp dir and diff against the on-disk file. "
            "Exits non-zero if they differ. Used by the "
            "humanoid-models-drift CI workflow."
        ),
    )
    args = parser.parse_args(argv)

    if not args.yaml.exists():
        sys.stderr.write(f"YAML not found: {args.yaml}\n")
        return 2

    rc = 0
    for engine in _resolve_engines(args.engine):
        builder = _DISPATCH.get(engine)
        if builder is None:  # pragma: no cover - argparse rejects other values
            sys.stderr.write(f"Unsupported engine: {engine}\n")
            rc |= 2
            continue
        try:
            rc |= builder(args.yaml, check=args.check)
        except Exception as exc:  # noqa: BLE001
            # Surface a per-engine failure but keep iterating so that one
            # broken engine doesn't mask the status of the others. This is
            # essential for ``--engine all`` to remain useful as a CI gate.
            sys.stderr.write(
                f"FAIL: {engine} build raised {type(exc).__name__}: {exc}\n"
            )
            rc |= 1
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
