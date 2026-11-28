#!/usr/bin/env python3
"""Export and validate private golfer delivery candidates.

Use ``--scratch`` for preliminary output in a fresh temp directory. The
``--skip-library-import`` option is scratch-only while #15957 is pending.
``--final`` requires all twelve profiles, current main, accepted appearance
features, a fresh sibling output, and verified temporary-library bindings.

Examples::

    python scripts/prepare_golfer_delivery.py --scratch --players rory_mcilroy
    python scripts/prepare_golfer_delivery.py --scratch --skip-library-import
    python scripts/prepare_golfer_delivery.py --final --output ../golfer-delivery-final
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable
from zipfile import ZipFile

PLAYERS = (
    "mac_ogrady",
    "rory_mcilroy",
    "jon_rahm",
    "dustin_johnson",
    "sergio_garcia",
    "tom_watson",
    "greg_norman_1986",
    "moe_norman",
    "george_knudson",
    "john_schlee",
    "johnny_miller",
    "lee_trevino",
)
BASELINE_ARCHIVE = Path("assets/player_models/player-models.zip")
BASELINE_MEMBER = "reference-skins/tiger_woods/model.json"
BASELINE_ID = f"{BASELINE_ARCHIVE.as_posix()}:{BASELINE_MEMBER}"
HEIGHT_TOL_M = 0.12
ANATOMY_TOL_M = 1e-4


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _object(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def _branch(repo: Path) -> str:
    return subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _output(path: Path | None, repo: Path, final: bool) -> Path:
    tmp = Path(tempfile.gettempdir()).resolve()
    if path is None:
        if final:
            raise ValueError("--final requires a fresh sibling --output")
        return Path(tempfile.mkdtemp(prefix="golfer-delivery-scratch-", dir=tmp))
    target = path.expanduser().absolute()
    parent = target.parent.resolve()
    if final and parent != repo.parent.resolve():
        raise ValueError("Final output must be a sibling of the repository")
    if not final and not parent.is_relative_to(tmp):
        raise ValueError("Scratch output must be under the system temp directory")
    if target.exists():
        raise FileExistsError(f"Refusing to reuse output: {target}")
    target.mkdir(parents=True)
    return target


def _native_checks(
    path: Path, spec: dict[str, Any], profile: dict[str, Any], mode: str
) -> dict[str, Any]:
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(path))
    order = spec.get("coordinate_order")
    if (
        not isinstance(order, list)
        or not order
        or model.nq != model.nv
        or model.nv != len(order)
    ):
        raise ValueError("Native coordinate order/count differs")
    if not all(
        np.isfinite(a).all()
        for a in (model.qpos0, model.jnt_axis, model.body_pos, model.body_quat)
    ):
        raise ValueError("Non-finite native coordinates")
    if not (np.all(model.body_mass[1:] > 0) and np.all(model.body_inertia[1:] > 0)):
        raise ValueError("Native mass/inertia must be positive")
    body_mass = sum(float(s["mass_kg"]) for b in spec["bodies"] for s in b["solids"])
    club_mass = float(spec["club"]["total_mass_kg"])
    if not math.isclose(float(model.body_mass[1:].sum()), body_mass, abs_tol=1e-8):
        raise ValueError("Native mass differs from spec")
    stature = float(spec["subject"]["stature_m"])
    if not math.isfinite(stature) or stature <= 0:
        raise ValueError("Spec stature must be positive")
    calibration = spec.get("subject", {}).get("neutral_visual_calibration")
    anatomy = (
        float(calibration["neutral_visual_height_m"])
        if isinstance(calibration, dict)
        and calibration.get("neutral_visual_height_m") is not None
        else stature
    )
    if mode == "sized_candidate":
        if not math.isclose(stature, float(profile["stature_m"]), abs_tol=1e-9):
            raise ValueError("Sized stature differs from profile")
        if not math.isclose(
            anatomy, float(profile["stature_m"]), abs_tol=ANATOMY_TOL_M
        ):
            raise ValueError("Anatomical height differs from profile")
        if not math.isclose(
            body_mass - club_mass, float(profile["mass_kg"]), abs_tol=1e-8
        ):
            raise ValueError("Sized mass differs from profile")
    eig_min = float("inf")
    for phase in (0.0, 0.03, -0.04):
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = model.qpos0 + phase, phase
        mujoco.mj_forward(model, data)
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        eig = np.linalg.eigvalsh(mass)
        if not np.isfinite(eig).all() or eig.min() <= 0:
            raise ValueError("qM is not positive definite")
        if not np.isfinite(data.qfrc_bias).all() or not np.isfinite(data.qacc).all():
            raise ValueError("Native dynamics are non-finite")
        eig_min = min(eig_min, float(eig.min()))
    return {
        "human_mass_kg": body_mass - club_mass,
        "profile_mass_kg": float(profile["mass_kg"])
        if mode == "sized_candidate"
        else None,
        "height_m": stature,
        "anatomy_height_m": anatomy,
        "anatomy_tolerance_m": ANATOMY_TOL_M if mode == "sized_candidate" else None,
        "qM_min_eig_3_states": eig_min,
    }


def _same_reference_physics(path: Path, baseline: Any) -> None:
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(path))
    names = "body_mass body_inertia body_pos body_quat jnt_type jnt_qposadr jnt_dofadr jnt_range qpos0".split()  # noqa: SIM905
    for name in names:
        np.testing.assert_array_equal(getattr(baseline, name), getattr(model, name))
    for phase in (0.0, 0.03, -0.04):
        states = []
        for current in (baseline, model):
            data = mujoco.MjData(current)
            data.qpos[:], data.qvel[:] = current.qpos0 + phase, phase
            mujoco.mj_forward(current, data)
            states.append(data)
        for name in ("qM", "qfrc_bias", "qacc"):
            np.testing.assert_array_equal(
                getattr(states[0], name), getattr(states[1], name)
            )


def _import_api(required: bool) -> Callable[..., Any] | None:
    try:
        api = importlib.import_module(
            "src.shared.python.workspace.necromatcher_catalog_import"
        ).import_catalog_model
    except (ImportError, AttributeError) as exc:
        if required:
            raise RuntimeError(
                "Final verification requires merged #15957 importer"
            ) from exc
        return None
    if not callable(api):
        raise RuntimeError("Catalog importer is not callable")
    return api


def _final_gate(repo: Path, players: tuple[str, ...]) -> None:
    if _branch(repo) != "main" or players != PLAYERS:
        raise ValueError("Final packaging requires main and all twelve profiles")
    from src.shared.python.model_appearance.head_detail import HeadDetail
    from src.shared.python.model_appearance.schema import CAP_STYLES

    if "baseball_cap_structured" not in CAP_STYLES or not HeadDetail():
        raise ValueError("Merged face detail and structured cap are required")


def run(
    repo: Path,
    output: Path,
    players: tuple[str, ...],
    importer: Callable[..., Any] | None,
) -> dict[str, Any]:
    """Export both variants, validate them, render previews, and optionally import."""
    from src.shared.python.model_appearance.player_models import (
        player_ids,
        player_profile,
    )
    from src.shared.python.model_appearance.schema import physics_spec_sha256

    missing = sorted(set(players) - set(player_ids()))
    if missing:
        raise ValueError(f"Catalog is missing profiles: {missing}")
    archive_path = repo / BASELINE_ARCHIVE
    with ZipFile(archive_path) as archive:
        baseline_bytes = archive.read(BASELINE_MEMBER)
    baseline_spec = json.loads(baseline_bytes)
    if not isinstance(baseline_spec, dict):
        raise ValueError("Reference baseline must be a JSON object")
    with tempfile.TemporaryDirectory(prefix="golfer-delivery-reference-") as tmp:
        baseline_path = Path(tmp) / "reference-model.json"
        baseline_path.write_bytes(baseline_bytes)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            (
                str(repo),
                str(repo / "src"),
                str(repo / "vendor/ud-tools/src"),
                env.get("PYTHONPATH", ""),
            )
        )
        env.setdefault("MUJOCO_GL", "egl")
        for mode, base in (
            ("sized_candidate", None),
            ("appearance_only", baseline_path),
        ):
            command = [
                sys.executable,
                str(repo / "scripts/export_player_models.py"),
                "--player",
                *players,
                "--output",
                str(output / "models" / mode),
            ]
            if base:
                command.extend(("--base-spec", str(base)))
            subprocess.run(command, cwd=repo, env=env, check=True)
        from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
            export_full_body_mjcf,
        )
        import mujoco

        baseline_xml, _ = export_full_body_mjcf(baseline_bytes)
        baseline_native = mujoco.MjModel.from_xml_string(baseline_xml)
        library_tmp = (
            tempfile.TemporaryDirectory(prefix="golfer-delivery-library-")
            if importer
            else None
        )
        library = None
        if importer and library_tmp:
            from src.shared.python.workspace.necromatcher import NecromatcherLibrary

            library = NecromatcherLibrary.create(
                Path(library_tmp.name) / "chosen-library"
            )
        manifest: dict[str, Any] = {
            "status": "scratch_preliminary_not_final",
            "branch": _branch(repo),
            "players": list(players),
            "baseline": {
                "id": BASELINE_ID,
                "archive_sha256": _sha(archive_path.read_bytes()),
                "member_sha256": _sha(baseline_bytes),
                "physics_sha256": physics_spec_sha256(baseline_spec),
            },
            "library_import": "verified"
            if importer
            else "omitted_by_explicit_scratch_mode",
            "models": [],
            "warnings": [
                "No historical motion is fitted.",
                "Reference skins retain Tiger fitted physics, not golfer dimensions.",
                "Media is not copied; verify reuse rights before redistribution.",
            ],
        }
        try:
            for player in players:
                profile = player_profile(player)
                variants = []
                for mode in ("sized_candidate", "appearance_only"):
                    folder = output / "models" / mode / player
                    spec, appearance = (
                        _object(folder / "model.json"),
                        _object(folder / "model.appearance.json"),
                    )
                    visual, receipt = (
                        _object(folder / "model.visual.json"),
                        _object(folder / "receipt.json"),
                    )
                    if visual.pop("appearance", None) != appearance or visual != spec:
                        raise ValueError(f"Visual sidecar mismatch: {player}/{mode}")
                    if (
                        receipt.get("mode") != mode
                        or receipt.get("visual_definition") != "model.visual.json"
                    ):
                        raise ValueError(f"Bad source receipt: {player}/{mode}")
                    xml_path = folder / "model.xml"
                    checks = _native_checks(xml_path, spec, profile, mode)
                    if mode == "appearance_only":
                        if spec != baseline_spec:
                            raise ValueError(
                                "Skin changed the supplied physics baseline"
                            )
                        _same_reference_physics(xml_path, baseline_native)
                    preview_dir = folder / "previews" / "neutral"
                    preview = _object(preview_dir / "preview_receipt.json")
                    height = float(preview["rendered_geometry_height_m"])
                    delta = height - float(checks["anatomy_height_m"])
                    if (
                        not math.isfinite(height)
                        or height <= 0
                        or abs(delta) > HEIGHT_TOL_M
                    ):
                        raise ValueError(
                            f"Rendered height differs from anatomy: {player}/{mode}"
                        )
                    checks.update(
                        rendered_height_m=height,
                        rendered_delta_m=delta,
                        rendered_tolerance_m=HEIGHT_TOL_M,
                    )
                    imported = None
                    if importer and library:
                        model_record, bundle_record = importer(
                            library,
                            player,
                            folder,
                            reference_baseline=baseline_bytes
                            if mode == "appearance_only"
                            else None,
                            reference_baseline_id=BASELINE_ID
                            if mode == "appearance_only"
                            else None,
                        )
                        model = library.load_asset(model_record.dataset_id)
                        definition = library.load_model_definition(
                            model_record.dataset_id
                        )
                        if definition != (folder / "model.visual.json").read_bytes():
                            raise ValueError("Imported visual definition changed")
                        if (
                            library.load_model_source_manifest(model_record.dataset_id)
                            != (folder / "receipt.json").read_bytes()
                        ):
                            raise ValueError("Imported source receipt changed")
                        if (
                            mode == "appearance_only"
                            and library.load_reference_baseline(model_record.dataset_id)
                            != baseline_bytes
                        ):
                            raise ValueError("Imported baseline bytes changed")
                        from src.shared.python.workspace.necromatcher_native import (
                            load_native_model_binding,
                        )

                        binding = load_native_model_binding(
                            library,
                            model_record.dataset_id,
                            definition,
                            tuple(model.metadata["coordinate_units"]),
                        )
                        if binding.model_hash != model.metadata["hash"]:
                            raise ValueError("Native binding hash mismatch")
                        imported = {
                            "dataset_id": model_record.dataset_id,
                            "bundle_id": bundle_record.dataset_id,
                            "model_hash": model.metadata["hash"],
                            "native_binding_verified": True,
                            "coordinate_units": list(binding.coordinate_units),
                            "coordinate_order": list(binding.plant.coordinate_order),
                        }
                    variants.append(
                        {
                            "mode": mode,
                            "height_basis": "catalog anatomy"
                            if mode == "sized_candidate"
                            else BASELINE_ID,
                            "physics_sha256": receipt["physics_spec_sha256"],
                            "checks": checks,
                            "body_build": appearance.get("body_build", 0.0),
                            "frame_prior": receipt.get("frame_prior"),
                            "import": imported,
                        }
                    )
                manifest["models"].append(
                    {
                        "player_id": player,
                        "display_name": profile["display_name"],
                        "stature_m": profile["stature_m"],
                        "mass_kg": profile["mass_kg"],
                        "sources": profile.get("sources", []),
                        "uncertainties": profile.get("limitations", []),
                        "variants": variants,
                    }
                )
        finally:
            if library_tmp:
                library_tmp.cleanup()
    files = sorted(p for p in output.rglob("*") if p.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha(p.read_bytes())}  {p.relative_to(output)}\n" for p in files),
        encoding="utf-8",
    )
    manifest_path = output / "delivery-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "delivery-manifest.sha256").write_text(
        f"{_sha(manifest_path.read_bytes())}  delivery-manifest.json\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    """Run only after the caller explicitly selects scratch or final mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--scratch", action="store_true")
    modes.add_argument("--final", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--players", nargs="+", choices=PLAYERS, default=PLAYERS)
    parser.add_argument("--skip-library-import", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    sys.path[:0] = [str(repo), str(repo / "src"), str(repo / "vendor/ud-tools/src")]
    if args.skip_library_import and not args.scratch:
        parser.error("--skip-library-import is scratch-only")
    if args.final and tuple(args.players) != PLAYERS:
        parser.error("--final requires all twelve profiles")
    selected = PLAYERS if args.final else tuple(dict.fromkeys(args.players))
    if args.final:
        _final_gate(repo, selected)
    importer = _import_api(required=args.final or not args.skip_library_import)
    if args.skip_library_import:
        importer = None
    os.environ.setdefault("MUJOCO_GL", "egl")
    output = _output(args.output, repo, args.final)
    manifest = run(repo, output, selected, importer)
    if args.final:
        manifest["status"] = "final_candidate"
        path = output / "delivery-manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (output / "delivery-manifest.sha256").write_text(
            f"{_sha(path.read_bytes())}  delivery-manifest.json\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
