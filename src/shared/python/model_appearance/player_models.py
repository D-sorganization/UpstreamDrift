"""Source-attributed player candidates on the current full-body golfer pipeline.

Dimensions are published priors; segment proportions and visual likeness are
interpretations, not measured player anatomy or fitted historical motion.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeGuard

from src.shared.python.model_appearance.schema import (
    AppearanceDocument,
    document_from_dict,
    document_to_dict,
    physics_spec_sha256,
    save_appearance,
)

CATALOG = Path(__file__).resolve().parents[3] / "config/player_appearance_presets.json"


def player_ids() -> tuple[str, ...]:
    """Return the stable sorted identifiers in the public appearance catalog."""
    return tuple(sorted(json.loads(CATALOG.read_text(encoding="utf-8"))["players"]))


def player_profile(player: str) -> dict[str, Any]:
    """Return a fresh profile; reject unknown IDs instead of guessing a subject."""
    profiles = json.loads(CATALOG.read_text(encoding="utf-8"))["players"]
    if player not in profiles:
        raise ValueError(f"Unknown player {player!r}; choose {sorted(profiles)}")
    return profiles[player]


def player_appearance(player: str) -> AppearanceDocument:
    """Appearance usable directly on any compatible existing full-body spec."""
    return document_from_dict(player_profile(player)["appearance"])


def _finite_number(value: Any) -> TypeGuard[int | float]:
    """Reject booleans, nonnumeric inputs and numbers too large to convert."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def frame_prior(profile: dict[str, Any]) -> dict[str, Any]:
    """Resolve an explicit frame assumption and its non-physical provenance."""
    setting = profile.get("frame_prior")
    if setting is None:
        return {"method": "unknown", "value": 0.0}
    if not isinstance(setting, dict):
        raise ValueError("frame_prior must be an object or null")
    method = setting.get("method")
    if method == "override":
        value, reason = setting.get("value"), setting.get("rationale")
        if (
            not _finite_number(value)
            or abs(value) > 1
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError("frame override needs a bounded value and rationale")
        return {"method": method, "value": float(value), "rationale": reason}
    if method != "height_mass":
        raise ValueError("unknown frame prior method")
    height, mass = profile.get("stature_m"), profile.get("mass_kg")
    if any(not _finite_number(v) or v <= 0 for v in (height, mass)):
        raise ValueError("frame proxy needs finite positive stature and mass")
    index, sources = setting.get("source_index"), profile.get("sources")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not isinstance(sources, list)
        or not 0 <= index < len(sources)
        or not isinstance(sources[index], dict)
        or not isinstance(sources[index].get("url"), str)
        or not sources[index]["url"].strip()
    ):
        raise ValueError("frame proxy needs an explicit source index")
    height = float(profile["stature_m"])
    mass = float(profile["mass_kg"])
    try:
        raw = math.sqrt((mass / 78.0) * (1.71 / height) ** 3)
    except OverflowError as exc:
        raise ValueError("frame proxy inputs are out of range") from exc
    if not math.isfinite(raw):
        raise ValueError("frame proxy inputs are out of range")
    value = max(-1.0, min(1.0, (raw - 1.0) / 0.18))
    return {
        "method": method,
        "value": value,
        "raw_transverse_scale": raw,
        "saturated": raw < 0.82 or raw > 1.18,
        "source": sources[index],
        "reference_stature_m": 1.71,
        "reference_mass_kg": 78.0,
        "formula": "sqrt((M/78)*(1.71/H)^3); b=clip((s-1)/0.18,-1,1)",
        "assumption": "equal density and similar relative segment lengths; not measured girths",
    }


def build_player_spec(player: str, club: str = "driver") -> dict[str, Any]:
    """Build a newly sized current-model candidate, including the flexible shaft.

    This intentionally changes physics. Existing C3D motion/torques require a
    new fit; callers wanting only a skin must use :func:`player_appearance`.
    """
    from src.shared.python.model_appearance.player_stature import (
        build_stature_calibrated,
    )

    profile = player_profile(player)
    spec, calibration = build_stature_calibrated(
        profile["stature_m"],
        profile["mass_kg"],
        club,
        frame_build=frame_prior(profile)["value"],
    )
    spec["subject"]["neutral_visual_calibration"] = calibration
    return spec


def export_player(
    player: str,
    output: Path,
    club: str = "driver",
    *,
    base_spec: Path | None = None,
    cap_style: str | None = None,
) -> Path:
    """Write MJCF, spec, sidecar and receipt; a supplied base keeps all physics.

    Without ``base_spec`` this builds a separately sized anatomy candidate.
    Never applies published dimensions implicitly to an existing fitted spec.
    ``cap_style`` overrides only the visual head setting and is schema-validated
    before the output directory or any export files are created.
    """
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    profile, doc = player_profile(player), player_appearance(player)
    if cap_style is not None:
        head_changes: dict[str, Any] = {"cap_style": cap_style}
        if cap_style in {"flat_cap", "newsboy_cap"}:
            head_changes["cap_brim"] = 0.12
        if cap_style == "straw_hat":
            head_changes["cap_material"] = "hat_straw"
        elif cap_style in {"baseball_cap", "flat_cap", "newsboy_cap"} and (
            doc.head.cap_material == "hat_straw"
        ):
            head_changes["cap_material"] = "cap_white"
        overridden = replace(doc, head=replace(doc.head, **head_changes))
        doc = document_from_dict(document_to_dict(overridden))
    spec = (
        build_player_spec(player, club)
        if base_spec is None
        else json.loads(Path(base_spec).read_text(encoding="utf-8"))
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(spec, indent=2).encode()
    (output / "model.json").write_bytes(raw + b"\n")
    save_appearance(doc, output / "model.appearance.json")
    (output / "model.visual.json").write_text(
        json.dumps({**spec, "appearance": document_to_dict(doc)}, indent=2) + "\n"
    )
    xml, metadata = export_full_body_mjcf(raw, appearance=doc)
    target = output / "model.xml"
    target.write_text(xml, encoding="utf-8")
    receipt = {
        "player": player,
        "mode": "sized_candidate" if base_spec is None else "appearance_only",
        "profile": profile,
        "calibration": spec.get("subject", {}).get("neutral_visual_calibration"),
        "physics_spec_sha256": physics_spec_sha256(spec),
        "coordinate_order": spec["coordinate_order"],
        "visual_definition": "model.visual.json",
        "export": metadata,
        "status": "visual interpretation; supplied base dimensions retained"
        if base_spec
        else "published-dimension candidate; motion unqualified",
    }
    if base_spec is None:
        receipt["frame_prior"] = frame_prior(profile)
    if cap_style is not None:
        receipt["appearance_override"] = {
            "head.cap_style": doc.head.cap_style,
            "head.cap_material": doc.head.cap_material,
            "head.cap_brim": doc.head.cap_brim,
        }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return target
