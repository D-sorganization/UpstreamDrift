"""Source-attributed player candidates on the current full-body golfer pipeline.

Dimensions are published priors; segment proportions and visual likeness are
interpretations, not measured player anatomy or fitted historical motion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.shared.python.model_appearance.schema import (
    AppearanceDocument,
    document_from_dict,
    physics_spec_sha256,
    save_appearance,
)

CATALOG = Path(__file__).resolve().parents[3] / "config/player_appearance_presets.json"


def player_profile(player: str) -> dict[str, Any]:
    """Return a fresh profile; reject unknown IDs instead of guessing a subject."""
    profiles = json.loads(CATALOG.read_text(encoding="utf-8"))["players"]
    if player not in profiles:
        raise ValueError(f"Unknown player {player!r}; choose {sorted(profiles)}")
    return profiles[player]


def player_appearance(player: str) -> AppearanceDocument:
    """Appearance usable directly on any compatible existing full-body spec."""
    return document_from_dict(player_profile(player)["appearance"])


def build_player_spec(player: str, club: str = "driver") -> dict[str, Any]:
    """Build a newly sized current-model candidate, including the flexible shaft.

    This intentionally changes physics. Existing C3D motion/torques require a
    new fit; callers wanting only a skin must use :func:`player_appearance`.
    """
    from src.shared.python.model_appearance.canonical_golfer_gate import (
        require_canonical_golfer,
    )
    from src.shared.python.motion_matching.execution.spec_builder import (
        compose_anthropometric_document,
    )
    from src.shared.python.motion_matching.anthropometry import segment_parameters
    from src.shared.python.motion_matching.execution.spec_revisions import (
        with_current_golfer_model,
        with_grip_revision,
    )

    profile = player_profile(player)
    built = compose_anthropometric_document(
        stature_m=profile["stature_m"],
        mass_kg=profile["mass_kg"],
        club=club,
        match_subject_mass=True,
        thigh_length_m=segment_parameters(
            profile["stature_m"], profile["mass_kg"], "thigh"
        ).length_m,
        shank_length_m=segment_parameters(
            profile["stature_m"], profile["mass_kg"], "shank"
        ).length_m,
    )
    spec, _ = with_current_golfer_model(
        *with_grip_revision(built.document, built.subject, None)
    )
    require_canonical_golfer(spec)
    return spec


def export_player(
    player: str, output: Path, club: str = "driver", *, base_spec: Path | None = None
) -> Path:
    """Write MJCF, spec, sidecar and receipt; a supplied base keeps all physics.

    Without ``base_spec`` this builds a separately sized anatomy candidate.
    Never applies published dimensions implicitly to an existing fitted spec.
    """
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    profile, doc = player_profile(player), player_appearance(player)
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
    xml, metadata = export_full_body_mjcf(raw, appearance=doc)
    target = output / "model.xml"
    target.write_text(xml, encoding="utf-8")
    receipt = {
        "player": player,
        "mode": "sized_candidate" if base_spec is None else "appearance_only",
        "profile": profile,
        "physics_spec_sha256": physics_spec_sha256(spec),
        "coordinate_order": spec["coordinate_order"],
        "export": metadata,
        "status": "visual interpretation; supplied base dimensions retained"
        if base_spec
        else "published-dimension candidate; motion unqualified",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return target
