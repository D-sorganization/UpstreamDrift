"""One grip resolver for the CLI, desktop and web (HRG-GR4, #13205).

Precedence, highest first, the same rule as the appearance colours
(``model_appearance.appearance_selection``)::

    explicit picks (CLI flags / the user's controls)
      -> the document (the spec's grip, or the sidecar's visual block)
      -> the defaults (a default :class:`GripSpec`)

Two kinds of pick are kept apart:

* **Physical** picks change the hands' orientation, place on the shaft and
  contact topology: the rotation about the shaft (``alpha``), the palm diagonal
  (``psi``), the set angle (``gamma``), the lead hand's depth, the hand spacing
  and ``grip_style`` (since HRG-GS-G1, #13299).  They mint a new spec
  revision through :func:`~.grip_rotation.apply_grip_config` and are **never**
  written to an ``.appearance.json`` sidecar; the revision needs a refit.  When
  ``grip_style`` is the only physical pick and equals the document's current
  physical style (default ``overlap``), it is a no-op and the v1 bytes stay
  identical.
* **Visual** picks (the lead thumb offset, visual grip strength and per-hand
  roll offsets) only change the picture.  They go in the sidecar's ``grip``
  block and never move ``physics_spec_sha256``.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from src.shared.python.core.handedness import subject_handedness
from src.shared.python.contracts import require
from src.shared.python.model_appearance.grip_pose import GripStyle
from src.shared.python.motion_matching.execution.grip_rotation import (
    GRIP_PARAMETERS,
    apply_grip_config,
    grip_spec_of,
)
from src.shared.python.motion_matching.grip_contract import (
    GRIP_STRENGTH_PRESETS_DEG,
    NAMED_GRIP_PRESETS,
    GripSpec,
    from_grip_rotation,
    named_grip_preset,
)

NAMED_PRESET_CHOICES = tuple(NAMED_GRIP_PRESETS)
GRIP_PRESET_CHOICES = (*GRIP_STRENGTH_PRESETS_DEG, *NAMED_PRESET_CHOICES)
GRIP_STYLE_CHOICES = tuple(s.value for s in GripStyle)
#: The only keys a sidecar's ``grip`` block may carry (visual values).
VISUAL_KEYS = (
    "lead_thumb_offset_deg",
    "grip_strength_deg",
    "lead_roll_offset_deg",
    "trail_roll_offset_deg",
)
#: pick name -> (hand, angle) of the physical hand angles.
_HAND_PICKS = {
    "lead_shaft_rotation": ("lead", "alpha_deg"),
    "trail_shaft_rotation": ("trail", "alpha_deg"),
    "lead_palm_diagonal": ("lead", "psi_deg"),
    "trail_palm_diagonal": ("trail", "psi_deg"),
    "lead_set_angle": ("lead", "gamma_deg"),
    "trail_set_angle": ("trail", "gamma_deg"),
}
_SHARED_PICKS = {
    "lead_below_butt": "lead_below_butt_m",
    "hand_spacing": "hand_spacing_m",
}
PHYSICAL_PICKS = ("preset", *_HAND_PICKS, *_SHARED_PICKS, "grip_style")
VISUAL_PICKS = (
    "grip_thumb_offset_deg",
    "grip_strength_deg",
    "lead_roll_offset_deg",
    "trail_roll_offset_deg",
)


@dataclass(frozen=True)
class GripOverrides:
    """Explicit grip picks; ``None`` leaves a field to the lower layers.

    ``preset`` names a grip strength (``weak``, ``neutral``, ``strong``) or
    a named preset and sets both hands; an explicit hand pick then wins over
    it.  ``grip_style`` is physical (selects the contact topology since G1).
    ``grip_thumb_offset_deg``, ``grip_strength_deg``, ``lead_roll_offset_deg``
    and ``trail_roll_offset_deg`` are visual (picture only).
    """

    preset: str | None = None
    lead_shaft_rotation: float | None = None
    trail_shaft_rotation: float | None = None
    lead_palm_diagonal: float | None = None
    trail_palm_diagonal: float | None = None
    lead_set_angle: float | None = None
    trail_set_angle: float | None = None
    lead_below_butt: float | None = None
    hand_spacing: float | None = None
    grip_style: str | None = None  # physical (selects contact topology)
    grip_thumb_offset_deg: float | None = None  # visual
    grip_strength_deg: float | None = None  # visual
    lead_roll_offset_deg: float | None = None  # visual
    trail_roll_offset_deg: float | None = None  # visual

    def __post_init__(self) -> None:
        require(
            self.preset is None or self.preset in GRIP_PRESET_CHOICES,
            f"grip preset must be one of {GRIP_PRESET_CHOICES}, got {self.preset!r}",
        )
        require(
            self.grip_style is None or self.grip_style in GRIP_STYLE_CHOICES,
            f"grip style must be one of {GRIP_STYLE_CHOICES}, got {self.grip_style!r}",
        )
        for name in (
            *_HAND_PICKS,
            *_SHARED_PICKS,
            "grip_thumb_offset_deg",
            "grip_strength_deg",
            "lead_roll_offset_deg",
            "trail_roll_offset_deg",
        ):
            value = getattr(self, name)
            require(
                value is None
                or (isinstance(value, (int, float)) and math.isfinite(value)),
                f"{name} must be a finite number, got {value!r}",
            )

    @classmethod
    def from_picks(cls, picks: Mapping[str, Any]) -> GripOverrides:
        """Overrides from a picks mapping (unknown keys raise ``ValueError``)."""
        unknown = sorted(set(picks) - set(PHYSICAL_PICKS) - set(VISUAL_PICKS))
        if unknown:
            raise ValueError(f"unknown grip picks: {unknown}")
        return cls(**dict(picks))

    @property
    def physical(self) -> dict[str, Any]:
        """The physical picks that are set."""
        return {
            k: getattr(self, k) for k in PHYSICAL_PICKS if getattr(self, k) is not None
        }

    @property
    def visual(self) -> dict[str, Any]:
        """The visual picks that are set."""
        return {
            k: getattr(self, k) for k in VISUAL_PICKS if getattr(self, k) is not None
        }

    def visual_only(self) -> GripOverrides:
        """A copy with the physical picks cleared (what a sidecar may carry)."""
        return replace(self, **dict.fromkeys(PHYSICAL_PICKS))

    @property
    def has_physical(self) -> bool:
        return bool(self.physical)

    @property
    def has_visual(self) -> bool:
        return bool(self.visual)


def _document_physical_style(
    document: Mapping[str, Any] | GripSpec | GripStyle | str | None = None,
) -> GripStyle:
    """Resolve the physical ``GripStyle`` of a spec document or ``GripSpec``."""
    if document is None:
        return GripStyle.OVERLAP
    if isinstance(document, GripStyle):
        return document
    if isinstance(document, GripSpec):
        return document.style
    if isinstance(document, str):
        return GripStyle(document)
    if isinstance(document, Mapping):
        cfg = (document.get("subject") or {}).get("grip_config") or {}
        return GripStyle(cfg.get("style", GripStyle.OVERLAP.value))
    raise TypeError(f"unsupported document type for physical style: {type(document)!r}")


def check_sidecar_grip(block: Mapping[str, Any]) -> None:
    """Refuse a sidecar ``grip`` block that carries a physical value.

    ``ValueError`` names the offending keys and where physical values belong.
    """
    moved = sorted(set(block) - set(VISUAL_KEYS))
    if moved:
        raise ValueError(
            f"the appearance sidecar may hold only visual grip values "
            f"{list(VISUAL_KEYS)}, but it carries {moved}: style, strength, "
            "palm diagonal, set angle, depth and hand spacing are physical and "
            "make a new model revision (--grip-preset, --grip-style and the "
            "other grip flags of spec_builder); they need a refit"
        )


def migrate_sidecar_grip(
    block: Mapping[str, Any] | None,
    document: Mapping[str, Any] | GripSpec | GripStyle | str | None = None,
) -> dict[str, Any]:
    """Validate a sidecar ``grip`` block and migrate a legacy read-only ``style``.

    Compares any legacy ``grip.style`` with the physical
    ``subject.grip_config.style`` (default ``overlap``), raising ``ValueError``
    naming both values if they disagree and dropping ``style`` on agreement.
    """
    if block is None:
        return {}
    require(isinstance(block, Mapping), "sidecar grip block must be a mapping")
    cleaned = dict(block)
    if "style" in cleaned:
        raw_legacy = cleaned.pop("style")
        legacy = (
            raw_legacy.value if isinstance(raw_legacy, GripStyle) else str(raw_legacy)
        )
        physical = _document_physical_style(document).value
        if legacy not in GRIP_STYLE_CHOICES or legacy != physical:
            raise ValueError(
                f"legacy sidecar grip.style={legacy!r} disagrees with physical "
                f"subject.grip_config.style={physical!r}; grip_style is a "
                "physical pick and sidecar and physical style can never disagree"
            )
    check_sidecar_grip(cleaned)
    return cleaned


def document_grip(document: Mapping[str, Any]) -> GripSpec | None:
    """The grip a spec document carries: its configuration, else its fit.

    A document without ``subject.grip_config`` yields the grip of its
    ``subject.grip_rotation_deg`` (the fitted default) and the default depth and
    spacing; without either, a default :class:`GripSpec`.  ``None`` when the
    fitted rotation lies outside the golf-angle ranges (for example a lead hand
    turned weaker than the V azimuth allows): it is then not a grip the picks
    can start from.
    """
    subject = document.get("subject", {})
    if subject.get("grip_config") is not None:
        return grip_spec_of(document)
    rotation = subject.get("grip_rotation_deg")
    template = GripSpec(handedness=subject_handedness(document))
    if rotation is None:
        return template
    try:
        return from_grip_rotation(rotation, template=template)
    except ValueError:  # includes DbC PreconditionError
        return None


def _determines_both_hands(picks: GripOverrides) -> bool:
    """True when the picks name every angle of both hands (nothing is inherited)."""
    if picks.preset is not None:
        return True
    return all(getattr(picks, name) is not None for name in _HAND_PICKS)


def _has_effective_physical_change(
    document: Mapping[str, Any], overrides: GripOverrides
) -> bool:
    """True when ``overrides`` changes any physical grip parameter on ``document``."""
    if not overrides.has_physical:
        return False
    if set(overrides.physical) - {"grip_style"}:
        return True
    return overrides.grip_style is not None and GripStyle(
        overrides.grip_style
    ) is not _document_physical_style(document)


def _with_hand_picks(grip: GripSpec, overrides: GripOverrides) -> GripSpec:
    hands = {"lead": grip.lead, "trail": grip.trail}
    for pick, (hand, angle) in _HAND_PICKS.items():
        value = getattr(overrides, pick)
        if value is not None:
            hands[hand] = replace(hands[hand], **{angle: float(value)})
    depth = overrides.lead_below_butt
    spacing = overrides.hand_spacing
    style = (
        grip.style if overrides.grip_style is None else GripStyle(overrides.grip_style)
    )
    return replace(
        grip,
        lead=hands["lead"],
        trail=hands["trail"],
        lead_below_butt_m=grip.lead_below_butt_m if depth is None else float(depth),
        hand_spacing_m=grip.hand_spacing_m if spacing is None else float(spacing),
        style=style,
    )


def _with_visual(grip: GripSpec, thumb: Any) -> GripSpec:
    if thumb is None:
        return grip
    return replace(grip, lead_thumb_offset_deg=float(thumb))


def _apply_named_preset(grip: GripSpec, name: str) -> GripSpec:
    """Apply a named grip preset, keeping the grip's handedness."""
    preset = named_grip_preset(name)
    return preset.for_handedness(grip.handedness)


def _named_preset_provenance(overrides: GripOverrides) -> dict[str, str]:
    """Per-parameter provenance for a named preset with any explicit overrides."""
    prov_map = dict.fromkeys(GRIP_PARAMETERS, "photo_estimate")
    for pick, (hand, angle) in _HAND_PICKS.items():
        if getattr(overrides, pick) is not None:
            prov_map[f"{hand}.{angle}"] = "user_declared"
    if overrides.lead_below_butt is not None:
        prov_map["lead_below_butt_m"] = "user_declared"
    if overrides.hand_spacing is not None:
        prov_map["hand_spacing_m"] = "user_declared"
    return prov_map


def resolve_grip(
    overrides: GripOverrides | None,
    document: GripSpec | None = None,
    sidecar_grip: Mapping[str, Any] | None = None,
) -> GripSpec:
    """The grip a build renders: picks, then the document, then the defaults.

    ``document`` is the spec's grip (default: a default :class:`GripSpec`) and
    ``sidecar_grip`` the sidecar's visual block (a physical key or a disagreeing
    legacy ``style`` raises ``ValueError``).  Pure: nothing is read from the
    environment or the disk.  Postcondition: a validated :class:`GripSpec`.
    """
    picks = overrides or GripOverrides()
    grip = document if document is not None else GripSpec()
    if sidecar_grip:
        migrated = migrate_sidecar_grip(sidecar_grip, grip)
        grip = _with_visual(grip, migrated.get("lead_thumb_offset_deg"))
    if picks.preset is not None:
        if picks.preset in NAMED_GRIP_PRESETS:
            grip = _apply_named_preset(grip, picks.preset)
        else:
            hand = GripSpec.preset(picks.preset)
            grip = replace(grip, lead=hand.lead, trail=hand.trail)
    grip = _with_hand_picks(grip, picks)
    return _with_visual(grip, picks.grip_thumb_offset_deg)


def apply_grip_selection(
    document: Mapping[str, Any],
    overrides: GripOverrides,
    sidecar_grip: Mapping[str, Any] | None = None,
    *,
    provenance: str | Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], GripSpec | None]:
    """``(document, grip)`` with the picks resolved against ``document``.

    A physical pick that changes the configuration returns a new GR2 revision
    (:func:`apply_grip_config`, ``spec_revision`` set, a refit needed).  Visual
    picks alone (or picking the unchanged default style) return an unchanged
    copy: ``physics_spec_sha256`` does not move.
    """
    require(isinstance(overrides, GripOverrides), "overrides must be GripOverrides")
    migrated = (
        migrate_sidecar_grip(sidecar_grip, document)
        if sidecar_grip is not None
        else None
    )
    effective_physical = _has_effective_physical_change(document, overrides)
    non_style_physical = bool(set(overrides.physical) - {"grip_style"})
    has_config = (document.get("subject") or {}).get("grip_config") is not None
    if effective_physical and not non_style_physical and not has_config:
        base: GripSpec | None = GripSpec(handedness=subject_handedness(document))
    else:
        base = document_grip(document)
    if base is None:
        if non_style_physical and not _determines_both_hands(overrides):
            raise ValueError(
                "the document's fitted hand rotation is not a golf grip the "
                "picks can adjust (outside the strength or palm ranges): pick a "
                "grip preset, or give the shaft rotation, palm diagonal and set "
                "angle of both hands"
            )
        if not effective_physical:
            return {**document}, None
        base = GripSpec(handedness=subject_handedness(document))
    grip = resolve_grip(overrides, base, migrated)
    if effective_physical:
        prov = (
            provenance
            if provenance is not None
            else (
                _named_preset_provenance(overrides)
                if overrides.preset in NAMED_GRIP_PRESETS
                else "user_declared"
            )
        )
        return apply_grip_config(document, grip, provenance=prov), grip
    return {**document}, grip


def visual_grip_block(
    grip: GripSpec, overrides: GripOverrides | None = None
) -> dict[str, Any]:
    """The sidecar ``grip`` block of ``grip``: visual values only."""
    block: dict[str, Any] = {
        "lead_thumb_offset_deg": grip.lead_thumb_offset_deg,
    }
    if overrides is not None:
        if overrides.grip_strength_deg is not None:
            block["grip_strength_deg"] = float(overrides.grip_strength_deg)
        if overrides.lead_roll_offset_deg is not None:
            block["lead_roll_offset_deg"] = float(overrides.lead_roll_offset_deg)
        if overrides.trail_roll_offset_deg is not None:
            block["trail_roll_offset_deg"] = float(overrides.trail_roll_offset_deg)
    return block


def apply_visual_picks(
    appearance: Mapping[str, Any], overrides: GripOverrides
) -> dict[str, Any]:
    """``appearance`` with the visual picks in its ``grip`` block.

    Precedence: a pick, then the document's own block.  Physical picks raise
    ``ValueError`` (a sidecar never carries them).
    """
    refuse_physical_in_sidecar(overrides)
    out = dict(appearance)
    block = dict(out.get("grip", {}))
    check_sidecar_grip(block)
    if overrides.grip_thumb_offset_deg is not None:
        block["lead_thumb_offset_deg"] = float(overrides.grip_thumb_offset_deg)
    if overrides.grip_strength_deg is not None:
        block["grip_strength_deg"] = float(overrides.grip_strength_deg)
    if overrides.lead_roll_offset_deg is not None:
        block["lead_roll_offset_deg"] = float(overrides.lead_roll_offset_deg)
    if overrides.trail_roll_offset_deg is not None:
        block["trail_roll_offset_deg"] = float(overrides.trail_roll_offset_deg)
    if block:
        out["grip"] = block
    return out


def refuse_physical_in_sidecar(overrides: GripOverrides) -> None:
    """``ValueError`` if a physical pick is headed for an appearance sidecar."""
    if overrides.has_physical:
        raise ValueError(
            f"{sorted(overrides.physical)} change the hands' physics, so they "
            "cannot be applied as a picture-only (appearance) choice: build a "
            "new model revision with spec_builder (--grip-preset, --grip-style "
            "and the other grip flags), refit it, then export from that "
            "revision's receipt"
        )


_FLAGS: tuple[tuple[str, str, type, str], ...] = (
    (
        "--lead-shaft-rotation",
        "lead_shaft_rotation",
        float,
        "lead hand rotation about the shaft, degrees; positive is a stronger "
        "grip (more knuckles showing), negative weaker",
    ),
    (
        "--trail-shaft-rotation",
        "trail_shaft_rotation",
        float,
        "trail hand rotation about the shaft, degrees (positive is stronger)",
    ),
    (
        "--lead-palm-diagonal",
        "lead_palm_diagonal",
        float,
        "angle of the shaft across the lead palm, degrees: smaller sits more in "
        "the fingers, larger more in the palm (default 40)",
    ),
    (
        "--trail-palm-diagonal",
        "trail_palm_diagonal",
        float,
        "angle of the shaft across the trail palm, degrees",
    ),
    (
        "--hand-spacing",
        "hand_spacing",
        float,
        "distance between the hands on the grip, metres (default 0.0762, three "
        "inches); must be positive",
    ),
)


def add_grip_arguments(parser: argparse.ArgumentParser, *, physical: bool) -> None:
    """The grip flags (HRG-GR4, HRG-A2).  ``physical`` False keeps picture-only flags.

    Precedence: a flag, then the document, then the defaults.  Physical flags
    (including ``--grip-style`` since HRG-GS-G1) make a new model revision that
    needs a refit; ``--grip-thumb-offset-deg``, ``--visual-grip-strength``,
    ``--lead-roll-offset`` and ``--trail-roll-offset`` are picture-only.
    """
    group = parser.add_argument_group(
        "grip",
        "golf grip; physical flags (including --grip-style) make a new model "
        "revision (refit needed), --grip-thumb-offset-deg, --visual-grip-strength, "
        "--lead-roll-offset, and --trail-roll-offset are picture-only",
    )
    group.add_argument(
        "--grip-thumb-offset-deg",
        dest="grip_thumb_offset_deg",
        type=float,
        default=None,
        help="lead thumb azimuth offset from shaft top, degrees (picture only; no refit)",
    )
    group.add_argument(
        "--visual-grip-strength",
        dest="grip_strength_deg",
        type=float,
        default=None,
        help="visual grip strength azimuth shift in degrees ([-30, 30]; picture only; no refit)",
    )
    group.add_argument(
        "--lead-roll-offset",
        dest="lead_roll_offset_deg",
        type=float,
        default=None,
        help="visual lead hand roll offset in degrees ([-30, 30]; picture only; no refit)",
    )
    group.add_argument(
        "--trail-roll-offset",
        dest="trail_roll_offset_deg",
        type=float,
        default=None,
        help="visual trail hand roll offset in degrees ([-30, 30]; picture only; no refit)",
    )
    if not physical:
        return
    group.add_argument(
        "--grip-style",
        default=None,
        choices=GRIP_STYLE_CHOICES,
        help="how the little finger of the trail hand sits: overlap, interlock "
        "or ten_finger (physical: selects contact topology; refit needed)",
    )
    group.add_argument(
        "--grip-preset",
        default=None,
        choices=GRIP_PRESET_CHOICES,
        help="both hands at a named strength: weak (about one lead knuckle "
        "showing), neutral (two) or strong (three)",
    )
    for flag, dest, kind, text in _FLAGS:
        group.add_argument(flag, dest=dest, type=kind, default=None, help=text)


def overrides_from_args(args: argparse.Namespace) -> GripOverrides:
    """:class:`GripOverrides` of parsed arguments (missing flags are ``None``)."""
    names = (
        "preset",
        *(dest for _f, dest, _k, _t in _FLAGS),
        "grip_style",
        "grip_thumb_offset_deg",
        "grip_strength_deg",
        "lead_roll_offset_deg",
        "trail_roll_offset_deg",
    )
    picks = {n: getattr(args, f"grip_{n}" if n == "preset" else n, None) for n in names}
    return GripOverrides(**picks)
