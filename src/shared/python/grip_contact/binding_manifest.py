"""Versioned per-model grip binding manifest loading and validation (HRG-GS-M2, #13324).

Implements the ``grip-binding/v1`` manifest schema, frame/unit validators, and
calibration-time resolution of :class:`~src.shared.python.grip_contact.binding.HandClubBinding`
from :class:`~src.shared.python.grip_contact.contact_topology.ContactTopology`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

import numpy as np

from src.shared.python.core.handedness import (
    Handedness,
    validate_handedness,
)
from src.shared.python.grip_contact.binding import (
    HandClubBinding,
    SiteBinding,
    _require_non_empty_str,
    _validate_transform_4x4,
    bind_rigid_hand_sites,
)
from src.shared.python.grip_contact.contact_topology import ContactTopology
from src.shared.python.grip_contact.interface import GripInterface

__all__ = [
    "AXIAL_BOUNDS_M",
    "AXIS_ALIGNMENT_TOL",
    "MANIFEST_SCHEMA",
    "RADIAL_BOUNDS_M",
    "binding_from_manifest",
    "validate_manifest",
    "validate_resolved_binding",
]

MANIFEST_SCHEMA: Final[str] = "grip-binding/v1"
AXIAL_BOUNDS_M: Final[tuple[float, float]] = (-0.35, 0.05)
RADIAL_BOUNDS_M: Final[tuple[float, float]] = (0.005, 0.060)
AXIS_ALIGNMENT_TOL: Final[float] = 0.999999


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate a grip binding manifest against schema ``grip-binding/v1``.

    Preconditions:
        ``manifest`` is a Mapping with schema ``"grip-binding/v1"``, valid
        units (metres, radians), valid handedness, and all required keys.
    Raises:
        TypeError: if ``manifest`` is not a Mapping.
        ValueError: if required keys are missing, the schema version is
            unsupported, units are not metres/radians, handedness or roles are
            invalid, or grip axis is not aligned with +x.
    """
    if not isinstance(manifest, Mapping):
        raise TypeError(f"manifest must be a Mapping, got {type(manifest).__name__}")
    if "schema" not in manifest:
        raise ValueError("manifest missing required key: 'schema'")
    schema = manifest["schema"]
    if schema != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported schema {schema!r}, expected {MANIFEST_SCHEMA!r}")
    required_keys = (
        "schema",
        "model_id",
        "model_hash",
        "handedness",
        "side_roles",
        "calibration_pose",
        "units",
        "provenance",
        "attachment_mode",
        "club_body",
        "lead_hand_body",
        "trail_hand_body",
        "unavailable_sites",
    )
    for key in required_keys:
        if key not in manifest:
            raise ValueError(f"manifest missing required key: {key!r}")

    _require_non_empty_str(manifest["model_id"], "model_id")
    _require_non_empty_str(manifest["model_hash"], "model_hash")
    validate_handedness(manifest["handedness"])
    _require_non_empty_str(manifest["attachment_mode"], "attachment_mode")
    _require_non_empty_str(manifest["club_body"], "club_body")
    _require_non_empty_str(manifest["lead_hand_body"], "lead_hand_body")
    _require_non_empty_str(manifest["trail_hand_body"], "trail_hand_body")

    units = manifest["units"]
    if not isinstance(units, Mapping):
        raise TypeError("units must be a Mapping")
    if units.get("length") != "m":
        raise ValueError(
            f"units length must be metres ('m'), got {units.get('length')!r}"
        )
    if units.get("angle") != "rad":
        raise ValueError(
            f"units angle must be radians ('rad'), got {units.get('angle')!r}"
        )

    side_roles = manifest["side_roles"]
    if not isinstance(side_roles, Mapping):
        raise TypeError("side_roles must be a Mapping")
    roles_set = set(side_roles.values())
    if roles_set != {"lead", "trail"}:
        raise ValueError(
            f"side_roles must map anatomical sides to {{'lead', 'trail'}}, got {roles_set}"
        )

    unavail = manifest["unavailable_sites"]
    if not isinstance(unavail, Mapping):
        raise TypeError("unavailable_sites must be a Mapping")
    for k, v in unavail.items():
        _require_non_empty_str(k, "unavailable_sites key")
        _require_non_empty_str(v, f"unavailable_sites[{k!r}] reason")

    if "segment_map" in manifest:
        seg_map = manifest["segment_map"]
        if not isinstance(seg_map, Mapping):
            raise TypeError("segment_map must be a Mapping")
        for s_key, b_val in seg_map.items():
            _require_non_empty_str(s_key, "segment_map key")
            _require_non_empty_str(b_val, f"segment_map[{s_key!r}] body")

    grip_axis = manifest.get("grip_axis", manifest.get("shaft_axis"))
    if grip_axis is not None:
        if isinstance(grip_axis, str) and grip_axis not in ("x", "+x"):
            raise ValueError(f"grip axis must be along +x, got {grip_axis!r}")
        if isinstance(grip_axis, tuple | list | np.ndarray):
            arr = np.asarray(grip_axis, dtype=float)
            x_unit = np.array([1.0, 0.0, 0.0])
            if arr.shape != (3,) or float(np.dot(arr, x_unit)) < AXIS_ALIGNMENT_TOL:
                raise ValueError(
                    f"grip axis vector must align with +x [1, 0, 0], got {grip_axis!r}"
                )


def validate_resolved_binding(
    binding: HandClubBinding,
    topology: ContactTopology,
    interface: Any = None,
    *,
    expected_axis: np.ndarray | Sequence[float] | None = None,
) -> None:
    """Validate geometry, axis alignment, and rotation of a resolved binding.

    Preconditions:
        ``binding`` is a :class:`HandClubBinding`;
        ``topology`` is a :class:`ContactTopology`;
        every site's rotation is proper within 1e-9;
        every site's radial offset from grip axis is in [0.005, 0.060] m;
        every site's axial position is in [-0.35, 0.05] m;
        every site's +x axis dot product against expected axis is >= 0.999999.
    Raises:
        TypeError: if ``binding`` or ``topology`` is of wrong type.
        ValueError: on out-of-bounds geometry, improper rotation, or axis misalignment.
    """
    if not isinstance(binding, HandClubBinding):
        raise TypeError(
            f"binding must be HandClubBinding, got {type(binding).__name__}"
        )
    if not isinstance(topology, ContactTopology):
        raise TypeError(
            f"topology must be ContactTopology, got {type(topology).__name__}"
        )

    exp_ax = (
        np.array([1.0, 0.0, 0.0])
        if expected_axis is None
        else np.asarray(expected_axis, dtype=float)
    )
    if exp_ax.shape != (3,):
        raise ValueError("expected_axis must be a 3-vector")
    norm = float(np.linalg.norm(exp_ax))
    if norm < 1e-9:
        raise ValueError("expected_axis must have non-zero length")
    exp_ax = exp_ax / norm

    r_min, r_max = RADIAL_BOUNDS_M
    x_min, x_max = AXIAL_BOUNDS_M

    for sb in binding.sites:
        _validate_transform_4x4(sb.body_to_site, f"site {sb.site_id!r} transform")
        p = sb.translation_m
        axial_x = float(p[0])
        if not (x_min <= axial_x <= x_max):
            raise ValueError(
                f"site {sb.site_id!r} axial position {axial_x:.4f} m "
                f"out of bounds [{x_min}, {x_max}] m"
            )
        radial_r = float(math.hypot(p[1], p[2]))
        if not (r_min <= radial_r <= r_max):
            raise ValueError(
                f"site {sb.site_id!r} radial offset {radial_r:.4f} m "
                f"out of bounds [{r_min}, {r_max}] m"
            )
        ux = sb.body_to_site[:3, 0]
        dot_x = float(np.dot(ux, exp_ax))
        if dot_x < AXIS_ALIGNMENT_TOL:
            raise ValueError(
                f"site {sb.site_id!r} grip +x axis dot product {dot_x:.6f} "
                f"< {AXIS_ALIGNMENT_TOL} against expected axis"
            )

    for side_key, mat in binding.club_grip_frames.items():
        _validate_transform_4x4(mat, f"club_grip_frames[{side_key!r}]")


def _resolve_club_grip_frames(interface_or_resolver: Any) -> dict[str, np.ndarray]:
    """Extract per-hand club grip frame transforms from interface or mapping."""
    if isinstance(interface_or_resolver, GripInterface):
        left_f = interface_or_resolver.left
        right_f = interface_or_resolver.right
        return {
            "L": left_f.matrix(),
            "R": right_f.matrix(),
        }
    if isinstance(interface_or_resolver, Mapping) and (
        "L" in interface_or_resolver or "R" in interface_or_resolver
    ):
        return {k: np.asarray(v, dtype=float) for k, v in interface_or_resolver.items()}
    if hasattr(interface_or_resolver, "left") and hasattr(
        interface_or_resolver, "right"
    ):
        left_f = interface_or_resolver.left
        right_f = interface_or_resolver.right
        return {
            "L": left_f.matrix(),
            "R": right_f.matrix(),
        }
    return {}


def _resolve_callable_sites(
    topology: ContactTopology,
    resolver: Any,
    *,
    lead_hand_body: str,
    trail_hand_body: str,
    unavail: Mapping[str, str],
    seg_map: Mapping[str, str],
) -> tuple[SiteBinding, ...]:
    sites_list: list[SiteBinding] = []
    for site in topology.sites:
        if site.name in unavail:
            continue
        is_lead = site.hand == "lead"
        rep: Literal["rigid_proxy", "phalange"]
        if site.name in seg_map:
            b_id = str(seg_map[site.name])
            rep = "phalange"
        else:
            b_id = lead_hand_body if is_lead else trail_hand_body
            rep = "rigid_proxy"
        try:
            res = resolver(site.name)
        except TypeError:
            res = resolver(site)
        if isinstance(res, SiteBinding):
            sites_list.append(res)
        else:
            mat = _validate_transform_4x4(res, f"resolver({site.name!r})")
            sites_list.append(
                SiteBinding(
                    site_id=site.name,
                    body_id=b_id,
                    body_to_site=mat,
                    representation=rep,
                )
            )
    return tuple(sites_list)


def binding_from_manifest(
    manifest: Mapping[str, Any],
    topology: ContactTopology,
    interface_or_resolver: Any = None,
) -> HandClubBinding:
    """Resolve a :class:`HandClubBinding` from a manifest and topology at calibration.

    Preconditions:
        ``manifest`` satisfies :func:`validate_manifest`;
        ``topology`` is a :class:`ContactTopology`;
        ``interface_or_resolver`` provides club grip frames or custom site resolution.
    Postconditions:
        Returns a validated :class:`HandClubBinding` satisfying
        :func:`validate_resolved_binding`.
    """
    validate_manifest(manifest)
    if not isinstance(topology, ContactTopology):
        raise TypeError(
            f"topology must be ContactTopology, got {type(topology).__name__}"
        )

    model_id = str(manifest["model_id"])
    model_hash = str(manifest["model_hash"])
    handedness = validate_handedness(manifest["handedness"])
    attachment_mode = str(manifest["attachment_mode"])
    club_body = str(manifest["club_body"])
    lead_hand_body = str(manifest["lead_hand_body"])
    trail_hand_body = str(manifest["trail_hand_body"])
    unavail = dict(manifest.get("unavailable_sites", {}))
    seg_map = dict(manifest.get("segment_map", {}))

    club_grip_frames = _resolve_club_grip_frames(interface_or_resolver)

    if callable(interface_or_resolver):
        resolved_sites = _resolve_callable_sites(
            topology,
            interface_or_resolver,
            lead_hand_body=lead_hand_body,
            trail_hand_body=trail_hand_body,
            unavail=unavail,
            seg_map=seg_map,
        )
    else:
        raw_sites = bind_rigid_hand_sites(
            topology,
            lead_hand_body=lead_hand_body,
            trail_hand_body=trail_hand_body,
            handedness=handedness,
        )
        filtered: list[SiteBinding] = []
        for s in raw_sites:
            if s.site_id in unavail:
                continue
            if s.site_id in seg_map:
                filtered.append(
                    SiteBinding(
                        site_id=s.site_id,
                        body_id=str(seg_map[s.site_id]),
                        body_to_site=s.body_to_site,
                        representation="phalange",
                    )
                )
            else:
                filtered.append(s)
        resolved_sites = tuple(filtered)

    binding = HandClubBinding(
        model_id=model_id,
        model_hash=model_hash,
        handedness=handedness,
        attachment_mode=attachment_mode,
        club_body=club_body,
        lead_hand_body=lead_hand_body,
        trail_hand_body=trail_hand_body,
        sites=resolved_sites,
        unavailable_sites=unavail,
        club_grip_frames=club_grip_frames,
    )
    validate_resolved_binding(binding, topology, interface_or_resolver)
    return binding
