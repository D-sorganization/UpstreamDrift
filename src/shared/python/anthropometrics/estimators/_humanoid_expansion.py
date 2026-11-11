"""Expand coarse published ratio tables to the humanoid segment graph.

Dempster (1955) and Zatsiorsky & Seluyanov (1985) tabulate eight segment
classes (head, trunk, upper_arm, forearm, hand, thigh, shank, foot). The
UpstreamDrift humanoid uses 21 anatomical names (``_SEGMENT_NAME_MAP`` in
``humanoid_character_builder.core.anthropometry``). Assigning every trunk
partition and virtual joint the *entire* trunk/limb class mass and then
renormalising inflated totals (#15014).

This module expands a coarse class table into per-class rows that close to
unit body mass without renormalisation:

* **Limbs** — left/right anatomical names share the published bilateral row
  mass (e.g. Dempster thigh 10 % per side).
* **Trunk interior** — the published trunk fraction splits across thorax,
  lumbar and pelvis by stature-length shares from model geometry (Drillis &
  Contini 1966; Winter 2009, segment-length table; same proportions as
  ``_GEOMETRY`` in ``anthropometry.py``).
* **Virtual joints** — neck, shoulder and hip masses follow
  ``VIRTUAL_SEGMENTS`` (de Leva humanoid convention, #13010).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from humanoid_character_builder.core.anthropometry import (
    VIRTUAL_SEGMENTS,
    _GEOMETRY,
    _SEGMENT_NAME_MAP,
    _VIRTUAL_INERTIA,
)

from ._base import SegmentRatios

if TYPE_CHECKING:
    from collections.abc import Mapping


_COARSE_CLASS_IDS = frozenset(
    {
        "head",
        "trunk",
        "upper_arm",
        "forearm",
        "hand",
        "thigh",
        "shank",
        "foot",
    }
)


#: Trunk partition uses male model geometry; both coarse tables are sex-agnostic.
_GEOM = _GEOMETRY["male"]

#: Interior trunk length sum (neck mass is carved from head, not trunk).
_TRUNK_LENGTH_SUM = _GEOM["thorax"][0] + _GEOM["lumbar"][0] + _GEOM["pelvis"][0]


def _virtual_mass_total(virtual_key: str) -> float:
    """Return total body-mass fraction for a bilateral virtual segment class."""
    parent, share = VIRTUAL_SEGMENTS[virtual_key]
    _ = parent
    from humanoid_character_builder.core.anthropometry import virtual_segment_count

    return share * virtual_segment_count(virtual_key)


def _trunk_piece_mass(trunk_mass_ratio: float, piece: str) -> float:
    """Allocate a fraction of published trunk mass to thorax/lumbar/pelvis."""
    length_share = _GEOM[piece][0] / _TRUNK_LENGTH_SUM
    return trunk_mass_ratio * length_share


def _virtual_ratios(virtual_key: str) -> SegmentRatios:
    """Build ratio row for a model-only virtual segment."""
    length, _width, _depth = _GEOM[virtual_key]
    com, sagittal, transverse, longitudinal = _VIRTUAL_INERTIA[virtual_key]
    parent, share = VIRTUAL_SEGMENTS[virtual_key]
    _ = parent
    return SegmentRatios(
        mass_ratio=share,
        length_ratio=length,
        com_proximal_ratio=com,
        gyration_sagittal=sagittal,
        gyration_transverse=transverse,
        gyration_longitudinal=longitudinal,
    )


def _scaled_ratios(
    template: SegmentRatios,
    *,
    mass_ratio: float,
    length_ratio: float,
) -> SegmentRatios:
    """Copy published radii/CoM onto a derived mass and length."""
    return SegmentRatios(
        mass_ratio=mass_ratio,
        length_ratio=length_ratio,
        com_proximal_ratio=template.com_proximal_ratio,
        gyration_sagittal=template.gyration_sagittal,
        gyration_transverse=template.gyration_transverse,
        gyration_longitudinal=template.gyration_longitudinal,
    )


def is_coarse_published_table(
    segment_classes: Mapping[str, SegmentRatios],
) -> bool:
    """Return True when *segment_classes* is an eight-class Dempster/Zatsiorsky table."""
    return _COARSE_CLASS_IDS.issubset(segment_classes.keys())


def _partition_trunk_masses(trunk: SegmentRatios) -> tuple[float, float, float]:
    """Return thorax/lumbar/pelvis masses carved from published trunk fraction."""
    thorax_mass = _trunk_piece_mass(trunk.mass_ratio, "thorax")
    lumbar_mass = _trunk_piece_mass(trunk.mass_ratio, "lumbar")
    pelvis_mass = _trunk_piece_mass(trunk.mass_ratio, "pelvis")
    shoulder_mass = _virtual_mass_total("shoulder")
    hip_mass = _virtual_mass_total("hip")
    thorax_mass -= shoulder_mass
    pelvis_mass -= hip_mass
    if thorax_mass <= 0 or pelvis_mass <= 0:
        raise ValueError(
            "trunk partition masses must stay positive after virtual carve"
        )
    return thorax_mass, lumbar_mass, pelvis_mass


def _limb_class_ratios(
    coarse_classes: Mapping[str, SegmentRatios],
) -> dict[str, SegmentRatios]:
    """Map published limb classes onto humanoid geometry length ratios."""
    mapping = (
        ("upper_arm", "upper_arm"),
        ("forearm", "forearm"),
        ("hand", "hand"),
        ("thigh", "thigh"),
        ("shin", "shank"),
        ("foot", "foot"),
    )
    return {
        humanoid_key: _scaled_ratios(
            coarse_classes[coarse_key],
            mass_ratio=coarse_classes[coarse_key].mass_ratio,
            length_ratio=_GEOM[humanoid_key][0],
        )
        for humanoid_key, coarse_key in mapping
    }


def resolve_humanoid_segment_table(
    segment_classes: Mapping[str, SegmentRatios],
    segment_name_map: Mapping[str, str],
) -> tuple[dict[str, SegmentRatios], dict[str, str], bool]:
    """Expand published coarse tables; pass custom tables through unchanged.

    Returns
    -------
    segment_classes, segment_name_map, normalize_mass
    """
    if is_coarse_published_table(segment_classes):
        expanded, name_map = expand_coarse_table_to_humanoid(segment_classes)
        return expanded, name_map, False
    return dict(segment_classes), dict(segment_name_map), True


def expand_coarse_table_to_humanoid(
    coarse_classes: Mapping[str, SegmentRatios],
) -> tuple[dict[str, SegmentRatios], dict[str, str]]:
    """Return ``(segment_classes, segment_name_map)`` for :func:`_base.build_subject_from_ratio_table`.

    Parameters
    ----------
    coarse_classes
        Published eight-class table keyed by Dempster / Zatsiorsky class id.

    Raises
    ------
    ValueError
        If required coarse keys are missing.
    """
    required = (
        "head",
        "trunk",
        "upper_arm",
        "forearm",
        "hand",
        "thigh",
        "shank",
        "foot",
    )
    missing = [key for key in required if key not in coarse_classes]
    if missing:
        raise ValueError(f"coarse table missing classes: {missing}")

    head = coarse_classes["head"]
    trunk = coarse_classes["trunk"]

    neck_mass = VIRTUAL_SEGMENTS["neck"][1]
    head_mass = head.mass_ratio - neck_mass
    if head_mass <= 0:
        raise ValueError("head mass after neck carve must stay positive")

    thorax_mass, lumbar_mass, pelvis_mass = _partition_trunk_masses(trunk)

    expanded: dict[str, SegmentRatios] = {
        "head": _scaled_ratios(
            head,
            mass_ratio=head_mass,
            length_ratio=_GEOM["head"][0],
        ),
        "neck": _virtual_ratios("neck"),
        "thorax": _scaled_ratios(
            trunk,
            mass_ratio=thorax_mass,
            length_ratio=_GEOM["thorax"][0],
        ),
        "lumbar": _scaled_ratios(
            trunk,
            mass_ratio=lumbar_mass,
            length_ratio=_GEOM["lumbar"][0],
        ),
        "pelvis": _scaled_ratios(
            trunk,
            mass_ratio=pelvis_mass,
            length_ratio=_GEOM["pelvis"][0],
        ),
        "shoulder": _virtual_ratios("shoulder"),
        "hip": _virtual_ratios("hip"),
    }
    expanded.update(_limb_class_ratios(coarse_classes))

    return expanded, dict(_SEGMENT_NAME_MAP)
