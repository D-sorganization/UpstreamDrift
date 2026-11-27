"""X-factor variant ids mapped to literature families (GM-1b / #14423).

Every published X-factor series should carry :data:`X_FACTOR_REGISTRY_VERSION`
and a :class:`LiteratureFamily` label. ``x_factor_position_proxy`` remains the
engine default for legacy continuity but is explicitly **non-literature**;
switching the default API is a versioned registry change, not a silent edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

LiteratureFamily = Literal[
    "mclean_line_projection",
    "brown_kwon_joint_angle",
    "non_literature_legacy",
]

X_FACTOR_REGISTRY_VERSION = "1.0.0"

# Brown et al. 2013 (PubMed) and Kwon et al. 2013 (DOI) per epic #14421 decision.
_CITE_BROWN2013 = "pubmed:23463985"
_CITE_KWON2013 = "doi:10.1080/14763141.2013.771896"
# McLean (1992) popularised shoulder–pelvis line separation; no DOI in-repo.
_CITE_MCLEAN1992 = "McLean1992-GolfMagazine-shoulder-pelvis-line"


@dataclass(frozen=True)
class XFactorVariantMeta:
    """Static metadata for one ``x_factor_*`` golf term."""

    variant_id: str
    literature_family: LiteratureFamily
    literature_aligned: bool
    cites: tuple[str, ...]
    non_literature_note: str | None = None


_VARIANTS: dict[str, XFactorVariantMeta] = {
    "x_factor_line_turn": XFactorVariantMeta(
        variant_id="x_factor_line_turn",
        literature_family="mclean_line_projection",
        literature_aligned=True,
        cites=(_CITE_MCLEAN1992,),
    ),
    "x_factor_jcs_axial": XFactorVariantMeta(
        variant_id="x_factor_jcs_axial",
        literature_family="brown_kwon_joint_angle",
        literature_aligned=True,
        cites=(_CITE_BROWN2013, _CITE_KWON2013),
    ),
    "x_factor_euler_yaw_legacy_v1": XFactorVariantMeta(
        variant_id="x_factor_euler_yaw_legacy_v1",
        literature_family="brown_kwon_joint_angle",
        literature_aligned=True,
        cites=(_CITE_BROWN2013, _CITE_KWON2013),
    ),
    "x_factor_position_proxy": XFactorVariantMeta(
        variant_id="x_factor_position_proxy",
        literature_family="non_literature_legacy",
        literature_aligned=False,
        cites=("ADR-0055-item-8", "swing_metrics.compute_x_factor_proxy"),
        non_literature_note=(
            "Joint-coordinate shoulder-minus-hip proxy; not a Brown/Kwon or McLean "
            "literature definition. Prefer jcs_axial or line_turn for cross-study "
            "labeling unless you intentionally publish this legacy measurand."
        ),
    ),
}


def x_factor_variant_meta(variant_id: str) -> XFactorVariantMeta:
    """Return metadata for a registered X-factor variant.

    Raises:
        ValueError: if ``variant_id`` is not one of the four X-factor variants.
    """
    try:
        return _VARIANTS[variant_id]
    except KeyError as exc:
        known = ", ".join(sorted(_VARIANTS))
        raise ValueError(
            f"unknown X-factor variant {variant_id!r}; expected one of: {known}"
        ) from exc


def x_factor_publish_payload(
    variant_id: str,
    values_deg: np.ndarray | Sequence[float],
    *,
    registry_version: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-friendly dict with values and mandatory registry labels."""
    meta = x_factor_variant_meta(variant_id)
    arr = np.asarray(values_deg, dtype=float)
    out: dict[str, Any] = {
        "variant_id": meta.variant_id,
        "registry_version": registry_version or X_FACTOR_REGISTRY_VERSION,
        "literature_family": meta.literature_family,
        "literature_aligned": meta.literature_aligned,
        "cites": list(meta.cites),
        "values_deg": arr.tolist(),
    }
    if meta.non_literature_note:
        out["non_literature_note"] = meta.non_literature_note
    return out


# Convenience for typing imports
__all__ = [
    "LiteratureFamily",
    "X_FACTOR_REGISTRY_VERSION",
    "XFactorVariantMeta",
    "x_factor_publish_payload",
    "x_factor_variant_meta",
]
