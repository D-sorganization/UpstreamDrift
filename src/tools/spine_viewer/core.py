"""Qt-free data layer of the Spine Viewer (SPN-15b, #12772).

Opens a hash-verified study receipt (SPN-13a/13c) and exposes it through the
merged visualization adapter (:mod:`src.shared.python.spine.viz_data`) and
figures (:mod:`src.shared.python.spine.plots`). Nothing is recomputed here.

Scientific status is fail-closed: a view is ``"qualified"`` only when the
receipt both declares ``scientific_status == "qualified"`` and has
``status == "pass"``; anything else is ``"unqualified"``. The viewer never
promotes a status; the pass/fail verdict of a run is not a qualification.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from src.engines.physics_engines.opensim.python.spine.study_aggregation import (
    load_verified_run,
)
from src.engines.physics_engines.opensim.python.spine.study_receipt import (
    SpineStudyReceipt,
)
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.loads import LoadObservable, SpinalLoadSeries
from src.shared.python.spine.plots import plot_angle_stack, plot_load_heatmap
from src.shared.python.spine.segmental import SegmentalPose
from src.shared.python.spine.viz_data import (
    HeatmapData,
    StackData,
    level_angle_stack,
    load_heatmap,
)

__all__ = [
    "LOAD_QUANTITIES",
    "SpineStudyView",
    "angle_stack_figure",
    "heatmap_figure",
    "open_study",
    "resolve_scientific_status",
    "status_banner",
    "view_angle_stack",
    "view_to_dict",
    "view_heatmap",
]

LOAD_QUANTITIES = ("compression", "tension", "ap_shear", "ml_shear")
_NOT_RECORDED = "not recorded in receipt"


@dataclass(frozen=True)
class SpineStudyView:
    """A verified study ready for display (immutable).

    ``series`` is the joint-reaction wrench per level; ``scientific_status``
    is ``"unqualified"`` unless the receipt earned ``"qualified"``.
    """

    study_id: str
    receipt_status: str
    status_reasons: tuple[str, ...]
    scientific_status: str
    distribution_source: str
    levels: tuple[SpinalLevel, ...]
    series: SpinalLoadSeries


def resolve_scientific_status(receipt: SpineStudyReceipt) -> str:
    """Fail-closed status: ``"qualified"`` only for a passing, declared receipt.

    Raises:
        TypeError: if ``receipt`` is not a :class:`SpineStudyReceipt`.
    """
    if not isinstance(receipt, SpineStudyReceipt):
        raise TypeError("receipt must be a SpineStudyReceipt")
    declared = receipt.scientific_status == "qualified"
    return "qualified" if declared and receipt.status == "pass" else "unqualified"


def _provenance(receipt: SpineStudyReceipt) -> dict[str, str]:
    request = receipt.request
    return {
        "model_id": str(request.get("model_id") or _NOT_RECORDED),
        "model_sha256": str(request.get("model_sha256") or _NOT_RECORDED),
        "solver": "JointReaction",
        "cost_exponent": "n/a",
        "distribution_source": receipt.distribution_source,
        "external_loads": str(request.get("external_loads") or _NOT_RECORDED),
    }


def open_study(
    receipt_path: str | Path, artifact_dir: str | Path | None = None
) -> SpineStudyView:
    """Load a receipt, verify its artifact hash and build a display view.

    Postconditions: ``scientific_status`` is ``"unqualified"`` unless
    :func:`resolve_scientific_status` says otherwise; arrays are read-only.

    Raises:
        OSError: the receipt file cannot be read.
        AggregationIntegrityError: artifact missing, tampered or inconsistent.
        ValueError: the receipt is malformed.
    """
    run = load_verified_run(receipt_path, artifact_dir)
    receipt = run.receipt
    series = SpinalLoadSeries(
        time_s=run.time_s,
        levels=run.levels,
        observable=LoadObservable.JOINT_REACTION,
        force_n=run.joint_reaction_force_n,
        moment_nm=run.joint_reaction_moment_nm,
        provenance=_provenance(receipt),
    )
    return SpineStudyView(
        study_id=receipt.study_id,
        receipt_status=receipt.status,
        status_reasons=receipt.status_reasons,
        scientific_status=resolve_scientific_status(receipt),
        distribution_source=receipt.distribution_source,
        levels=run.levels,
        series=series,
    )


def _check_view(view: object) -> SpineStudyView:
    if not isinstance(view, SpineStudyView):
        raise TypeError(f"view must be a SpineStudyView, got {type(view).__name__}")
    return view


def status_banner(view: SpineStudyView) -> str:
    """Prominent status text: status word first, then source and reasons."""
    _check_view(view)
    word = "QUALIFIED" if view.scientific_status == "qualified" else "UNQUALIFIED"
    parts = [
        f"{word} (run verdict: {view.receipt_status})",
        f"distribution_source: {view.distribution_source}",
    ]
    if view.status_reasons:
        parts.append("reasons: " + "; ".join(view.status_reasons))
    return " | ".join(parts)


def view_heatmap(view: SpineStudyView, quantity: str) -> HeatmapData:
    """One load component of the study, carrying its status and source.

    Raises:
        TypeError: ``view`` is not a :class:`SpineStudyView`.
        ValueError: ``quantity`` not in :data:`LOAD_QUANTITIES`.
    """
    _check_view(view)
    if quantity not in LOAD_QUANTITIES:
        raise ValueError(f"quantity must be one of {LOAD_QUANTITIES}, got {quantity!r}")
    return load_heatmap(view.series, quantity, scientific_status=view.scientific_status)


def _heatmap_dict(view: SpineStudyView, quantity: str) -> dict[str, object]:
    heat = view_heatmap(view, quantity)
    return {
        "quantity": heat.quantity,
        "units": heat.units,
        "sign_convention": heat.sign_convention,
        "observable": heat.observable,
        "distribution_source": heat.distribution_source,
        "scientific_status": heat.scientific_status,
        "status_label": heat.status_label(),
        "time_s": heat.time_s.tolist(),
        "values": heat.values.tolist(),
    }


def view_to_dict(view: SpineStudyView) -> dict[str, object]:
    """JSON-serialisable form of a verified study for the web viewer.

    Reuses :func:`status_banner` and :func:`view_heatmap`; nothing is
    recomputed. Postconditions: ``scientific_status`` and
    ``distribution_source`` are always present, and the dict survives
    ``json.dumps(..., allow_nan=False)`` (arrays are finite by the
    verified-run contract).

    Raises:
        TypeError: ``view`` is not a :class:`SpineStudyView`.
    """
    _check_view(view)
    return {
        "study_id": view.study_id,
        "receipt_status": view.receipt_status,
        "status_reasons": list(view.status_reasons),
        "scientific_status": view.scientific_status,
        "distribution_source": view.distribution_source,
        "banner": status_banner(view),
        "levels": [level.name for level in view.levels],
        "quantities": list(LOAD_QUANTITIES),
        "heatmaps": {q: _heatmap_dict(view, q) for q in LOAD_QUANTITIES},
    }


def heatmap_figure(view: SpineStudyView, quantity: str) -> Figure:
    """Agg figure of :func:`view_heatmap`; its title states the status."""
    return plot_load_heatmap(view_heatmap(view, quantity))


def view_angle_stack(
    view: SpineStudyView,
    time_s: ArrayLike,
    poses: Sequence[SegmentalPose],
    term_id: str,
    *,
    handedness: Literal["right", "left"],
) -> StackData:
    """Per-level angle shares labelled with the study's source and status."""
    _check_view(view)
    return level_angle_stack(
        time_s,
        poses,
        term_id,
        handedness=handedness,
        distribution_source=view.distribution_source,
        scientific_status=view.scientific_status,
    )


def angle_stack_figure(stack: StackData) -> Figure:
    """Agg figure of a :class:`StackData` (thin pass-through to the plots)."""
    return plot_angle_stack(stack)
