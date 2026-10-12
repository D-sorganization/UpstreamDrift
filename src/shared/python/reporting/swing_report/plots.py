"""Plot registry for the swing report contract (#12326, epic #11419).

A ``PlotSpec`` is the static, data-driven declaration of a plot: which quantity
it draws and how. ``resolve_descriptor`` turns a spec plus a validated report
into the contract ``PlotDescriptor``, deriving ``render``/units/frame/engines
from the report so a descriptor can never disagree with its series (I7) and a
withheld or absent quantity is always ``unavailable`` with a reason (I8).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import logging
import re
from types import MappingProxyType

from .contract import PlotDescriptor, Quantity, SwingReport
from .policy import (
    NOT_VALIDATED_LABEL,
    STATUS_WITHHELD,
    QuantityClass,
    classify,
)

logger = logging.getLogger(__name__)

PLOT_KINDS = ("time_series", "bar", "sequence_timeline", "scatter")
_PLOT_ID = re.compile(r"^[a-z][a-z0-9_]*$")
UNCERTAINTY_FIELDS = ("lower", "upper")


@dataclass(frozen=True)
class PlotSpec:
    """Declaration of one plot; ``quantity_id`` is its required input."""

    plot_id: str
    title: str
    quantity_id: str
    kind: str

    def __post_init__(self) -> None:
        if not _PLOT_ID.match(self.plot_id):
            raise ValueError(f"invalid plot_id {self.plot_id!r}")
        if not self.title:
            raise ValueError("title must be non-empty")
        if not self.quantity_id:
            raise ValueError("quantity_id must be non-empty")
        if self.kind not in PLOT_KINDS:
            raise ValueError(f"kind must be one of {PLOT_KINDS}, got {self.kind!r}")


def make_registry(specs: Iterable[PlotSpec]) -> Mapping[str, PlotSpec]:
    """Build a read-only, insertion-ordered registry keyed by ``plot_id``.

    Raises:
        ValueError: if ``specs`` is empty or contains a duplicate ``plot_id``.
    """
    registry: dict[str, PlotSpec] = {}
    for spec in specs:
        if spec.plot_id in registry:
            raise ValueError(f"duplicate plot_id {spec.plot_id!r}")
        registry[spec.plot_id] = spec
    if not registry:
        raise ValueError("registry needs at least one PlotSpec")
    return MappingProxyType(registry)


PLOT_REGISTRY: Mapping[str, PlotSpec] = make_registry(
    [
        PlotSpec(
            "clubhead_speed_ts", "Clubhead Speed", "clubhead_speed", "time_series"
        ),
        PlotSpec("clubhead_path", "Clubhead Path", "clubhead_position", "time_series"),
        PlotSpec(
            "kinematic_sequence_timeline",
            "Kinematic Sequence",
            "kinematic_sequence",
            "sequence_timeline",
        ),
        PlotSpec("face_angle_ts", "Face Angle", "face_angle", "time_series"),
        PlotSpec("joint_load_ts", "Joint Loads", "joint_load", "time_series"),
        PlotSpec("impact_state_plot", "Impact State", "impact_state", "bar"),
        PlotSpec("ball_flight_plot", "Ball Flight", "ball_flight", "scatter"),
        PlotSpec("joint_angle_ts", "Joint Angles", "joint_angle", "time_series"),
    ]
)


def _unavailable(
    spec: PlotSpec, engines: tuple[str, ...], reason: str
) -> PlotDescriptor:
    return PlotDescriptor(
        plot_id=spec.plot_id,
        title=spec.title,
        quantity_id=spec.quantity_id,
        kind=spec.kind,
        engines=engines,
        render="unavailable",
        unavailable_reason=reason,
        selectable=True,
        default_selected=False,
    )


def _unavailable_reason(spec: PlotSpec, quantity: Quantity | None) -> str | None:
    """Why ``spec`` cannot render values, or ``None`` if it can."""
    qid = spec.quantity_id
    if classify(qid) is QuantityClass.WITHHELD:
        reason = quantity.reason if quantity and quantity.reason else None
        return reason or f"{NOT_VALIDATED_LABEL} (#10382)"
    if quantity is None:
        return f"required quantity {qid} is not in the report (no comparable data)"
    if quantity.status == STATUS_WITHHELD or not quantity.series:
        return quantity.reason or f"quantity {qid} has no series"
    first = quantity.series[0]
    if any((s.units, s.frame) != (first.units, first.frame) for s in quantity.series):
        return f"series of {qid} differ in units or frame"
    return None


def resolve_descriptor(spec: PlotSpec, report: SwingReport) -> PlotDescriptor:
    """Resolve ``spec`` against ``report`` into a contract ``PlotDescriptor``.

    Postconditions: the descriptor is ``values`` only for an advertised
    quantity with consistent series, and then its engines/units/frame equal
    those series (I7); otherwise it is ``unavailable`` with a non-empty reason
    and carries no units or uncertainty (I8). Never raises on missing data.
    """
    quantity = next(
        (q for q in report.quantities if q.quantity_id == spec.quantity_id), None
    )
    reason = _unavailable_reason(spec, quantity)
    if reason is not None or quantity is None:
        run_engines = tuple(dict.fromkeys(r.engine for r in report.runs))
        return _unavailable(spec, run_engines, reason or "unavailable")
    first = quantity.series[0]
    return PlotDescriptor(
        plot_id=spec.plot_id,
        title=spec.title,
        quantity_id=spec.quantity_id,
        kind=spec.kind,
        engines=tuple(s.engine for s in quantity.series),
        render="values",
        units=first.units,
        frame=first.frame,
        uncertainty_fields=UNCERTAINTY_FIELDS,
        selectable=True,
        default_selected=True,
    )
