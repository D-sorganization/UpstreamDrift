"""Framework-free plot model of the loop-nullspace report (NSX-42, #12574).

One model, two renderers: the PyQt tabs (NSX-43) and the React panel (NSX-44)
draw exactly what :class:`NSXPlotModel` contains and compute nothing.  There is
no PyQt, matplotlib or numpy here; ``to_dict()`` is plain JSON and
:meth:`NSXPlotModel.from_dict` is its exact inverse.

Cross-UI fixture
----------------
The checked-in golden is ``tests/fixtures/loop_nullspace/goldens/
plot_model_v1.json`` (``{"fixture": ..., "plot_model": ...}``).  Both UIs read
that one file, never a copy.  Regenerate it with::

    python3 -m tests.unit.loop_nullspace.plot_model_golden --write

Panels
------
``invariants``
    One series per invariant quantity, per frame (``SHOW_VALUE`` only).
``envelopes``
    Band first: lower/upper per frame, split into *segments*.  A segment has a
    ``display_class``: ``DETERMINED_SIGN`` (sign class positive, negative or
    zero over a bounded envelope), ``INDETERMINATE`` (shaded) or ``UNBOUNDED``
    (hatched; a bound that is not ``BOUNDED``).  The class and the sign class
    are data; how they are drawn is the renderer's business.  Infeasible frames
    are listed on the band and never inside a segment.
``hand_split``
    The six grip-frame split coordinates (``SPLIT_COLUMN_NAMES``) as bands,
    routed here by name.
``allocation``
    The hypothesis-conditioned representative, as an *overlay* that names its
    band.  It is never emitted alone: building a model whose overlay has no
    shown band raises ``ValueError``.
``diagnostics``
    Rank and condition of the decomposition, the admission verdict and scaled
    residuals, and the allocation KKT residuals.

Binding hypotheses
------------------
Each segment lists ``active_constraints`` (labels of the rows active at the
optimum of the bounding linear programs, i.e. the support of the optimal duals
by complementary slackness) and ``binding_hypotheses`` (the hypothesis ids of
those rows).  Both come from the report (NSX-22 ``binding_*`` fields); this
module only unions and sorts them.

Display decisions are obeyed, not remade
----------------------------------------
``display_decisions`` maps :func:`decision_key` strings to the
:class:`~.output_policy_integration.DisplayDecision` that
``display_decision`` returned for the quantity.  A quantity whose decision does
not admit what the panel would draw (a value, a band) appears only as a
:class:`WithheldItem` (or an unshown :class:`FrameValue`) with the decision's
reason and no number.  A quantity with no decision is withheld by name
(fail closed).  A decision key the report does not use raises ``ValueError``.
"""

from __future__ import annotations

import math
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from src.shared.python.loop_nullspace.grip_split import SPLIT_COLUMN_NAMES
from src.shared.python.loop_nullspace.output_policy_integration import (
    DisplayAction,
    DisplayDecision,
)

__all__ = [
    "PLOT_MODEL_SCHEMA_VERSION",
    "AdmissionSummary",
    "AllocationPanel",
    "Band",
    "DecisionView",
    "DiagnosticsPanel",
    "DisplayClass",
    "EnvelopesPanel",
    "FrameDiagnostic",
    "FrameStatus",
    "FrameValue",
    "HandSplitPanel",
    "InvariantSeries",
    "InvariantsPanel",
    "KktSummary",
    "NSXPlotModel",
    "Overlay",
    "Segment",
    "ValueSeries",
    "WithheldItem",
    "build_plot_model",
    "decision_key",
]

PLOT_MODEL_SCHEMA_VERSION = "1.0.0"
GRIP_FRAME = "grip frame"
PANELS = ("invariants", "envelopes", "hand_split", "allocation", "diagnostics")
_RESIDUAL_LEVELS = ("phi", "phi_dot", "phi_ddot")
_SHOW_BAND = (DisplayAction.SHOW_BAND, DisplayAction.SHOW_REPRESENTATIVE_ON_BAND)
_DETERMINED = frozenset({"positive", "negative", "zero"})
_REQUIRED_BOUND_KEYS = (
    "lower",
    "upper",
    "status_lower",
    "status_upper",
    "sign_class",
    "binding_lower",
    "binding_upper",
    "binding_hypotheses_lower",
    "binding_hypotheses_upper",
)


class DisplayClass(str, Enum):
    """How a band segment is to be read (renderer: determined / shaded / hatched)."""

    DETERMINED_SIGN = "DETERMINED_SIGN"
    INDETERMINATE = "INDETERMINATE"
    UNBOUNDED = "UNBOUNDED"


def decision_key(panel: str, name: str, frame_index: int | None = None) -> str:
    """The key under which a quantity's :class:`DisplayDecision` is supplied.

    ``invariants`` keys carry the frame (``invariants/<name>/<frame>``); every
    other panel's key is ``<panel>/<name>`` (``envelopes``, ``hand_split``,
    ``allocation`` with name ``torque``, ``diagnostics`` with the residual
    level ``phi``/``phi_dot``/``phi_ddot``).

    Raises:
        ValueError: Unknown panel, empty name, or a frame index given for a
            panel that takes none (or missing for ``invariants``).
    """
    if panel not in PANELS:
        raise ValueError(f"decision_key: unknown panel {panel!r}")
    if not name:
        raise ValueError("decision_key: name must be non-empty")
    if (frame_index is None) != (panel != "invariants"):
        raise ValueError("decision_key: frame_index is given for invariants only")
    return f"{panel}/{name}" + ("" if frame_index is None else f"/{int(frame_index)}")


# ------------------------------------------------------------------ values
@dataclass(frozen=True)
class DecisionView:
    """The parts of a :class:`DisplayDecision` a renderer needs (chips, badges)."""

    action: str
    reason: str
    label: str
    badges: tuple[str, ...]
    hypothesis_chips: tuple[str, ...]
    conditioning_chips: tuple[str, ...]
    preset: str | None

    def __post_init__(self) -> None:
        if self.action not in {a.value for a in DisplayAction}:
            raise ValueError(f"unknown display action {self.action!r}")

    @classmethod
    def of(cls, decision: DisplayDecision) -> DecisionView:
        """The view of ``decision``."""
        return cls(
            decision.action.value,
            decision.reason,
            decision.label,
            tuple(decision.badges),
            tuple(decision.hypothesis_chips),
            tuple(decision.conditioning_chips),
            decision.preset,
        )

    def admits(self, *actions: DisplayAction) -> bool:
        """Whether the decision's action is one of ``actions``."""
        return self.action in {a.value for a in actions}


@dataclass(frozen=True)
class WithheldItem:
    """A quantity that is not drawn: its name, the decision action and why."""

    quantity_id: str
    action: str
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("a withheld item needs a reason")


@dataclass(frozen=True)
class Segment:
    """Contiguous analysed frames sharing a display class and binding rows."""

    first_frame_index: int
    last_frame_index: int
    display_class: DisplayClass
    sign_class: str | None
    binding_hypotheses: tuple[str, ...]
    active_constraints: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.last_frame_index < self.first_frame_index:
            raise ValueError("a segment ends before it starts")


@dataclass(frozen=True)
class Band:
    """A band-first envelope series with its segments and decision."""

    quantity_id: str
    units: str
    frame_indices: tuple[int, ...]
    lower: tuple[float | None, ...]
    upper: tuple[float | None, ...]
    status_lower: tuple[str, ...]
    status_upper: tuple[str, ...]
    segments: tuple[Segment, ...]
    infeasible_frame_indices: tuple[int, ...]
    decision: DecisionView

    def __post_init__(self) -> None:
        n = len(self.frame_indices)
        sizes = {len(self.lower), len(self.upper), len(self.status_lower)}
        if sizes != {n} or len(self.status_upper) != n:
            raise ValueError(f"band {self.quantity_id!r}: series lengths differ")
        if not self.decision.admits(*_SHOW_BAND):
            raise ValueError(f"band {self.quantity_id!r}: its decision admits no band")


@dataclass(frozen=True)
class FrameStatus:
    """Per-frame feasibility of the envelope stage."""

    index: int
    feasible: bool
    chebyshev_status: str
    chebyshev_margin: float | None
    conflict_hypotheses: tuple[str, ...]
    rests_on_unverified: bool


@dataclass(frozen=True)
class FrameValue:
    """One invariant at one frame; ``values`` is empty unless ``shown``."""

    frame_index: int
    shown: bool
    values: tuple[float | None, ...]
    decision: DecisionView | None
    reason: str

    def __post_init__(self) -> None:
        if self.shown and not self.values:
            raise ValueError("a shown value needs numbers")
        if not self.shown and (self.values or not self.reason):
            raise ValueError("an unshown value carries a reason and no numbers")
        if self.shown and not (
            self.decision and self.decision.admits(DisplayAction.SHOW_VALUE)
        ):
            raise ValueError("a value is shown only under a SHOW_VALUE decision")


@dataclass(frozen=True)
class InvariantSeries:
    """One invariant quantity across the analysed frames."""

    name: str
    units: str
    frames: tuple[FrameValue, ...]


@dataclass(frozen=True)
class ValueSeries:
    """A per-frame series that is a single quantity (admission residuals)."""

    name: str
    units: str
    shown: bool
    values: tuple[float | None, ...]
    decision: DecisionView | None
    reason: str

    def __post_init__(self) -> None:
        if self.shown != bool(self.values):
            raise ValueError("a series is shown exactly when it carries numbers")
        if not self.shown and not self.reason:
            raise ValueError("an unshown series needs a reason")


@dataclass(frozen=True)
class InvariantsPanel:
    """Invariant series, or why there are none."""

    series: tuple[InvariantSeries, ...]
    unavailable_reason: str | None


@dataclass(frozen=True)
class EnvelopesPanel:
    """Torque / wrench / ground-load bands, withheld names and frame status."""

    bands: tuple[Band, ...]
    withheld: tuple[WithheldItem, ...]
    frames: tuple[FrameStatus, ...]
    unavailable_reason: str | None


@dataclass(frozen=True)
class HandSplitPanel:
    """Hand-split coordinates (``frame`` names the coordinate system)."""

    frame: str
    bands: tuple[Band, ...]
    withheld: tuple[WithheldItem, ...]
    unavailable_reason: str | None


@dataclass(frozen=True)
class Overlay:
    """A representative drawn ON ``band_id``; ``values`` align to its frames."""

    band_id: str
    units: str
    frame_indices: tuple[int, ...]
    values: tuple[float | None, ...]
    decision: DecisionView

    def __post_init__(self) -> None:
        if len(self.values) != len(self.frame_indices):
            raise ValueError(f"overlay {self.band_id!r}: series lengths differ")
        if not self.decision.admits(DisplayAction.SHOW_REPRESENTATIVE_ON_BAND):
            raise ValueError(
                f"overlay {self.band_id!r}: its decision is not "
                "SHOW_REPRESENTATIVE_ON_BAND"
            )


@dataclass(frozen=True)
class AllocationPanel:
    """The representative allocation as overlays on bands, never alone."""

    status: str | None
    reason: str | None
    overlays: tuple[Overlay, ...]
    withheld: tuple[WithheldItem, ...]
    excluded_frame_indices: tuple[int, ...]
    objective_total: float | None
    non_claim: str | None
    unavailable_reason: str | None


@dataclass(frozen=True)
class FrameDiagnostic:
    """Rank and condition of one frame's decomposition."""

    index: int
    rank_A: int
    nullity: int
    rank_J_c: int
    condition_number: float | None


@dataclass(frozen=True)
class AdmissionSummary:
    """The admission verdict (the residual series are in ``residuals``)."""

    verdict: str
    reasons: tuple[str, ...]
    failing_frames: tuple[int, ...]
    derivative_method: str
    noise_floor: float | None
    position_evaluated: bool


@dataclass(frozen=True)
class KktSummary:
    """Scaled Clarabel KKT residuals of the allocation."""

    primal: float | None
    dual: float | None
    complementarity: float | None


@dataclass(frozen=True)
class DiagnosticsPanel:
    """Rank, condition, admission residuals and allocation optimality."""

    frames: tuple[FrameDiagnostic, ...]
    admission: AdmissionSummary | None
    residuals: tuple[ValueSeries, ...]
    kkt: KktSummary | None


@dataclass(frozen=True)
class NSXPlotModel:
    """The whole model; ``to_dict`` is the cross-UI JSON."""

    schema_version: str
    status: str
    reason: str | None
    preset: str
    registry_version: str
    engine_name: str
    frame_indices: tuple[int, ...]
    non_claims: tuple[str, ...]
    invariants: InvariantsPanel
    envelopes: EnvelopesPanel
    hand_split: HandSplitPanel
    allocation: AllocationPanel
    diagnostics: DiagnosticsPanel

    def __post_init__(self) -> None:
        bands = {b.quantity_id: b for b in self._bands()}
        for overlay in self.allocation.overlays:
            band = bands.get(overlay.band_id)
            if band is None:
                raise ValueError(
                    f"representative {overlay.band_id!r} has no band: a "
                    "representative is never shown alone"
                )
            if band.frame_indices != overlay.frame_indices:
                raise ValueError(
                    f"representative {overlay.band_id!r} does not align with its band"
                )

    def _bands(self) -> tuple[Band, ...]:
        return (*self.envelopes.bands, *self.hand_split.bands)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form (tuples become lists; no NaN)."""
        return _plain(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NSXPlotModel:
        """Inverse of :meth:`to_dict`.

        Raises:
            TypeError: ``data`` is not a mapping.
            ValueError: Unknown or missing keys, or a value of the wrong kind.
        """
        if not isinstance(data, Mapping):
            raise TypeError("from_dict expects a mapping")
        built = _build(cls, data)
        assert isinstance(built, cls)
        return built


# ------------------------------------------------- generic (de)serialisation
def _plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple | list):
        return [_plain(x) for x in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj


def _build(tp: Any, data: Any) -> Any:
    """Rebuild a value of type ``tp`` from its plain form (strict)."""
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        inner = [a for a in typing.get_args(tp) if a is not type(None)]
        return None if data is None else _build(inner[0], data)
    if origin is tuple:
        if not isinstance(data, Sequence) or isinstance(data, str):
            raise ValueError(f"expected a list for {tp}, got {data!r}")
        return tuple(_build(typing.get_args(tp)[0], x) for x in data)
    if isinstance(tp, type) and is_dataclass(tp):
        return _build_dataclass(tp, data)
    if isinstance(tp, type) and issubclass(tp, Enum):
        return tp(data)
    return _build_scalar(tp, data)


def _build_scalar(tp: Any, data: Any) -> Any:
    if tp is float:
        if isinstance(data, bool) or not isinstance(data, int | float):
            raise ValueError(f"expected a number, got {data!r}")
        return float(data)
    if tp is int:
        if isinstance(data, bool) or not isinstance(data, int):
            raise ValueError(f"expected an integer, got {data!r}")
        return int(data)
    if tp not in (str, bool) or not isinstance(data, tp):
        raise ValueError(f"expected {tp}, got {data!r}")
    return data


def _build_dataclass(cls: type, data: Any) -> Any:
    if not isinstance(data, Mapping):
        raise ValueError(f"{cls.__name__} expects a mapping, got {data!r}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    unknown = sorted(set(data) - set(names))
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {unknown}")
    missing = [n for n in names if n not in data]
    if missing:
        raise ValueError(f"{cls.__name__}: missing keys {missing}")
    return cls(**{n: _build(hints[n], data[n]) for n in names})


# ----------------------------------------------------------------- builders
def _num(value: Any) -> float | None:
    """A finite float, else ``None`` (JSON has no NaN)."""
    if value is None:
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def _flat(value: Any) -> list[float | None]:
    if isinstance(value, list | tuple):
        return [x for item in value for x in _flat(item)]
    return [_num(value)]


class _Decisions:
    """The supplied decisions, remembering which ones the report used."""

    def __init__(self, decisions: Mapping[str, DisplayDecision]) -> None:
        self._decisions = decisions
        self._used: set[str] = set()

    def get(self, key: str) -> DisplayDecision | None:
        self._used.add(key)
        return self._decisions.get(key)

    def check_all_used(self) -> None:
        extra = sorted(set(self._decisions) - self._used)
        if extra:
            raise ValueError(f"unknown display decision keys: {extra}")


def _withheld(name: str, key: str, decision: DisplayDecision | None) -> WithheldItem:
    if decision is None:
        return WithheldItem(
            name, DisplayAction.WITHHOLD.value, f"no display decision for {key}"
        )
    return WithheldItem(name, decision.action.value, decision.reason)


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(
            f"report {where} lacks {key!r}; it predates report schema 1.1.0"
        )
    return mapping[key]


def _class_of(
    status_lo: str, status_hi: str, sign: str | None
) -> tuple[DisplayClass, str | None]:
    if status_lo != "BOUNDED" or status_hi != "BOUNDED":
        return DisplayClass.UNBOUNDED, sign
    if sign in _DETERMINED:
        return DisplayClass.DETERMINED_SIGN, sign
    return DisplayClass.INDETERMINATE, sign


def _segments(
    frame_indices: Sequence[int], feasible: Sequence[bool], bounds: Mapping[str, Any]
) -> tuple[Segment, ...]:
    """Merge adjacent feasible frames with equal class, sign and binding rows."""
    out: list[Segment] = []
    current: tuple[Any, ...] | None = None
    for k, frame in enumerate(frame_indices):
        if not feasible[k]:
            current = None
            continue
        cls, sign = _class_of(
            bounds["status_lower"][k],
            bounds["status_upper"][k],
            bounds["sign_class"][k],
        )
        rows = sorted({*bounds["binding_lower"][k], *bounds["binding_upper"][k]})
        hyps = sorted(
            {
                *bounds["binding_hypotheses_lower"][k],
                *bounds["binding_hypotheses_upper"][k],
            }
        )
        key = (cls, sign, tuple(hyps), tuple(rows))
        if key == current:
            last = out[-1]
            out[-1] = Segment(last.first_frame_index, frame, *key)
        else:
            out.append(Segment(frame, frame, *key))
            current = key
    return tuple(out)


def _band(
    name: str,
    units: str,
    frames: Sequence[Mapping[str, Any]],
    bounds: Mapping[str, Any],
    decision: DisplayDecision,
) -> Band:
    for key in _REQUIRED_BOUND_KEYS:
        _require(bounds, key, f"bounds of {name!r}")
    idx = tuple(int(f["index"]) for f in frames)
    feasible = [bool(f["feasible"]) for f in frames]
    return Band(
        quantity_id=name,
        units=units,
        frame_indices=idx,
        lower=tuple(_num(v) for v in bounds["lower"]),
        upper=tuple(_num(v) for v in bounds["upper"]),
        status_lower=tuple(bounds["status_lower"]),
        status_upper=tuple(bounds["status_upper"]),
        segments=_segments(idx, feasible, bounds),
        infeasible_frame_indices=tuple(
            i for i, ok in zip(idx, feasible, strict=True) if not ok
        ),
        decision=DecisionView.of(decision),
    )


def _route(name: str) -> str:
    return "hand_split" if name in SPLIT_COLUMN_NAMES else "envelopes"


def _bands(
    env: Mapping[str, Any], decs: _Decisions
) -> tuple[dict[str, list[Band]], dict[str, list[WithheldItem]]]:
    bands: dict[str, list[Band]] = {"envelopes": [], "hand_split": []}
    held: dict[str, list[WithheldItem]] = {"envelopes": [], "hand_split": []}
    for name in env["quantities"]:
        panel = _route(name)
        key = decision_key(panel, name)
        decision = decs.get(key)
        if decision is None or decision.action not in _SHOW_BAND:
            held[panel].append(_withheld(name, key, decision))
            continue
        bounds = _require(env["bounds"], name, "envelopes.bounds")
        units = str(bounds["units"])
        bands[panel].append(_band(name, units, env["frames"], bounds, decision))
    return bands, held


def _frame_statuses(env: Mapping[str, Any]) -> tuple[FrameStatus, ...]:
    return tuple(
        FrameStatus(
            index=int(f["index"]),
            feasible=bool(f["feasible"]),
            chebyshev_status=str(f["chebyshev_status"]),
            chebyshev_margin=_num(f["chebyshev_margin"]),
            conflict_hypotheses=tuple(f["conflict_hypotheses"]),
            rests_on_unverified=bool(f["rests_on_unverified"]),
        )
        for f in env["frames"]
    )


def _panels_from_envelopes(
    report: Mapping[str, Any], decs: _Decisions
) -> tuple[EnvelopesPanel, HandSplitPanel]:
    env = report["envelopes"]
    if env is None:
        why = f"envelopes not computed (report status {report['status']!r})"
        return (
            EnvelopesPanel((), (), (), why),
            HandSplitPanel(GRIP_FRAME, (), (), why),
        )
    bands, held = _bands(env, decs)
    return (
        EnvelopesPanel(
            tuple(bands["envelopes"]),
            tuple(held["envelopes"]),
            _frame_statuses(env),
            None,
        ),
        HandSplitPanel(
            GRIP_FRAME,
            tuple(bands["hand_split"]),
            tuple(held["hand_split"]),
            None
            if bands["hand_split"] or held["hand_split"]
            else "no hand-split coordinate in the report",
        ),
    )


def _frame_value(
    index: int, quantity: Mapping[str, Any], decision: DisplayDecision | None, key: str
) -> FrameValue:
    if decision is None:
        return FrameValue(index, False, (), None, f"no display decision for {key}")
    view = DecisionView.of(decision)
    values = quantity["values"]
    if decision.action is DisplayAction.SHOW_VALUE and values is not None:
        return FrameValue(index, True, tuple(_flat(values)), view, decision.reason)
    reason = decision.reason
    if decision.action is DisplayAction.SHOW_VALUE:
        reason = "the report carries no values for this quantity"
    elif decision.action is not DisplayAction.WITHHOLD and not decision.reason:
        reason = decision.action.value
    return FrameValue(index, False, (), view, reason)


def _invariants_panel(report: Mapping[str, Any], decs: _Decisions) -> InvariantsPanel:
    inv = report["invariants"]
    if inv is None:
        why = f"invariants not computed (report status {report['status']!r})"
        return InvariantsPanel((), why)
    order: dict[str, list[FrameValue]] = {}
    units: dict[str, str] = {}
    for frame in inv["frames"]:
        index = int(frame["index"])
        for name, quantity in frame["quantities"].items():
            key = decision_key("invariants", name, index)
            value = _frame_value(index, quantity, decs.get(key), key)
            order.setdefault(name, []).append(value)
            units.setdefault(name, str(quantity["units"]))
    return InvariantsPanel(
        tuple(InvariantSeries(n, units[n], tuple(v)) for n, v in order.items()), None
    )


def _aligned(
    band: Band, alloc_frames: Sequence[int], torque: Sequence[Sequence[Any]], col: int
) -> tuple[float | None, ...]:
    where = {int(i): pos for pos, i in enumerate(alloc_frames)}
    return tuple(
        _num(torque[where[i]][col]) if i in where else None for i in band.frame_indices
    )


def _overlays(
    alloc: Mapping[str, Any],
    decision: DisplayDecision,
    bands: Mapping[str, Band],
) -> tuple[list[Overlay], list[WithheldItem]]:
    torque = alloc["torque"]["values"]
    overlays: list[Overlay] = []
    held: list[WithheldItem] = []
    for col in range(len(torque[0]) if torque else 0):
        name = f"tau_{col + 1}"
        band = bands.get(name)
        if band is None:
            held.append(
                WithheldItem(
                    name,
                    DisplayAction.WITHHOLD.value,
                    f"band for {name} is not shown; a representative is never "
                    "shown alone",
                )
            )
            continue
        values = _aligned(band, alloc["frame_indices"], torque, col)
        overlays.append(
            Overlay(
                name, band.units, band.frame_indices, values, DecisionView.of(decision)
            )
        )
    return overlays, held


def _allocation_panel(
    report: Mapping[str, Any], decs: _Decisions, bands: Mapping[str, Band]
) -> AllocationPanel:
    alloc = report["allocation"]
    if alloc is None:
        return AllocationPanel(
            None, None, (), (), (), None, None, "allocation not requested"
        )
    base: dict[str, Any] = {
        "status": str(alloc["status"]),
        "reason": alloc.get("reason"),
        "excluded_frame_indices": tuple(
            int(e["index"]) for e in alloc["excluded_frames"]
        ),
        "objective_total": _num(alloc.get("objective_total")),
        "non_claim": alloc.get("non_claim"),
        "unavailable_reason": None,
    }
    if alloc.get("torque") is None:
        base["unavailable_reason"] = alloc.get("reason") or "no allocation torque"
        return AllocationPanel(overlays=(), withheld=(), **base)
    key = decision_key("allocation", "torque")
    decision = decs.get(key)
    if (
        decision is None
        or decision.action is not DisplayAction.SHOW_REPRESENTATIVE_ON_BAND
    ):
        item = _withheld("torque", key, decision)
        if decision is not None:
            item = WithheldItem(
                "torque",
                decision.action.value,
                f"{decision.reason} (action {decision.action.value} does not "
                "admit a representative)",
            )
        return AllocationPanel(overlays=(), withheld=(item,), **base)
    overlays, held = _overlays(alloc, decision, bands)
    return AllocationPanel(overlays=tuple(overlays), withheld=tuple(held), **base)


def _admission(adm: Mapping[str, Any]) -> AdmissionSummary:
    return AdmissionSummary(
        verdict=str(adm["verdict"]),
        reasons=tuple(adm["reasons"]),
        failing_frames=tuple(int(i) for i in adm["failing_frames"]),
        derivative_method=str(adm["derivative_method"]),
        noise_floor=_num(adm["noise_floor"]),
        position_evaluated=bool(adm["position_evaluated"]),
    )


def _residual(level: str, quantity: Mapping[str, Any], decs: _Decisions) -> ValueSeries:
    key = decision_key("diagnostics", level)
    decision = decs.get(key)
    units = str(quantity["units"])
    if decision is None:
        return ValueSeries(
            level, units, False, (), None, f"no display decision for {key}"
        )
    view = DecisionView.of(decision)
    values = quantity["values"]
    if decision.action is DisplayAction.SHOW_VALUE and values is not None:
        return ValueSeries(
            level, units, True, tuple(_flat(values)), view, decision.reason
        )
    return ValueSeries(level, units, False, (), view, decision.reason)


def _diagnostics_panel(report: Mapping[str, Any], decs: _Decisions) -> DiagnosticsPanel:
    frames = tuple(
        FrameDiagnostic(
            int(d["index"]),
            int(d["rank_A"]),
            int(d["nullity"]),
            int(d["rank_J_c"]),
            _num(d["condition_number"]),
        )
        for d in _require(report, "frame_diagnostics", "")
    )
    adm = report["admission"]
    residuals = (
        ()
        if adm is None
        else tuple(_residual(lv, adm["residuals"][lv], decs) for lv in _RESIDUAL_LEVELS)
    )
    alloc = report["allocation"]
    kkt = alloc.get("kkt_residuals") if alloc else None
    return DiagnosticsPanel(
        frames=frames,
        admission=None if adm is None else _admission(adm),
        residuals=residuals,
        kkt=None
        if kkt is None
        else KktSummary(
            _num(kkt["primal"]), _num(kkt["dual"]), _num(kkt["complementarity"])
        ),
    )


def _check_inputs(report: object, decisions: object) -> None:
    if not isinstance(report, Mapping):
        raise TypeError("report must be the mapping returned by analyze()")
    if not isinstance(decisions, Mapping):
        raise TypeError("display_decisions must be a mapping of DisplayDecision")
    for key, value in decisions.items():
        if not isinstance(key, str) or not isinstance(value, DisplayDecision):
            raise TypeError(f"display_decisions[{key!r}] must be a DisplayDecision")
    for key in ("status", "preset", "registry_version", "engine", "non_claims"):
        if key not in report:
            raise ValueError(f"report lacks {key!r}")


def build_plot_model(
    report: Mapping[str, Any], display_decisions: Mapping[str, DisplayDecision]
) -> NSXPlotModel:
    """Build the plot model from an ``analyze()`` report and its decisions.

    Args:
        report: The report mapping (schema 1.1.0 or later).
        display_decisions: ``decision_key`` -> :class:`DisplayDecision`, as
            produced by ``display_decision`` for each quantity of the report
            (see :func:`~.plot_model_decisions.report_decisions`).

    Returns:
        A frozen :class:`NSXPlotModel`.  Postconditions: every band and overlay
        has a decision that admits it; a representative overlay always has a
        shown band over the same frames; a quantity without an admitting
        decision carries no number; a non-``complete`` report gives empty
        panels that say why.

    Raises:
        TypeError: ``report`` or ``display_decisions`` of the wrong kind.
        ValueError: A report without the sign/binding fields, or a decision
            key the report does not use.
    """
    _check_inputs(report, display_decisions)
    decs = _Decisions(display_decisions)
    envelopes, hand_split = _panels_from_envelopes(report, decs)
    shown = {b.quantity_id: b for b in (*envelopes.bands, *hand_split.bands)}
    model = NSXPlotModel(
        schema_version=PLOT_MODEL_SCHEMA_VERSION,
        status=str(report["status"]),
        reason=report.get("reason"),
        preset=str(report["preset"]),
        registry_version=str(report["registry_version"]),
        engine_name=str(report["engine"]["name"]),
        frame_indices=tuple(
            int(i)
            for i in report.get("trajectory", {}).get("analyzed_frame_indices", ())
        ),
        non_claims=tuple(report["non_claims"]),
        invariants=_invariants_panel(report, decs),
        envelopes=envelopes,
        hand_split=hand_split,
        allocation=_allocation_panel(report, decs, shown),
        diagnostics=_diagnostics_panel(report, decs),
    )
    decs.check_all_used()
    return model
