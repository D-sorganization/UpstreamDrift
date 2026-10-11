"""Validator for the swing report contract v1 (#12324, epic #11419).

``validate_swing_report`` first applies ``schemas/swing-report-v1.json`` and
then the cross-reference and output-policy invariants the schema cannot
express. It is fail-closed: every violation is collected and reported with a
stable code, and no ``SwingReport`` is returned unless there are none.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
import json
import math
from pathlib import Path
from typing import Any

import jsonschema

from src.shared.python.reporting.swing_report.contract import (
    SwingReport,
    report_from_dict,
)
from src.shared.python.reporting.swing_report.policy import (
    STATUS_ADVERTISED,
    STATUS_VALIDATED,
    STATUS_WITHHELD,
    UNCERTAINTY_BUDGET_SOURCE,
    UNCERTAINTY_KIND,
    POLICY,
    WITHHELD,
    QuantityClass,
    classify,
)

SCHEMA_PATH = Path(__file__).resolve().parents[5] / "schemas" / "swing-report-v1.json"
_SERIES_REQUIRED = ("engine", "run_id", "units", "frame")
_VALUE_KEYS = ("series", "values", "value", "uncertainty", "lower", "upper", "time_s")


@dataclass(frozen=True)
class Violation:
    """One contract violation: a stable ``code``, a ``path`` and a message."""

    code: str
    path: str
    message: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.path}: {self.message}"


class SwingReportContractError(ValueError):
    """Raised when a document violates the swing report contract."""

    def __init__(self, violations: Iterable[Violation]) -> None:
        self.violations: tuple[Violation, ...] = tuple(violations)
        lines = "\n".join(f"  {v}" for v in self.violations)
        super().__init__(
            f"swing report violates contract ({len(self.violations)}):\n{lines}"
        )


@cache
def load_schema() -> dict[str, Any]:
    """Load the v1 schema once (never mutate the returned dict)."""
    with SCHEMA_PATH.open(encoding="utf-8") as handle:
        schema: dict[str, Any] = json.load(handle)
    return schema


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _finite(x: Any) -> bool:
    return _is_number(x) and math.isfinite(x)


def _items(doc: Mapping[str, Any], key: str) -> list[tuple[int, dict[str, Any]]]:
    """Indexed dict elements of ``doc[key]``; tolerant of malformed shapes."""
    raw = doc.get(key)
    if not isinstance(raw, list):
        return []
    return [(i, x) for i, x in enumerate(raw) if isinstance(x, dict)]


def _has_number(node: Any) -> bool:
    if _is_number(node):
        return True
    if isinstance(node, dict):
        return any(_has_number(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return any(_has_number(v) for v in node)
    return False


def _leaky_keys(node: Any, path: str) -> Iterable[str]:
    """Paths of withheld-named keys holding numeric payload."""
    if isinstance(node, dict):
        for key, val in node.items():
            here = f"{path}.{key}"
            if key in WITHHELD and _has_number(val):
                yield here
            else:
                yield from _leaky_keys(val, here)
    elif isinstance(node, list):
        for i, val in enumerate(node):
            yield from _leaky_keys(val, f"{path}[{i}]")


class _Checker:
    """Collects invariant violations for one document."""

    def __init__(self, doc: Mapping[str, Any]) -> None:
        self.doc = doc
        self.out: list[Violation] = []
        self.runs = {r.get("run_id"): r for _, r in _items(doc, "runs")}
        self.quantities = {
            q.get("quantity_id"): q for _, q in _items(doc, "quantities")
        }
        self.plots = {p.get("plot_id"): p for _, p in _items(doc, "plots")}

    def add(self, code: str, path: str, message: str) -> None:
        self.out.append(Violation(code, path, message))

    def run(self) -> list[Violation]:
        self.provenance()
        self.unique_ids()
        for i, q in _items(self.doc, "quantities"):
            self.quantity(q, f"quantities[{i}]")
        for i, p in _items(self.doc, "plots"):
            self.plot(p, f"plots[{i}]")
        self.sections()
        for path in _leaky_keys(self.doc.get("extensions"), "extensions"):
            self.add("I9_WITHHELD_LEAK", path, "withheld quantity carries numbers")
        return self.out

    def provenance(self) -> None:
        prov = self.doc.get("provenance")
        cap = prov.get("capture") if isinstance(prov, dict) else None
        if (
            isinstance(cap, dict)
            and cap.get("visibility") == "private"
            and "capture_sha256" in cap
        ):
            self.add(
                "PRIVATE_CAPTURE_HASH",
                "provenance.capture.capture_sha256",
                "private captures must omit capture_sha256",
            )

    def unique_ids(self) -> None:
        for key, field in (
            ("runs", "run_id"),
            ("quantities", "quantity_id"),
            ("plots", "plot_id"),
            ("sections", "section_id"),
        ):
            seen: set[Any] = set()
            for i, item in _items(self.doc, key):
                ident = item.get(field)
                if ident in seen:
                    self.add("I10_DUPLICATE_ID", f"{key}[{i}].{field}", f"{ident!r}")
                seen.add(ident)

    # -- quantities -------------------------------------------------------
    def quantity(self, q: Mapping[str, Any], path: str) -> None:
        qid, status = q.get("quantity_id"), q.get("status")
        klass = classify(qid) if isinstance(qid, str) else QuantityClass.WITHHELD
        if not self.known(qid):
            self.add(
                "UNKNOWN_QUANTITY", f"{path}.quantity_id", f"{qid!r} not in catalog"
            )
        if status == STATUS_VALIDATED:
            self.add("STATUS_VALIDATED", f"{path}.status", "not allowed in v1")
        elif status in (STATUS_ADVERTISED, STATUS_WITHHELD) and status != klass.value:
            if self.known(qid):
                self.add(
                    "I6_POLICY_MISMATCH",
                    f"{path}.status",
                    f"{qid} is {klass.value} by policy, document says {status}",
                )
        if klass is QuantityClass.WITHHELD:
            leaked = [k for k in _VALUE_KEYS if k in q]
            if leaked:
                self.add("WITHHELD_HAS_VALUES", path, f"withheld quantity has {leaked}")
            return
        self.advertised(q, path)

    @staticmethod
    def known(qid: Any) -> bool:
        return isinstance(qid, str) and qid in POLICY

    def advertised(self, q: Mapping[str, Any], path: str) -> None:
        unc = q.get("uncertainty")
        if q.get("status") == STATUS_ADVERTISED and not isinstance(unc, dict):
            self.add("ADVERTISED_NO_UNCERTAINTY", path, "uncertainty is required")
        series = [
            (i, s) for i, s in enumerate(q.get("series") or []) if isinstance(s, dict)
        ]
        for i, s in series:
            self.series(s, f"{path}.series[{i}]")
        if isinstance(unc, dict):
            self.uncertainty(unc, [s for _, s in series], f"{path}.uncertainty")

    def series(self, s: Mapping[str, Any], path: str) -> None:
        for key in _SERIES_REQUIRED:
            if not isinstance(s.get(key), str) or not s.get(key):
                self.add("I1_SERIES_FIELDS", f"{path}.{key}", "required, non-empty")
        run = self.runs.get(s.get("run_id"))
        if run is None:
            self.add("I3_RUN_REF", f"{path}.run_id", f"{s.get('run_id')!r} not in runs")
        elif run.get("engine") != s.get("engine"):
            self.add(
                "I3_ENGINE_MISMATCH",
                f"{path}.engine",
                f"series engine {s.get('engine')!r} != run engine {run.get('engine')!r}",
            )
        if s.get("channel") in WITHHELD:
            self.add("I9_WITHHELD_LEAK", f"{path}.channel", f"{s['channel']} withheld")
        time, values = s.get("time_s"), s.get("values")
        if not isinstance(time, list) or not isinstance(values, list):
            return
        if len(time) != len(values):
            self.add(
                "I2_SERIES_LENGTH", path, f"{len(time)} times vs {len(values)} values"
            )
        bad = [x for x in time if not _finite(x)]
        bad += [x for x in values if x is not None and not _finite(x)]
        if bad:
            self.add("I2_NON_FINITE", path, "time_s/values must be finite or null")
        elif any(b <= a for a, b in zip(time, time[1:], strict=False)):
            self.add("I2_TIME_ORDER", f"{path}.time_s", "must be strictly increasing")

    def uncertainty(
        self, unc: Mapping[str, Any], series: list[Mapping[str, Any]], path: str
    ) -> None:
        if (
            unc.get("kind") != UNCERTAINTY_KIND
            or unc.get("budget_source") != UNCERTAINTY_BUDGET_SOURCE
        ):
            self.add(
                "I4_UNCERTAINTY_BASIS",
                path,
                f"need {UNCERTAINTY_KIND}/{UNCERTAINTY_BUDGET_SOURCE} in v1",
            )
        lower, upper = unc.get("lower"), unc.get("upper")
        if not isinstance(lower, list) or not isinstance(upper, list):
            return
        if not all(_finite(x) for x in (*lower, *upper)):
            self.add("I2_NON_FINITE", path, "bounds must be finite")
        for s in series:
            values = s.get("values")
            if isinstance(values, list) and (
                len(lower) != len(values) or len(upper) != len(values)
            ):
                self.add(
                    "I4_UNCERTAINTY_LENGTH",
                    path,
                    f"lower/upper length must equal values length ({len(values)})",
                )
            if s.get("units") != unc.get("units"):
                self.add(
                    "I11_UNCERTAINTY_UNITS",
                    f"{path}.units",
                    f"{unc.get('units')!r} != series units {s.get('units')!r}",
                )
        if any(
            _finite(lo) and _finite(hi) and lo > hi
            for lo, hi in zip(lower, upper, strict=False)
        ):
            self.add("I11_BOUNDS_ORDER", path, "lower must not exceed upper")

    # -- plots / sections -------------------------------------------------
    def plot(self, p: Mapping[str, Any], path: str) -> None:
        qid, render = p.get("quantity_id"), p.get("render")
        withheld_class = (
            not isinstance(qid, str) or classify(qid) is QuantityClass.WITHHELD
        )
        if withheld_class and render != "unavailable":
            self.add(
                "PLOT_WITHHELD_RENDER", f"{path}.render", f"{qid} must be unavailable"
            )
        if render == "unavailable" and not p.get("unavailable_reason"):
            self.add("I8_UNAVAILABLE_REASON", path, "unavailable plot needs a reason")
        quantity = self.quantities.get(qid)
        if quantity is None:
            self.add(
                "I7_PLOT_QUANTITY", f"{path}.quantity_id", f"{qid!r} not in quantities"
            )
            return
        if render != "values" or withheld_class:
            return
        by_engine = {
            s.get("engine"): s
            for s in quantity.get("series") or []
            if isinstance(s, dict)
        }
        for engine in p.get("engines") or []:
            series = by_engine.get(engine)
            if series is None:
                self.add(
                    "I7_PLOT_ENGINES", f"{path}.engines", f"{engine} has no series"
                )
            elif (series.get("units"), series.get("frame")) != (
                p.get("units"),
                p.get("frame"),
            ):
                self.add(
                    "I7_PLOT_UNITS", path, f"units/frame differ from {engine} series"
                )

    def sections(self) -> None:
        custom = self.doc.get("scope") == "custom"
        for i, sec in _items(self.doc, "sections"):
            for j, pid in enumerate(sec.get("plot_ids") or []):
                here = f"sections[{i}].plot_ids[{j}]"
                plot = self.plots.get(pid)
                if plot is None:
                    self.add("I10_SECTION_REF", here, f"{pid!r} not in plots")
                elif custom and plot.get("selectable", True) is False:
                    self.add("I12_CUSTOM_SELECTABLE", here, f"{pid} is not selectable")


def _schema_violations(doc: Any) -> list[Violation]:
    validator = jsonschema.Draft202012Validator(
        load_schema(), format_checker=jsonschema.FormatChecker()
    )
    return [
        Violation(
            "SCHEMA", "/".join(str(p) for p in e.absolute_path) or "$", e.message[:300]
        )
        for e in sorted(
            validator.iter_errors(doc), key=lambda e: list(map(str, e.path))
        )
    ]


def validate_swing_report(doc: Any) -> SwingReport:
    """Validate ``doc`` against the v1 contract and return the typed report.

    Preconditions: none (any input is accepted and checked).
    Postconditions: the returned ``SwingReport`` satisfies the schema and
    invariants I1-I12; withheld quantities hold no values; advertised
    quantities carry provisional engine-spread uncertainty.

    Raises:
        SwingReportContractError: listing every violation found.
    """
    violations = _schema_violations(doc)
    if isinstance(doc, dict):
        violations += _Checker(doc).run()
    if violations:
        raise SwingReportContractError(violations)
    return report_from_dict(doc)
