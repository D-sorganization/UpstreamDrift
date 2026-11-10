"""Physics engine lane guard and JUnit floor enforcement (issue #14948)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as ET  # type: ignore[import-untyped]

KNOWN_ENGINES = frozenset(
    {
        "mujoco",
        "pinocchio",
        "drake",
        "opensim",
        "myosuite",
        "jaxsim",
        "nimble",
        "matlab",
    }
)


@dataclass(frozen=True)
class EngineFloorReport:
    lane: str
    executed: int
    skipped: int
    xfailed: int
    xpassed_strict: int
    passed: int
    failed: int
    errors: int
    success: bool
    message: str

    @property
    def engine(self) -> str:
        return self.lane

    @property
    def p(self) -> int:
        return self.passed

    @property
    def s(self) -> int:
        return self.skipped

    @property
    def f(self) -> int:
        return self.failed


EngineFloorResult = EngineFloorReport


def parse_required_engines(raw: str | None) -> set[str]:
    """Parse comma-separated engine names. Raises ValueError on unknown engine."""
    if not raw or not raw.strip():
        return set()
    engines = {e.strip().lower() for e in raw.split(",") if e.strip()}
    if unknown := engines - KNOWN_ENGINES:
        raise ValueError(
            f"Unknown engine(s) in UD_REQUIRE_ENGINES: {', '.join(sorted(unknown))}. "
            f"Known engines: {', '.join(sorted(KNOWN_ENGINES))}."
        )
    return engines


def is_valid_declared_skip(item: Any) -> bool:
    """Check for @pytest.mark.declared_skip(issue='#NNNN', reason='...')."""
    if (
        m := getattr(item, "get_closest_marker", lambda _: None)("declared_skip")
    ) is None:
        return False
    issue, reason = m.kwargs.get("issue"), m.kwargs.get("reason")
    for a in m.args:
        if isinstance(a, str):
            issue, reason = (
                (a, reason)
                if a.startswith("#") and issue is None
                else (issue, a if reason is None else reason)
            )
    return (
        isinstance(issue, str)
        and issue.startswith("#")
        and len(issue.strip()) > 1
        and isinstance(reason, str)
        and bool(reason.strip())
    )


def _match_engine_from_markers_or_node(
    item: Any, node_lower: str, required: set[str]
) -> str | None:
    """Identify which required engine is exercised by item markers or nodeid."""
    if item is not None:
        markers = getattr(
            item, "iter_markers", lambda: getattr(item, "own_markers", [])
        )()
        for m in markers:
            name = getattr(m, "name", "").lower()
            if name.startswith("requires_"):
                eng = name[len("requires_") :]
                if eng in required:
                    return eng
    for eng in required:
        if (
            f"requires_{eng}" in node_lower
            or f"test_{eng}" in node_lower
            or f"_{eng}_" in node_lower
            or f"_{eng}." in node_lower
            or f"/{eng}/" in node_lower
            or f"\\{eng}\\" in node_lower
            or f"::{eng}" in node_lower
            or f"::test_{eng}" in node_lower
        ):
            return eng
    return None


def enforce_required_engines_on_report(
    report: Any, nodeid: str = "", item: Any = None
) -> None:
    """Turn a skipped engine test into failure when that engine is required."""
    if item is None and hasattr(nodeid, "skipped"):
        item, report, nodeid = report, nodeid, getattr(report, "nodeid", "")
    if not getattr(report, "skipped", False) or getattr(report, "wasxfail", None):
        return
    required = parse_required_engines(os.environ.get("UD_REQUIRE_ENGINES"))
    if not required:
        return
    node = nodeid or getattr(item, "nodeid", str(item or ""))
    if item is not None and is_valid_declared_skip(item):
        return
    matched = _match_engine_from_markers_or_node(item, node.lower(), required)
    if matched is None:
        return

    report.outcome = "failed"
    try:
        report.skipped = False
        report.failed = True
    except AttributeError:
        pass

    report.longrepr = (
        f"Engine lane guard (UD_REQUIRE_ENGINES): test '{node}' exercising "
        f"engine '{matched}' was skipped in a lane that requires '{matched}'. "
        f"Install the engine or use @pytest.mark.declared_skip(issue='#NNNN', reason='...')."
    )


enforce_required_engine_skip_guard = enforce_required_engines_on_report


def _matches_engine_token(
    tc: ET.Element, suite_name: str, engine: str, file_stem: str = ""
) -> bool:
    """Check if a JUnit testcase matches the given engine token."""
    eng = engine.lower()
    for val in [tc.get("classname", ""), tc.get("name", ""), tc.get("file", "")]:
        low = val.lower()
        if (
            low == eng
            or f"requires_{eng}" in low
            or f"test_{eng}" in low
            or f"_{eng}_" in low
            or f"_{eng}." in low
            or f"/{eng}/" in low
            or f"\\{eng}\\" in low
            or f"::{eng}" in low
        ):
            return True
    for val in [suite_name, file_stem]:
        low = val.lower()
        if (
            low == eng
            or f"requires_{eng}" in low
            or f"test_{eng}" in low
            or f"_{eng}_" in low
            or f"_{eng}." in low
            or f"-{eng}" in low
            or f"{eng}-" in low
        ):
            return True
    return False


def _classify_testcase(tc: ET.Element) -> tuple[int, int, int, int, int, int]:
    """Classify testcase into (passed, failed, errors, skipped, xfailed, xpassed_strict)."""
    fail, err, skip = tc.find("failure"), tc.find("error"), tc.find("skipped")
    if fail is not None:
        xp = 1 if "xpass(strict)" in (fail.get("message") or "").lower() else 0
        return 0, 1, 0, 0, 0, xp
    if err is not None:
        return 0, 0, 1, 0, 0, 0
    if skip is not None:
        info = ((skip.get("type") or "") + " " + (skip.get("message") or "")).lower()
        return (0, 0, 0, 0, 1, 0) if "xfail" in info else (0, 0, 0, 1, 0, 0)
    return 1, 0, 0, 0, 0, 0


def _collect_xml_files(junit_path: Path) -> list[Path]:
    """Collect xml file(s) from a file or directory path."""
    if junit_path.is_file():
        return [junit_path]
    if junit_path.is_dir():
        return sorted(junit_path.glob("*.xml"))
    return []


def _accumulate_xml_engine_counts(
    xml_path: Path, engine: str
) -> tuple[int, int, int, int, int, int]:
    """Accumulate (passed, failed, errors, skipped, xfailed, xpassed_strict)."""
    p, f, err, sk, xf, xp = 0, 0, 0, 0, 0, 0
    root = ET.parse(xml_path).getroot()
    file_stem = xml_path.stem
    suites = list(root.iter("testsuite")) or [root]
    all_tcs = [
        (tc, s.get("name", "")) for s in suites for tc in s.iter("testcase")
    ]
    matched = [
        tc
        for tc, sname in all_tcs
        if _matches_engine_token(tc, sname, engine, file_stem)
    ]
    has_other_engine = any(
        _matches_engine_token(tc, sname, other, file_stem)
        for tc, sname in all_tcs
        for other in KNOWN_ENGINES - {engine.lower()}
    )
    target_tcs = matched if (matched or has_other_engine) else [tc for tc, _ in all_tcs]
    for tc in target_tcs:
        dp, df, de, ds, dx, dxp = _classify_testcase(tc)
        p, f, err, sk, xf, xp = (
            p + dp,
            f + df,
            err + de,
            sk + ds,
            xf + dx,
            xp + dxp,
        )
    return p, f, err, sk, xf, xp


def _validate_lane_config(config_path: Path, lane: str) -> tuple[int, int]:
    """Validate lane in config and return (min_passed, max_skipped)."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    not_covered = {
        **config.get("not_covered", {}),
        **config.get("not_covered_engines", {}),
    }
    if lane in not_covered:
        info = not_covered[lane]
        raise ValueError(
            f"Lane '{lane}' is declared not_covered: {info.get('reason')} "
            f"(issue {info.get('issue')})"
        )
    lanes = config.get("lanes", {})
    if lane not in lanes:
        raise ValueError(
            f"Unknown lane '{lane}'. Configured: {', '.join(sorted(lanes))}"
        )
    lane_floor = lanes[lane]
    min_passed = int(lane_floor.get("min_passed", lane_floor.get("min_executed", 1)))
    max_skipped = int(lane_floor.get("max_skipped", 0))
    return min_passed, max_skipped


def verify_junit_engine_floor(
    junit_xml_path: Path, floor_config_path: Path, lane: str
) -> EngineFloorReport:
    """Validate JUnit results against min_passed and max_skipped floors."""
    min_passed, max_skipped = _validate_lane_config(floor_config_path, lane)
    files = _collect_xml_files(junit_xml_path)
    if not files:
        return EngineFloorReport(
            lane,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            False,
            f"JUnit XML file not found: {junit_xml_path}",
        )

    p, f, err, s, xf, xp = 0, 0, 0, 0, 0, 0
    for file in files:
        dp, df, de, ds, dx, dxp = _accumulate_xml_engine_counts(file, lane)
        p, f, err, s, xf, xp = (
            p + dp,
            f + df,
            err + de,
            s + ds,
            xf + dx,
            xp + dxp,
        )

    executed = p + f + err + xf
    failed_total = f + err

    if failed_total > 0:
        msg = f"Engine '{lane}' had {failed_total} failed/errored test(s) (passed={p}, skipped={s})."
        return EngineFloorReport(
            lane, executed, s, xf, xp, p, failed_total, err, False, msg
        )
    if p == 0 and s > 0:
        msg = f"All {s} test(s) for engine '{lane}' were skipped (0 passed; min_passed={min_passed})."
        return EngineFloorReport(lane, executed, s, xf, xp, p, 0, 0, False, msg)
    if p < min_passed:
        msg = f"Engine '{lane}' passed {p} test(s), below floor min_passed={min_passed} (skipped={s})."
        return EngineFloorReport(lane, executed, s, xf, xp, p, 0, 0, False, msg)
    if s > max_skipped:
        msg = f"Engine '{lane}' had {s} skipped test(s), exceeding max_skipped={max_skipped}."
        return EngineFloorReport(lane, executed, s, xf, xp, p, 0, 0, False, msg)

    msg = f"Lane '{lane}' passed floor check: {executed} executed (passed={p}, failed={failed_total}, skipped={s})."
    return EngineFloorReport(lane, executed, s, xf, xp, p, 0, 0, True, msg)
