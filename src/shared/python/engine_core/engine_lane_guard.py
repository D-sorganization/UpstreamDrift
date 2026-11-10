"""Physics engine lane guard and JUnit floor enforcement (issue #14948)."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def enforce_required_engine_skip_guard(
    item: Any, report: Any, required_engines: set[str] | None = None
) -> None:
    """Turn an engine-test skip into failure when that engine is required."""
    if not getattr(report, "skipped", False) or getattr(report, "wasxfail", None):
        return
    if required_engines is None:
        required_engines = parse_required_engines(os.environ.get("UD_REQUIRE_ENGINES"))
    if not required_engines:
        return

    markers = getattr(item, "iter_markers", lambda: getattr(item, "own_markers", []))()
    item_engine = next(
        (
            getattr(m, "name", "")[len("requires_") :].lower()
            for m in markers
            if getattr(m, "name", "").startswith("requires_")
            and getattr(m, "name", "")[len("requires_") :].lower() in required_engines
        ),
        None,
    )
    if item_engine is None or is_valid_declared_skip(item):
        return

    report.outcome = "failed"
    report.skipped, report.failed = False, True
    nodeid = getattr(item, "nodeid", str(item))
    report.longrepr = (
        f"Engine lane guard (UD_REQUIRE_ENGINES): test '{nodeid}' marked "
        f"requires_{item_engine} was skipped in a lane that requires '{item_engine}'. "
        f"Install the engine or use @pytest.mark.declared_skip(issue='#NNNN', reason='...')."
    )


def verify_junit_engine_floor(
    junit_xml_path: Path, floor_config_path: Path, lane: str
) -> EngineFloorReport:
    """Validate JUnit results against min_executed and max_skipped floors."""
    config = json.loads(floor_config_path.read_text(encoding="utf-8"))
    if lane in config.get("not_covered", {}):
        info = config["not_covered"][lane]
        raise ValueError(
            f"Lane '{lane}' is declared not_covered: {info.get('reason')} (issue {info.get('issue')})"
        )
    if lane not in config.get("lanes", {}):
        raise ValueError(
            f"Unknown lane '{lane}'. Configured: {', '.join(sorted(config.get('lanes', {})))}"
        )

    lane_floor = config["lanes"][lane]
    min_exec = int(lane_floor.get("min_executed", 1))
    max_skip = int(lane_floor.get("max_skipped", 0))
    if not junit_xml_path.is_file():
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

    p, f, err, sk, xf, xp_strict = 0, 0, 0, 0, 0, 0
    for tc in ET.parse(junit_xml_path).getroot().iter("testcase"):
        fail, error, skip = tc.find("failure"), tc.find("error"), tc.find("skipped")
        if fail is not None:
            f += 1
            xp_strict += int(
                any(
                    k in ((fail.get("message") or "") + (fail.text or "")).lower()
                    for k in ("xpass", "strict")
                )
            )
        elif error is not None:
            err += 1
        elif skip is not None:
            info = (
                (skip.get("type") or "") + " " + (skip.get("message") or "")
            ).lower()
            xf, sk = (xf + 1, sk) if "xfail" in info else (xf, sk + 1)
        else:
            p += 1

    executed = p + f + err + xf
    failures = [
        f"executed {executed} < min_executed {min_exec}" if executed < min_exec else "",
        (
            f"0 tests passed in lane '{lane}' (all tests skipped or 0 collected)"
            if not (p or f or err)
            else ""
        ),
        f"skipped {sk} > max_skipped {max_skip}" if sk > max_skip else "",
        f"xpassed_strict {xp_strict} > 0" if xp_strict > 0 else "",
    ]
    failures = [msg for msg in failures if msg]
    success = not failures
    msg = (
        f"Lane '{lane}' passed floor check: {executed} executed (passed={p}, failed={f}, errors={err}, xfailed={xf}), {sk} skipped."
        if success
        else f"Lane '{lane}' failed floor check: {'; '.join(failures)}."
    )
    return EngineFloorReport(lane, executed, sk, xf, xp_strict, p, f, err, success, msg)
