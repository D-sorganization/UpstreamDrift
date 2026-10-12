"""Lossless restriction must be authenticated before scheduling or publication."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pytest

from restriction_fixture import restricted_case
from test_necromatcher_authenticated_refit_queue import Service
from src.shared.python.motion_matching.historical_fit import ImageFitConfig
from src.shared.python.workspace import necromatcher_fit_jobs as jobs
from src.shared.python.workspace.necromatcher_fit_records import (
    build_native_fit_payload,
)

pytestmark = pytest.mark.unit


def _options(case: dict[str, Any]) -> jobs.NativeRefitOptions:
    record = deepcopy(case["request"]["options"])
    record["config"] = ImageFitConfig.from_record(record["config"])
    return jobs.NativeRefitOptions(**record)


def test_queue_binds_exact_receipt_prior_and_option_hash(
    fit_case: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    case = restricted_case(fit_case, tmp_path)
    library = case["library"]
    service = Service(library)
    monkeypatch.setattr(jobs, "fit_execution_stamp", lambda: case["stamp"])
    options = _options(case)
    _, root = jobs.start_native_refit(
        library,
        "restriction-parent",
        "restriction-seed",
        options,
        service,
        source_scope=case["scope"],
    )
    request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    assert request["spline_interval_restriction"] == case["receipt"].to_record()
    assert (
        request["spline_restriction_prior"]
        == case["request"]["spline_restriction_prior"]
    )
    scoped = {
        "options": asdict(options),
        "source_scope": request["source_scope"],
        "source_scope_binding": request["source_scope_binding"],
    }
    expected = jobs._digest(
        {
            "options": scoped,
            "spline_interval_restriction": request["spline_interval_restriction"],
            "spline_restriction_prior": request["spline_restriction_prior"],
        }
    )
    assert service.calls[0].hashes.controller_hash == expected


def test_unregistered_restriction_scope_fails_before_queue_effects(
    fit_case: Any, tmp_path: Path
) -> None:
    case = restricted_case(fit_case, tmp_path)
    library = case["library"]
    service = Service(library)
    with pytest.raises(ValueError, match="scope|Scope"):
        jobs.start_native_refit(
            library, "restriction-parent", "restriction-seed", _options(case), service
        )
    assert service.calls == []
    assert not (library.root / "runs").exists()


@pytest.mark.parametrize(
    "term", ["spline_interval_restriction", "spline_restriction_prior"]
)
def test_publication_rejects_candidate_receipt_or_prior_tampering(
    fit_case: Any, tmp_path: Path, term: str
) -> None:
    case = restricted_case(fit_case, tmp_path)
    payload = build_native_fit_payload(
        case["request"],
        case["source"],
        case["result"],
        case["dense"],
        case["stamp"],
        0.0,
    )
    payload["provenance"][term] = {}
    with pytest.raises(ValueError, match="Restriction|restriction"):
        jobs._verify_candidate_reads(case["library"], case["request"], payload)


@pytest.mark.parametrize("delayed", [False, True])
def test_restriction_tamper_blocks_candidate_and_delayed_fit_writes(
    fit_case: Any, tmp_path: Path, monkeypatch: Any, delayed: bool
) -> None:
    case = restricted_case(fit_case, tmp_path)
    library = case["library"]
    service = Service(library)
    payload = build_native_fit_payload(
        case["request"],
        case["source"],
        case["result"],
        case["dense"],
        case["stamp"],
        0.0,
    )
    monkeypatch.setattr(jobs, "fit_execution_stamp", lambda: case["stamp"])
    monkeypatch.setattr(jobs, "_execute_worker", lambda *args: {"fit": payload})
    _, root = jobs.start_native_refit(
        library,
        "restriction-parent",
        "restriction-seed",
        _options(case),
        service,
        source_scope=case["scope"],
    )
    outcome = service.work(lambda _: None, lambda: False) if delayed else None
    payload["provenance"]["spline_restriction_prior"]["frame_index"] = 1
    writes = []
    monkeypatch.setattr(library, "add_fit", lambda *args: writes.append(args))
    with pytest.raises(ValueError, match="Restriction|restriction"):
        if outcome is not None:
            outcome.publish()
        else:
            service.work(lambda _: None, lambda: False)
    assert writes == []
    assert (root / "candidate.json").exists() is delayed
