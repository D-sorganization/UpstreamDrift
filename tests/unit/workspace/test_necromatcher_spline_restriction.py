from copy import deepcopy
from dataclasses import replace
from typing import Any
from pathlib import Path
import pytest
from restriction_fixture import restricted_case
from src.shared.python.workspace.necromatcher_fit_records import (
    build_native_fit_payload,
)
from src.shared.python.workspace.necromatcher_spline import preserved_fit_spline

pytestmark = pytest.mark.unit


def build(c: dict[str, Any]) -> dict[str, Any]:
    return build_native_fit_payload(
        c["request"], c["source"], c["result"], c["dense"], c["stamp"], 0.0
    )


def test_public_builder_keeps_truthful_restriction_only_seed(
    fit_case: Any, tmp_path: Path
) -> None:
    c = restricted_case(fit_case, tmp_path)
    before = deepcopy(c["source"])
    record = build(c)
    assert (
        record["provenance"]["spline_interval_restriction"] == c["receipt"].to_record()
    )
    assert record["provenance"]["source_fit_scope"] == c["scope"].to_record()
    assert (
        record["provenance"]["warm_start_fit_hash"] == c["request"]["source_fit_hash"]
    )
    assert (
        record["evidence"]["original_fit"]["spline_start"]
        == c["receipt"].restricted_start.to_record()
    )
    assert record["evidence"]["original_fit"]["optimizer_ran"] is False
    assert record["evidence"]["original_fit"]["initialization"] is None
    assert "spline_interval_restriction_only" in record["evidence"]["rejection_reasons"]
    assert "authored_initialization_only" not in record["evidence"]["rejection_reasons"]
    assert c["source"] == before
    assert preserved_fit_spline(record) == c["receipt"].restricted_start


@pytest.mark.parametrize(
    "tamper",
    ["receipt", "locked", "frames", "training", "optimizer", "model", "options"],
)
def test_builder_rejects_tampered_restriction_seed(
    fit_case: Any, tmp_path: Path, tamper: str
) -> None:
    c = restricted_case(fit_case, tmp_path)
    if tamper == "receipt":
        c["request"]["spline_interval_restriction"]["restricted_start"][
            "spline_coefficients"
        ][0] += 0.01
    elif tamper == "locked":
        q = c["dense"][2].copy()
        q[:, 1] += 0.01
        c["dense"] = (c["dense"][0], c["dense"][1], q)
        c["result"] = replace(c["result"], q=q)
    elif tamper == "frames":
        c["dense"][1][1]["frame_sha256"] = "a" * 64
    elif tamper == "training":
        c["request"]["options"]["frame_indices"] = [0, 2]
    elif tamper == "optimizer":
        c["result"] = replace(c["result"], optimizer_ran=True)
    elif tamper == "model":
        c["result"] = replace(c["result"], model_sha="foreign-model")
    else:
        c["request"]["options"]["operation"] = "fit"
    with pytest.raises(ValueError):
        build(c)


def test_recall_rebinds_restriction_parent_not_caller_receipt(
    fit_case: Any, tmp_path: Path
) -> None:
    c = restricted_case(fit_case, tmp_path)
    record = build(c)
    from src.shared.python.workspace.necromatcher_spline_restriction import (
        validate_spline_restriction_payload,
    )

    validate_spline_restriction_payload(c["library"], record)
    record["provenance"]["warm_start_fit_hash"] = "sha256:" + "b" * 64
    with pytest.raises(ValueError):
        validate_spline_restriction_payload(c["library"], record)


@pytest.mark.parametrize(
    "tamper", ["dense", "snapshot", "camera", "config", "prior", "binding", "operation"]
)
def test_stored_restriction_rejects_independently_tampered_fields(
    fit_case: Any, tmp_path: Path, tamper: str
) -> None:
    from src.shared.python.workspace.necromatcher_spline_restriction import (
        validate_spline_restriction_payload,
    )

    c = restricted_case(fit_case, tmp_path)
    record = build(c)
    original = record["evidence"]["original_fit"]
    if tamper == "dense":
        record["q"][0][0] += 0.01
    elif tamper == "snapshot":
        original["initial_spline"] = c["start"].to_record()
    elif tamper == "camera":
        original["camera"] = {"foreign": True}
    elif tamper == "config":
        original["config"] = {}
    elif tamper == "prior":
        record["provenance"]["spline_restriction_prior"]["frame_index"] = False
    elif tamper == "binding":
        record["provenance"]["source_fit_scope_binding"]["first_pts"] = [1, 10]
    else:
        record["provenance"]["operation"] = "fit"
    with pytest.raises(ValueError):
        validate_spline_restriction_payload(c["library"], record)


@pytest.mark.parametrize(
    "tamper", ["narrow", "knot_count", "free_bounds", "locked_bounds"]
)
def test_derivation_rejects_scope_or_strict_bounds_before_effects(
    fit_case: Any, tmp_path: Path, tamper: str
) -> None:
    from src.shared.python.workspace.necromatcher_spline_restriction import (
        derive_spline_restriction,
    )

    c = restricted_case(fit_case, tmp_path)
    options = deepcopy(c["request"]["options"])
    if tamper == "narrow":
        # Approved scope spans 0..1; selected 0..2 cannot be a restriction.
        options["frame_indices"] = [0, 2]
    elif tamper == "knot_count":
        options["knot_count"] = 3
    elif tamper == "free_bounds":
        options["config"]["coordinate_bounds"] = [["hip", -0.01, 0.01]]
    else:
        options["config"]["coordinate_bounds"] = [["locked", -0.01, 0.01]]
    before = (c["library"].root / "project.json").read_bytes()
    with pytest.raises(ValueError):
        derive_spline_restriction(
            c["library"], "restriction-parent", options, c["scope"]
        )
    assert before == (c["library"].root / "project.json").read_bytes()


@pytest.mark.parametrize(
    "key", ["spline_interval_restriction", "spline_restriction_prior"]
)
def test_legacy_mode_cannot_silently_ignore_restriction_metadata(
    fit_case: Any, tmp_path: Path, key: str
) -> None:
    from src.shared.python.workspace.necromatcher_spline_restriction import (
        validate_spline_restriction_request,
    )

    c = restricted_case(fit_case, tmp_path)
    request = deepcopy(c["request"])
    request["options"].update(operation="fit", initialization_source="sampled_parent")
    request.pop("spline_interval_restriction")
    request.pop("spline_restriction_prior")
    assert validate_spline_restriction_request(c["library"], request) is None
    request[key] = c["request"][key]
    with pytest.raises(ValueError):
        validate_spline_restriction_request(c["library"], request)


def test_generic_later_interval_declares_selected_first_prior(
    fit_case: Any, tmp_path: Path
) -> None:
    from test_scope_fixtures import fixture_review_artifact
    from src.shared.python.workspace.necromatcher_capture_identity import (
        capture_identity,
    )
    from src.shared.python.workspace.necromatcher_spline_restriction import (
        derive_spline_restriction,
        spline_restriction_prior,
    )

    c = restricted_case(fit_case, tmp_path)
    identity = capture_identity(c["library"], c["source"]["capture_id"])
    (tmp_path / "later").mkdir()
    artifact = fixture_review_artifact(tmp_path / "later", identity, 1, 3)
    c["library"].add_source_scope_review(
        "later-review", "practice", Path(artifact.path)
    )
    scope = c["library"].load_source_scope_review("later-review")
    options = deepcopy(c["request"]["options"])
    options["frame_indices"] = [1, 2]
    assert spline_restriction_prior(options) == {
        "policy": "selected_first_parent_pose",
        "frame_index": 1,
    }
    receipt = derive_spline_restriction(
        c["library"], "restriction-parent", options, scope
    )
    assert receipt.restricted_start.knot_times == (0.1, 0.2)
    assert c["source"]["q"][1][0] != c["source"]["q"][0][0]


def test_restriction_seed_uses_canonical_store_and_recall(
    fit_case: Any, tmp_path: Path
) -> None:
    import json

    c = restricted_case(fit_case, tmp_path)
    record = build(c)
    path = tmp_path / "restriction-seed.json"
    path.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
    c["library"].add_fit("restriction-seed", "practice", path)
    saved = c["library"].load_fit("restriction-seed")
    assert (
        saved["provenance"]["spline_restriction_prior"]
        == c["request"]["spline_restriction_prior"]
    )
    assert preserved_fit_spline(saved) == c["receipt"].restricted_start


def test_restriction_owner_cold_import_is_sdk_free_and_fingerprinted() -> None:
    import os
    import subprocess
    import sys
    from src.shared.python.workspace.necromatcher_fit_jobs import fit_execution_stamp

    source = "src/shared/python/workspace/necromatcher_spline_restriction.py"
    stamp = fit_execution_stamp()
    assert source in stamp["source_files"]
    code = """import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco','pinocchio','pydrake','opensim','PyQt6'}:
            raise AssertionError('native import: ' + fullname)
sys.meta_path.insert(0, Block())
from src.shared.python.workspace.necromatcher_spline_restriction import derive_spline_restriction, validate_spline_restriction_request
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_erased_top_level_markers_do_not_downgrade_persisted_restriction(
    fit_case: Any, tmp_path: Path
) -> None:
    from src.shared.python.workspace.necromatcher_spline_restriction import (
        validate_spline_restriction_payload,
    )

    c = restricted_case(fit_case, tmp_path)
    record = build(c)
    provenance = record["provenance"]
    for key in ("operation", "spline_interval_restriction", "spline_restriction_prior"):
        provenance.pop(key)
    record["evidence"]["rejection_reasons"].remove("spline_interval_restriction_only")
    provenance["request_options"]["initialization_source"] = "sampled_parent"
    record["evidence"]["original_fit"]["initialization_source"] = "sampled_parent"
    with pytest.raises(ValueError):
        validate_spline_restriction_payload(c["library"], record)


def test_restriction_rejects_saved_parent_curve_disagreement_beyond_roundoff(
    fit_case: Any, tmp_path: Path
) -> None:
    c = restricted_case(fit_case, tmp_path)
    c["source"]["q"][0][0] += 1e-10
    with pytest.raises(ValueError, match="roundoff"):
        build(c)
