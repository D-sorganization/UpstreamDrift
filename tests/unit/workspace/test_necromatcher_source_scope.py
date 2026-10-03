"""Reviewed source-scope declarations and authenticated boundary contracts."""

from dataclasses import replace
from pathlib import Path
import os
import subprocess
import sys
import pytest
from tests.unit.workspace.test_scope_fixtures import (
    fixture_capture,
    fixture_review_artifact,
)
from src.shared.python.workspace.necromatcher_source_scope import (
    SourceScopeReview,
    SourceFitScope,
    bind_source_fit_scope,
    validate_scope_selection,
    validate_scope_descendant,
)
from src.shared.python.motion_matching.historical_fit.shaft_observations import (
    ShaftAxisEvidence,
    ShaftAxisSegment,
    SourceBoundShaftFrame,
)


def make_scope(tmp_path: Path, first: int = 0, end: int = 4) -> tuple:
    tmp_path.mkdir(parents=True, exist_ok=True)
    identity = fixture_capture()
    ref = fixture_review_artifact(tmp_path, identity, first, end)
    review = SourceScopeReview(
        ref,
        Path(ref.path).stat().st_size,
        identity.frames[first],
        identity.frames[end] if end < len(identity.frames) else None,
        "Synthetic conservative window",
        "exclude transition",
        False,
    )
    scope = SourceFitScope(
        identity.capture_id,
        identity.capture_hash,
        identity.source_clock_sha256,
        first,
        end,
        review,
    )
    return identity, scope


def test_cold_import_blocks_native_sdks() -> None:
    script = """import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'mujoco','PyQt6','opensim','pinocchio'}:
   raise AssertionError('Native SDK import forbidden: '+fullname)
sys.meta_path.insert(0,Block())
from src.shared.python.workspace.necromatcher_source_scope import SourceFitScope
assert not any(x.split('.')[0] in {'mujoco','PyQt6'} for x in sys.modules)
"""
    root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        str(x) for x in (root, root / "src", root / "src/shared/python")
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        cwd=root,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_half_open_positive_and_exclusive_end(tmp_path: Path) -> None:
    identity, scope = make_scope(tmp_path)
    bound = bind_source_fit_scope(identity, scope)
    validate_scope_selection(bound, (0, 3))
    with pytest.raises(ValueError, match="scope|outside"):
        validate_scope_selection(bound, (0, 4))


@pytest.mark.parametrize(
    "first,end", [(True, 4), ("0", 4), (0, True), (0, "4"), (1, 1), (-1, 4), (0, 1)]
)
def test_invalid_bounds_have_valid_review(
    tmp_path: Path, first: object, end: object
) -> None:
    _, scope = make_scope(tmp_path)
    with pytest.raises(ValueError):
        replace(scope, first_frame=first, end_exclusive_frame=end)


def test_wrong_clock_rejects_before_work(tmp_path: Path) -> None:
    identity, scope = make_scope(tmp_path)
    with pytest.raises(ValueError, match="clock"):
        bind_source_fit_scope(
            replace(identity, source_clock_sha256="sha256:" + "e" * 64), scope
        )


def test_boundary_pts_or_decoder_mutation_rejects(tmp_path: Path) -> None:
    identity, scope = make_scope(tmp_path)
    for changed in (
        replace(identity.frames[4], pts_ticks=105),
        replace(identity.frames[4], decoder_name="foreign-decoder"),
    ):
        with pytest.raises(ValueError, match="frame|boundary|identity"):
            bind_source_fit_scope(
                replace(identity, frames=identity.frames[:4] + (changed,)), scope
            )


def test_changed_review_bytes_reject_even_with_declared_digest(tmp_path: Path) -> None:
    identity, scope = make_scope(tmp_path)
    Path(scope.review.artifact.path).write_bytes(b"{}")
    with pytest.raises(ValueError, match="hash|review|bytes"):
        bind_source_fit_scope(identity, scope)


def test_descendant_cannot_widen_or_change_capture(tmp_path: Path) -> None:
    identity, scope = make_scope(tmp_path)
    bound = bind_source_fit_scope(identity, scope)
    validate_scope_descendant(bound, bound)
    _, narrow = make_scope(tmp_path / "narrow", 1, 3)
    validate_scope_descendant(bound, bind_source_fit_scope(identity, narrow))
    _, broader = make_scope(tmp_path / "broader", 0, 5)
    broad_bound = bind_source_fit_scope(identity, broader)
    with pytest.raises(ValueError, match="widen|scope"):
        validate_scope_descendant(bound, broad_bound)
    # A typed declaration alone is not a verified foreign-capture binding.
    with pytest.raises(ValueError, match="capture|scope"):
        bind_source_fit_scope(identity, replace(scope, capture_id="foreign"))


@pytest.mark.parametrize("status", ["observed", "ambiguous"])
def test_out_of_scope_shaft_never_dispatches(tmp_path: Path, status: str) -> None:
    identity, scope = make_scope(tmp_path)
    bound = bind_source_fit_scope(identity, scope)
    segment = (
        ShaftAxisSegment(
            "observed",
            ((1.0, 1.0), (2.0, 2.0)),
            "reviewer",
            "synthetic",
            0.5,
            None,
            3.0,
        )
        if status == "observed"
        else ShaftAxisSegment(
            "ambiguous", None, "reviewer", "uncertain", None, None, None
        )
    )
    frame = SourceBoundShaftFrame(
        4, identity.frames[4], identity.png_sha256[4], segment
    )
    evidence = ShaftAxisEvidence(
        identity.capture_id,
        identity.capture_hash,
        identity.source_sha256,
        (32, 24),
        (frame,),
    )
    dispatched = []
    with pytest.raises(ValueError, match="shaft|scope|outside"):
        validate_scope_selection(bound, (0, 3), evidence)
        dispatched.append("optimizer")
    assert dispatched == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("contact_calibrated", 0),
        ("first_frame", False),
        ("schema", "foreign/1"),
        ("reason", "unreviewed change"),
        ("end_exclusive_frame", 5),
    ],
)
def test_rehashed_artifact_content_must_match_declaration(
    tmp_path: Path, key: str, value: object
) -> None:
    import json
    import hashlib

    identity, scope = make_scope(tmp_path)
    path = Path(scope.review.artifact.path)
    payload = json.loads(path.read_bytes())
    payload[key] = value
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    reference = replace(
        scope.review.artifact, hash="sha256:" + hashlib.sha256(raw).hexdigest()
    )
    review = replace(scope.review, artifact=reference, receipt_bytes=len(raw))
    with pytest.raises(ValueError, match="schema|content|declaration"):
        bind_source_fit_scope(identity, replace(scope, review=review))


def test_review_metadata_and_scope_records_are_detached(tmp_path: Path) -> None:
    from dataclasses import FrozenInstanceError

    identity, scope = make_scope(tmp_path)
    record = scope.to_record()
    restored = SourceFitScope.from_record(record)
    assert restored == scope
    record["review"]["first_identity"]["frame_id"] = "tampered"
    assert scope.review.first_identity == identity.frames[0]
    with pytest.raises(TypeError):
        scope.review.artifact.metadata["changed"] = True
    with pytest.raises(FrozenInstanceError):
        scope.first_frame = 2


def test_approved_window_and_actual_selected_domain_are_distinct(
    tmp_path: Path,
) -> None:
    from fractions import Fraction

    identity, scope = make_scope(tmp_path)
    bound = bind_source_fit_scope(identity, scope)
    selected = bound.selected_domain((0, 2))
    assert selected.frame_indices == (0, 2)
    assert selected.first_pts == Fraction(100, 30)
    assert selected.last_pts == Fraction(102, 30)
    assert bound.end_exclusive_pts == Fraction(104, 30)


def test_omission_inherits_and_requested_scope_cannot_widen(tmp_path: Path) -> None:
    from src.shared.python.workspace.necromatcher_source_scope import (
        resolve_source_fit_scope,
    )

    identity, scope = make_scope(tmp_path)
    assert resolve_source_fit_scope(identity, None, None) is None
    assert resolve_source_fit_scope(identity, scope, None).scope == scope
    _, broader = make_scope(tmp_path / "broader", 0, 5)
    with pytest.raises(ValueError, match="widen"):
        resolve_source_fit_scope(identity, scope, broader)


@pytest.mark.parametrize(
    "indices", [(0, True), (0, "2"), (0, 0), (2, 1), (0,), (-1, 2)]
)
def test_body_selection_rejects_invalid_records(tmp_path: Path, indices: tuple) -> None:
    identity, scope = make_scope(tmp_path)
    with pytest.raises(ValueError):
        validate_scope_selection(bind_source_fit_scope(identity, scope), indices)


@pytest.mark.parametrize(
    "changes",
    [
        {"timing_mode": "estimated_cfr", "is_timing_exact": False},
        {"physical_time_s": 1.0, "physical_time_reason": "declared"},
    ],
)
def test_review_boundaries_require_exact_unqualified_source_clock(
    tmp_path: Path, changes: dict
) -> None:
    _, scope = make_scope(tmp_path)
    with pytest.raises(ValueError, match="PTS|physical|clock"):
        replace(
            scope.review, first_identity=replace(scope.review.first_identity, **changes)
        )


def test_relative_registered_review_resolves_against_artifact_root(
    tmp_path: Path,
) -> None:
    from src.shared.python.workspace.necromatcher_source_scope import (
        resolve_source_fit_scope,
    )

    identity, scope = make_scope(tmp_path)
    reference = replace(scope.review.artifact, path="review.json")
    scoped = replace(scope, review=replace(scope.review, artifact=reference))
    bound = bind_source_fit_scope(identity, scoped, artifact_root=tmp_path)
    assert bound.scope == scoped
    assert (
        resolve_source_fit_scope(identity, scoped, None, artifact_root=tmp_path).scope
        == scoped
    )
    with pytest.raises(FileNotFoundError):
        bind_source_fit_scope(identity, scoped, artifact_root=tmp_path / "foreign")


def test_review_link_is_rejected_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, scope = make_scope(tmp_path)
    target = Path(scope.review.artifact.path)
    original = Path.is_symlink
    monkeypatch.setattr(
        Path, "is_symlink", lambda value: value == target or original(value)
    )
    with pytest.raises(ValueError, match="link"):
        bind_source_fit_scope(identity, scope, artifact_root=tmp_path)


def test_relative_review_cannot_escape_artifact_root(tmp_path: Path) -> None:
    identity, scope = make_scope(tmp_path)
    child = tmp_path / "assets"
    child.mkdir()
    scoped = replace(
        scope,
        review=replace(
            scope.review, artifact=replace(scope.review.artifact, path="../review.json")
        ),
    )
    with pytest.raises(ValueError, match="root|relative|escape"):
        bind_source_fit_scope(identity, scoped, artifact_root=child)


def test_scope_factory_authenticates_same_review_buffer(tmp_path: Path) -> None:
    identity, scope = make_scope(tmp_path)
    actual = SourceFitScope.from_review_artifact(
        scope.review.artifact, scope.review.receipt_bytes
    )
    assert actual == scope
    relative = replace(scope.review.artifact, path="review.json")
    actual_relative = SourceFitScope.from_review_artifact(
        relative, scope.review.receipt_bytes, artifact_root=tmp_path
    )
    assert (
        bind_source_fit_scope(identity, actual_relative, artifact_root=tmp_path).scope
        == actual_relative
    )
    with pytest.raises(ValueError, match="bytes|hash"):
        SourceFitScope.from_review_artifact(
            relative, scope.review.receipt_bytes + 1, artifact_root=tmp_path
        )


def test_scope_factory_rejects_unregistered_receipt_schema(tmp_path: Path) -> None:
    _, scope = make_scope(tmp_path)
    foreign = replace(scope.review.artifact, schema="workspace.handoff/1.0.0")
    with pytest.raises(ValueError, match="schema"):
        SourceFitScope.from_review_artifact(foreign, scope.review.receipt_bytes)
