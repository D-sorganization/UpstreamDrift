"""Tests for :mod:`training.identifiers`."""

from __future__ import annotations

import dataclasses

import pytest

from training import (
    MAX_ID_LENGTH,
    JobId,
    RunId,
    TrainingConfigError,
    new_job_id,
    new_run_id,
)

pytestmark = pytest.mark.unit


class TestJobId:
    def test_construct_with_valid_value(self) -> None:
        job_id = JobId("alpha-1_beta")
        assert job_id.value == "alpha-1_beta"
        assert str(job_id) == "alpha-1_beta"

    def test_is_frozen(self) -> None:
        job_id = JobId("abc123")
        with pytest.raises(dataclasses.FrozenInstanceError):
            job_id.value = "different"  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        assert JobId("same") == JobId("same")
        assert JobId("same") != JobId("other")

    def test_hashable(self) -> None:
        # Used as dict keys / set members
        ids = {JobId("a"), JobId("b"), JobId("a")}
        assert len(ids) == 2

    def test_orderable_for_stable_listings(self) -> None:
        a = JobId("aaa")
        b = JobId("bbb")
        assert a < b

    @pytest.mark.parametrize(
        "bad_value",
        ["", " ", "with space", "punc!", "slash/path", "dot.dot"],
    )
    def test_rejects_invalid_characters(self, bad_value: str) -> None:
        with pytest.raises(TrainingConfigError):
            JobId(bad_value)

    def test_rejects_overlong(self) -> None:
        with pytest.raises(TrainingConfigError):
            JobId("a" * (MAX_ID_LENGTH + 1))

    def test_accepts_max_length(self) -> None:
        JobId("a" * MAX_ID_LENGTH)  # boundary

    def test_rejects_non_string(self) -> None:
        with pytest.raises(TypeError):
            JobId(123)  # type: ignore[arg-type]


class TestRunId:
    def test_construct_with_valid_value(self) -> None:
        run_id = RunId("run_001")
        assert run_id.value == "run_001"
        assert str(run_id) == "run_001"

    def test_jobid_and_runid_are_distinct_types(self) -> None:
        """Nominal typing — equality across the two opaque types is False."""
        assert JobId("same") != RunId("same")

    def test_rejects_empty(self) -> None:
        with pytest.raises(TrainingConfigError):
            RunId("")


class TestFactories:
    def test_new_job_id_returns_jobid(self) -> None:
        result = new_job_id()
        assert isinstance(result, JobId)

    def test_new_run_id_returns_runid(self) -> None:
        result = new_run_id()
        assert isinstance(result, RunId)

    def test_new_ids_are_unique(self) -> None:
        ids = {new_job_id().value for _ in range(100)}
        assert len(ids) == 100

    def test_factory_output_passes_validation(self) -> None:
        # Round-trip: factory output must be reconstructible.
        original = new_job_id()
        rebuilt = JobId(original.value)
        assert rebuilt == original
