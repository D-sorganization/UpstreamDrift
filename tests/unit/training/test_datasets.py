"""Tests for :mod:`training.datasets`."""

from __future__ import annotations

from pathlib import Path

import pytest

from training import (
    Dataset,
    DatasetRegistry,
    DuplicateJobError,
    JobNotFoundError,
    TrainingConfigError,
)

pytestmark = pytest.mark.unit


class TestDatasetConstruction:
    def test_minimal(self) -> None:
        ds = Dataset(
            dataset_id="ds-001",
            name="alpha",
            path=Path("/tmp/data"),
            format="csv",
        )
        assert ds.dataset_id == "ds-001"
        assert ds.format == "csv"
        assert ds.size_bytes == 0
        assert ds.schema_version == 1
        assert ds.description == ""

    def test_lowercase_format(self) -> None:
        ds = Dataset(
            dataset_id="ds-001",
            name="alpha",
            path=Path("/tmp/data"),
            format="CSV",
        )
        assert ds.format == "csv"

    @pytest.mark.parametrize("bad_format", ["xml", "exe", "", "custom2"])
    def test_rejects_unknown_format(self, bad_format: str) -> None:
        with pytest.raises(TrainingConfigError, match="format"):
            Dataset(
                dataset_id="ds-001",
                name="alpha",
                path=Path("/tmp/data"),
                format=bad_format,
            )

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(TrainingConfigError):
            Dataset(
                dataset_id="",
                name="alpha",
                path=Path("/tmp/data"),
                format="csv",
            )

    def test_rejects_negative_size(self) -> None:
        with pytest.raises(TrainingConfigError):
            Dataset(
                dataset_id="ds-001",
                name="alpha",
                path=Path("/tmp/data"),
                format="csv",
                size_bytes=-1,
            )

    def test_rejects_non_path(self) -> None:
        with pytest.raises(TrainingConfigError):
            Dataset(
                dataset_id="ds-001",
                name="alpha",
                path="/tmp/data",  # type: ignore[arg-type]
                format="csv",
            )

    def test_exists_when_path_present(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n", encoding="utf-8")
        ds = Dataset(
            dataset_id="ds-001",
            name="alpha",
            path=data_file,
            format="csv",
        )
        assert ds.exists() is True

    def test_does_not_exist_otherwise(self) -> None:
        ds = Dataset(
            dataset_id="ds-001",
            name="alpha",
            path=Path("/tmp/does-not-exist-zzz"),
            format="csv",
        )
        assert ds.exists() is False


class TestDatasetRegistry:
    def _make(self, dataset_id: str = "ds-001", fmt: str = "csv") -> Dataset:
        return Dataset(
            dataset_id=dataset_id,
            name=f"name-{dataset_id}",
            path=Path(f"/tmp/{dataset_id}"),
            format=fmt,
        )

    def test_register_and_get(self) -> None:
        registry = DatasetRegistry()
        ds = self._make()
        registry.register(ds)
        assert registry.get("ds-001") == ds

    def test_initial_dataset_iterable(self) -> None:
        ds = self._make()
        registry = DatasetRegistry(initial=(ds,))
        assert registry.has("ds-001")

    def test_duplicate_raises(self) -> None:
        registry = DatasetRegistry(initial=(self._make(),))
        with pytest.raises(DuplicateJobError):
            registry.register(self._make())

    def test_get_missing_raises(self) -> None:
        registry = DatasetRegistry()
        with pytest.raises(JobNotFoundError):
            registry.get("missing")

    def test_remove(self) -> None:
        ds = self._make()
        registry = DatasetRegistry(initial=(ds,))
        removed = registry.remove("ds-001")
        assert removed == ds
        assert not registry.has("ds-001")

    def test_remove_missing_raises(self) -> None:
        with pytest.raises(JobNotFoundError):
            DatasetRegistry().remove("missing")

    def test_replace_returns_previous(self) -> None:
        original = self._make()
        replacement = Dataset(
            dataset_id="ds-001",
            name="updated",
            path=Path("/tmp/ds-001"),
            format="csv",
        )
        registry = DatasetRegistry(initial=(original,))
        previous = registry.replace(replacement)
        assert previous == original
        assert registry.get("ds-001").name == "updated"

    def test_list_filters_by_format(self) -> None:
        a = self._make("a", "csv")
        b = self._make("b", "parquet")
        c = self._make("c", "csv")
        registry = DatasetRegistry(initial=(a, b, c))
        csv_only = registry.list(format="csv")
        assert set(csv_only) == {a, c}
        assert set(registry.list()) == {a, b, c}

    def test_len_and_iter(self) -> None:
        a = self._make("a")
        b = self._make("b", "json")
        registry = DatasetRegistry(initial=(a, b))
        assert len(registry) == 2
        assert set(registry) == {a, b}
