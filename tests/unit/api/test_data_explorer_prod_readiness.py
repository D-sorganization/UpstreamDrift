"""Production-readiness regressions for data explorer dataset handling."""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from src.api.routes import data_explorer


def _upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(content))


@pytest.fixture(autouse=True)
def clear_loaded_datasets() -> None:
    data_explorer._loaded_datasets.clear()


@pytest.mark.unit
async def test_import_dataset_rejects_duplicate_filename() -> None:
    """Imported datasets must not silently replace an existing name (#3943)."""
    await data_explorer.import_dataset(_upload("sample.csv", b"a\n1\n"))

    with pytest.raises(HTTPException) as excinfo:
        await data_explorer.import_dataset(_upload("sample.csv", b"a\n2\n"))

    assert excinfo.value.status_code == 409


@pytest.mark.unit
async def test_import_dataset_enforces_bounded_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Imported dataset cache has an LRU size ceiling (#3943)."""
    monkeypatch.setattr(data_explorer, "MAX_LOADED_DATASETS", 2)

    await data_explorer.import_dataset(_upload("one.csv", b"a\n1\n"))
    await data_explorer.import_dataset(_upload("two.csv", b"a\n2\n"))
    await data_explorer.preview_dataset("one.csv")
    await data_explorer.import_dataset(_upload("three.csv", b"a\n3\n"))

    assert "one.csv" in data_explorer._loaded_datasets
    assert "two.csv" not in data_explorer._loaded_datasets
    assert "three.csv" in data_explorer._loaded_datasets


@pytest.mark.unit
async def test_preview_dataset_rejects_ambiguous_disk_matches(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate filenames on disk require a disambiguated path (#3943)."""
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "dup.csv").write_text("x\n1\n", encoding="utf-8")
    (second_dir / "dup.csv").write_text("x\n2\n", encoding="utf-8")
    monkeypatch.setattr(data_explorer, "_get_output_dir", lambda: tmp_path)

    with pytest.raises(HTTPException) as excinfo:
        await data_explorer.preview_dataset("dup.csv")

    assert excinfo.value.status_code == 409
