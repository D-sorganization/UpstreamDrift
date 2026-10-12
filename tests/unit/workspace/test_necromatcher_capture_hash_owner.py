"""Capture identity reuses the public canonical raw SHA256 validator."""

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit


def test_curated_sha_validator_is_existing_owner() -> None:
    from src.shared.python.shadow_tracker import check_sha256
    from src.shared.python.shadow_tracker import _validation

    assert check_sha256 is _validation.check_sha256
    assert check_sha256("a" * 64, "raw_hash") == "a" * 64


@pytest.mark.parametrize("value", [True, [], "A" * 64, "sha256:" + "a" * 64, "a" * 63])
def test_curated_sha_validator_retains_raw_domain(value: Any) -> None:
    from src.shared.python.shadow_tracker import check_sha256

    with pytest.raises((ValueError, TypeError)):
        check_sha256(value, "raw_hash")


def test_shadow_facade_and_validator_are_fingerprinted(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from src.shared.python.workspace import necromatcher_fit_jobs as jobs

    directory = tmp_path / "src/shared/python/shadow_tracker"
    directory.mkdir(parents=True)
    facade, validator = directory / "__init__.py", directory / "_validation.py"
    facade.write_text("# facade", encoding="utf-8")
    validator.write_text("# validation", encoding="utf-8")
    monkeypatch.setattr(jobs, "get_repo_root", lambda: tmp_path)
    first = jobs.fit_execution_stamp()
    assert facade.relative_to(tmp_path).as_posix() in first["source_files"]
    assert validator.relative_to(tmp_path).as_posix() in first["source_files"]
    facade.write_text("# facade changed", encoding="utf-8")
    second = jobs.fit_execution_stamp()
    validator.write_text("# validation changed", encoding="utf-8")
    third = jobs.fit_execution_stamp()
    assert first["source_sha256"] != second["source_sha256"]
    assert second["source_sha256"] != third["source_sha256"]
