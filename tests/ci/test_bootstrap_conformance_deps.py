"""Unit tests for conformance dependency bootstrap under offline/no-index CI."""

from __future__ import annotations

from pathlib import Path
import pytest

from scripts.ci import bootstrap_conformance_deps as bootstrap

pytestmark = pytest.mark.unit


def test_conformance_canonical_pins() -> None:
    """Canonical pins must match repository standard exact versions."""
    assert bootstrap.CANONICAL_PINS["pydantic"] == "2.12.5"
    assert bootstrap.CANONICAL_PINS["numpy"] == "2.2.6"
    assert bootstrap.CANONICAL_PINS["scipy"] == "1.14.1"


def test_build_install_args_online() -> None:
    """Online mode generates pip install command with canonical pins."""
    args = bootstrap.build_pip_install_args(
        no_index=False,
        find_links=None,
    )
    assert "pydantic==2.12.5" in args
    assert "numpy==2.2.6" in args
    assert "scipy==1.14.1" in args
    assert "--no-index" not in args


def test_build_install_args_offline_with_find_links() -> None:
    """Offline mode includes --no-index and --find-links."""
    args = bootstrap.build_pip_install_args(
        no_index=True,
        find_links=Path("/tmp/wheels"),
    )
    assert "--no-index" in args
    assert "--find-links" in args
    assert "/tmp/wheels" in args or str(Path("/tmp/wheels")) in args
    assert "pydantic==2.12.5" in args


def test_offline_mode_fails_closed_when_wheels_missing(tmp_path: Path) -> None:
    """Offline mode raises explicit missing-artifact diagnostic when wheels are missing."""
    empty_wheel_dir = tmp_path / "wheels"
    empty_wheel_dir.mkdir()

    with pytest.raises(
        bootstrap.MissingArtifactError,
        match="Missing approved wheel artifact for pydantic==2.12.5",
    ):
        bootstrap.verify_offline_artifacts(
            wheel_dir=empty_wheel_dir,
            required_pins={"pydantic": "2.12.5", "numpy": "2.2.6", "scipy": "1.14.1"},
        )


def test_offline_mode_passes_when_all_wheels_present(tmp_path: Path) -> None:
    """Offline mode passes artifact verification when all required wheels exist."""
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    (wheel_dir / "pydantic-2.12.5-py3-none-any.whl").write_bytes(b"pydantic")
    (wheel_dir / "numpy-2.2.6-cp311-cp311-manylinux.whl").write_bytes(b"numpy")
    (wheel_dir / "scipy-1.14.1-cp311-cp311-manylinux.whl").write_bytes(b"scipy")

    missing = bootstrap.verify_offline_artifacts(
        wheel_dir=wheel_dir,
        required_pins={"pydantic": "2.12.5", "numpy": "2.2.6", "scipy": "1.14.1"},
    )
    assert missing == []
