"""Tests for the _load_engine_with_probe factory function.

These tests validate the DRY factory that underpins all 7 engine loaders.
Covers the shared probe → engine-create → model-load → DbC-postcondition path.

TDD anchors
-----------
- Successful path: probe available, engine created, model loaded.
- Probe failure: raises GolfModelingError with diagnostic message.
- Model file missing: warns but does NOT fail (graceful degradation).
- Model load error: warns but does NOT fail (graceful degradation).
- DbC postcondition: factory raises if engine_factory returns None.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.engines.loaders import _ensure_engine_loaded, _load_engine_with_probe
from src.shared.python.data_io.common_utils import GolfModelingError


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    """Minimal suite root for factory tests."""
    return tmp_path


# ---------------------------------------------------------------------------
# _ensure_engine_loaded
# ---------------------------------------------------------------------------


def test_ensure_engine_loaded_passes_for_non_none() -> None:
    """DbC helper passes silently when engine is non-None."""
    mock_engine = MagicMock()
    _ensure_engine_loaded(mock_engine, "TestEngine")  # Must not raise


def test_ensure_engine_loaded_raises_for_none() -> None:
    """DbC helper raises GolfModelingError if engine is None."""
    with pytest.raises(GolfModelingError, match="DbC postcondition violated"):
        _ensure_engine_loaded(None, "TestEngine")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _load_engine_with_probe — happy path
# ---------------------------------------------------------------------------


def _make_probe_cls(*, available: bool = True) -> MagicMock:
    """Return a probe class mock whose probe() result reflects availability."""
    probe_cls = MagicMock()
    probe_instance = probe_cls.return_value
    probe_result = MagicMock()
    probe_result.is_available.return_value = available
    if not available:
        probe_result.diagnostic_message = "Engine not found"
        probe_result.get_fix_instructions.return_value = "pip install engine"
    probe_instance.probe.return_value = probe_result
    return probe_cls


def test_factory_returns_engine_on_success(tmp_root: Path) -> None:
    """Factory returns the engine when probe is available."""
    mock_engine = MagicMock()
    probe_cls = _make_probe_cls(available=True)
    engine_factory = MagicMock(return_value=mock_engine)

    result = _load_engine_with_probe(
        engine_name="TestEngine",
        probe_factory=probe_cls,
        engine_factory=engine_factory,
        model_path_fn=None,
        load_model=False,
        suite_root=tmp_root,
    )

    assert result is mock_engine
    probe_cls.assert_called_once_with(tmp_root)
    engine_factory.assert_called_once()


def test_factory_loads_model_when_exists(tmp_root: Path) -> None:
    """Factory calls engine.load_from_path() when model file is present."""
    model_file = tmp_root / "model.xml"
    model_file.write_text("<model/>")

    mock_engine = MagicMock()
    probe_cls = _make_probe_cls(available=True)

    _load_engine_with_probe(
        engine_name="TestEngine",
        probe_factory=probe_cls,
        engine_factory=MagicMock(return_value=mock_engine),
        model_path_fn=lambda _root: model_file,
        load_model=True,
        suite_root=tmp_root,
    )

    mock_engine.load_from_path.assert_called_once_with(str(model_file))


def test_factory_skips_model_load_when_disabled(tmp_root: Path) -> None:
    """Factory does not call load_from_path when load_model=False."""
    model_file = tmp_root / "model.xml"
    model_file.write_text("<model/>")

    mock_engine = MagicMock()

    _load_engine_with_probe(
        engine_name="TestEngine",
        probe_factory=_make_probe_cls(available=True),
        engine_factory=MagicMock(return_value=mock_engine),
        model_path_fn=lambda _root: model_file,
        load_model=False,
        suite_root=tmp_root,
    )

    mock_engine.load_from_path.assert_not_called()


def test_factory_warns_when_model_missing(tmp_root: Path) -> None:
    """Factory logs a warning (does not raise) when model file is absent."""
    mock_engine = MagicMock()

    # Should not raise even though the path does not exist
    result = _load_engine_with_probe(
        engine_name="TestEngine",
        probe_factory=_make_probe_cls(available=True),
        engine_factory=MagicMock(return_value=mock_engine),
        model_path_fn=lambda _root: _root / "nonexistent.xml",
        load_model=True,
        suite_root=tmp_root,
    )

    assert result is mock_engine
    mock_engine.load_from_path.assert_not_called()


def test_factory_warns_on_model_load_error(tmp_root: Path) -> None:
    """Factory logs a warning (does not raise) when load_from_path raises."""
    model_file = tmp_root / "model.xml"
    model_file.write_text("<model/>")

    mock_engine = MagicMock()
    mock_engine.load_from_path.side_effect = RuntimeError("mesh not found")

    result = _load_engine_with_probe(
        engine_name="TestEngine",
        probe_factory=_make_probe_cls(available=True),
        engine_factory=MagicMock(return_value=mock_engine),
        model_path_fn=lambda _root: model_file,
        load_model=True,
        suite_root=tmp_root,
    )

    assert result is mock_engine


# ---------------------------------------------------------------------------
# _load_engine_with_probe — failure paths
# ---------------------------------------------------------------------------


def test_factory_raises_when_probe_fails(tmp_root: Path) -> None:
    """Factory raises GolfModelingError when probe reports unavailability."""
    with pytest.raises(GolfModelingError, match="TestEngine not ready"):
        _load_engine_with_probe(
            engine_name="TestEngine",
            probe_factory=_make_probe_cls(available=False),
            engine_factory=MagicMock(),
            model_path_fn=None,
            load_model=False,
            suite_root=tmp_root,
        )


def test_factory_raises_dbc_postcondition_when_engine_is_none(tmp_root: Path) -> None:
    """Factory raises GolfModelingError when engine_factory returns None."""
    with pytest.raises(GolfModelingError, match="DbC postcondition violated"):
        _load_engine_with_probe(
            engine_name="TestEngine",
            probe_factory=_make_probe_cls(available=True),
            engine_factory=MagicMock(return_value=None),  # type: ignore[arg-type]
            model_path_fn=None,
            load_model=False,
            suite_root=tmp_root,
        )
