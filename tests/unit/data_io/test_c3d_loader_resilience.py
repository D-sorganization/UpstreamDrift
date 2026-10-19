"""Tests verifying c3d_loader resilience against test pollution and aliased namespaces (#13348)."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from src.shared.python.data_io.c3d_loader import (
    c3d_reader_class,
    marker_set_types,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_marker_set_types_resilient_to_polluted_bio_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """marker_set_types() loads successfully even when sidekick.lab.bio in sys.modules lacks _c3d_marker_set."""
    # Simulate a polluted or partial Tools package without _c3d_marker_set
    dummy_bio = types.ModuleType("src.shared.python.sidekick.lab.bio")
    dummy_bio.__path__ = ["/nonexistent/tools/path"]  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "src.shared.python.sidekick.lab.bio", dummy_bio)
    monkeypatch.setitem(sys.modules, "shared.python.sidekick.lab.bio", dummy_bio)
    monkeypatch.setitem(sys.modules, "sidekick.lab.bio", dummy_bio)
    monkeypatch.delitem(
        sys.modules, "src.shared.python.sidekick.lab.bio._c3d_marker_set", raising=False
    )
    monkeypatch.delitem(
        sys.modules, "shared.python.sidekick.lab.bio._c3d_marker_set", raising=False
    )
    monkeypatch.delitem(sys.modules, "sidekick.lab.bio._c3d_marker_set", raising=False)

    marker_set, mismatch_err = marker_set_types()

    assert marker_set is not None
    assert issubclass(mismatch_err, Exception)
    assert (
        hasattr(marker_set, "DRIVER_QUALIFICATION")
        or hasattr(marker_set, "name")
        or len(marker_set) > 0
    )


def test_c3d_reader_class_resilient_to_polluted_bio_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """c3d_reader_class() loads successfully even when sidekick.lab.bio in sys.modules lacks c3d_reader."""
    dummy_bio = types.ModuleType("src.shared.python.sidekick.lab.bio")
    dummy_bio.__path__ = ["/nonexistent/tools/path"]  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "src.shared.python.sidekick.lab.bio", dummy_bio)
    monkeypatch.setitem(sys.modules, "shared.python.sidekick.lab.bio", dummy_bio)
    monkeypatch.setitem(sys.modules, "sidekick.lab.bio", dummy_bio)
    monkeypatch.delitem(
        sys.modules, "src.shared.python.sidekick.lab.bio.c3d_reader", raising=False
    )
    monkeypatch.delitem(
        sys.modules, "shared.python.sidekick.lab.bio.c3d_reader", raising=False
    )
    monkeypatch.delitem(sys.modules, "sidekick.lab.bio.c3d_reader", raising=False)

    reader_cls = c3d_reader_class()
    assert reader_cls is not None
    assert reader_cls.__name__ == "C3DDataReader"
