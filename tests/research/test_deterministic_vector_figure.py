"""Deterministic vector-figure persistence for governed research evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.research.proximal_distal_energy.deterministic_vector_figure import (
    save_vector_figure,
)


pytestmark = pytest.mark.unit


def _figure() -> plt.Figure:
    figure, axis = plt.subplots(figsize=(4.0, 3.0))
    x = np.linspace(0.0, 1.0, 11)
    axis.plot(x, x * x, label="Quadratic")
    axis.legend()
    return figure


def test_pdf_and_svg_bytes_repeat_exactly(tmp_path: Path) -> None:
    first = _figure()
    second = _figure()

    save_vector_figure(first, tmp_path / "first", salt="fixture-v1")
    save_vector_figure(second, tmp_path / "second", salt="fixture-v1")
    plt.close(first)
    plt.close(second)

    assert (tmp_path / "first.pdf").read_bytes() == (
        tmp_path / "second.pdf"
    ).read_bytes()
    assert (tmp_path / "first.svg").read_bytes() == (
        tmp_path / "second.svg"
    ).read_bytes()
    svg = (tmp_path / "first.svg").read_text(encoding="utf-8")
    assert "<dc:date>" not in svg


def test_pdf_only_atomic_write_is_repeatable(tmp_path: Path) -> None:
    first = _figure()
    second = _figure()

    save_vector_figure(
        first,
        tmp_path / "first.pdf",
        salt="fixture-pdf-v1",
        write_svg=False,
        atomic_pdf=True,
    )
    save_vector_figure(
        second,
        tmp_path / "second.pdf",
        salt="fixture-pdf-v1",
        write_svg=False,
        atomic_pdf=True,
    )
    plt.close(first)
    plt.close(second)

    assert (tmp_path / "first.pdf").read_bytes() == (
        tmp_path / "second.pdf"
    ).read_bytes()
