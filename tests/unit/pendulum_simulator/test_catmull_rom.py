"""Tests for src.shared.python.pendulum_simulator.gui.catmull_rom (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.pendulum_simulator.gui.catmull_rom import catmull_rom_smooth


class TestCatmullRomSmooth:
    def test_empty_returns_empty(self) -> None:
        result = catmull_rom_smooth([])
        assert result == []

    def test_less_than_4_returns_unchanged(self) -> None:
        points = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        result = catmull_rom_smooth(points)
        assert result == points

    def test_exactly_1_point_returns_unchanged(self) -> None:
        points = [(5.0, 3.0)]
        result = catmull_rom_smooth(points)
        assert result == points

    def test_4_points_returns_more_than_4(self) -> None:
        points = [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0)]
        result = catmull_rom_smooth(points, n_sub=4)
        assert len(result) > len(points)

    def test_last_point_preserved(self) -> None:
        points = [(0.0, 0.0), (1.0, 2.0), (2.0, 1.0), (3.0, 0.5)]
        result = catmull_rom_smooth(points, n_sub=4)
        assert result[-1] == pytest.approx(points[-1])

    def test_n_sub_1_minimal_interpolation(self) -> None:
        points = [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0)]
        result = catmull_rom_smooth(points, n_sub=1)
        # With n_sub=1, one point per segment + final point
        assert len(result) >= len(points)

    def test_n_sub_affects_output_length(self) -> None:
        points = [(float(i), float(i % 2)) for i in range(5)]
        result_4 = catmull_rom_smooth(points, n_sub=4)
        result_8 = catmull_rom_smooth(points, n_sub=8)
        assert len(result_8) > len(result_4)

    def test_zero_n_sub_raises(self) -> None:
        with pytest.raises(ValueError, match="n_sub must be >= 1"):
            catmull_rom_smooth(
                [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)], n_sub=0
            )

    def test_straight_line_stays_straight(self) -> None:
        # For a straight line, Catmull-Rom should stay on the line
        points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        result = catmull_rom_smooth(points, n_sub=4)
        for x, y in result:
            # On y=x line
            assert abs(y - x) < 1e-10

    def test_returns_list_of_tuples(self) -> None:
        points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        result = catmull_rom_smooth(points, n_sub=2)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_5_points_output_length(self) -> None:
        points = [(float(i), 0.0) for i in range(5)]
        result = catmull_rom_smooth(points, n_sub=4)
        # Each segment (between control points) produces n_sub points, plus final
        # Segments = 5 - 1 = 4 (with padding), each produces 4 points + 1 final
        assert len(result) == 4 * 4 + 1  # 17

    def test_preserves_coordinate_types(self) -> None:
        points = [(0.0, 0.0), (1.5, 2.5), (3.0, 1.0), (4.5, 0.5)]
        result = catmull_rom_smooth(points, n_sub=2)
        for x, y in result:
            assert isinstance(x, float)
            assert isinstance(y, float)
