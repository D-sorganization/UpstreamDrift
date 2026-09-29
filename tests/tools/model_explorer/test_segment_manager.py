"""Tests for src/tools/model_explorer/segment_manager.SegmentManager."""

from __future__ import annotations

import pytest

from src.tools.model_explorer.segment_manager import SegmentManager
from tests.tools.model_explorer._fixtures import make_segment


class TestSegmentManager:
    def setup_method(self) -> None:
        self.m = SegmentManager()

    def test_add_segment(self) -> None:
        self.m.add_segment(make_segment("root"))
        assert self.m.get_segment_count() == 1
        assert self.m.get_segment("root") is not None

    def test_add_duplicate_raises(self) -> None:
        self.m.add_segment(make_segment("a"))
        with pytest.raises(ValueError, match="already exists"):
            self.m.add_segment(make_segment("a"))

    def test_add_with_missing_parent_raises(self) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            self.m.add_segment(make_segment("child", parent="ghost"))

    def test_add_with_missing_name_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            self.m.add_segment({"name": ""})

    def test_remove_segment_recursive(self) -> None:
        self.m.add_segment(make_segment("a"))
        self.m.add_segment(make_segment("b", parent="a"))
        self.m.add_segment(make_segment("c", parent="b"))
        self.m.remove_segment("a")
        assert self.m.get_segment_count() == 0

    def test_remove_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            self.m.remove_segment("ghost")

    def test_modify_segment_reparent(self) -> None:
        self.m.add_segment(make_segment("a"))
        self.m.add_segment(make_segment("b"))
        self.m.add_segment(make_segment("c", parent="a"))
        updated = make_segment("c", parent="b")
        self.m.modify_segment(updated)
        assert "c" in self.m.get_children("b")
        assert "c" not in self.m.get_children("a")

    def test_modify_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            self.m.modify_segment(make_segment("ghost"))

    def test_get_root_segments(self) -> None:
        self.m.add_segment(make_segment("r1"))
        self.m.add_segment(make_segment("r2"))
        self.m.add_segment(make_segment("c", parent="r1"))
        assert set(self.m.get_root_segments()) == {"r1", "r2"}

    def test_hierarchy_order(self) -> None:
        self.m.add_segment(make_segment("r"))
        self.m.add_segment(make_segment("a", parent="r"))
        self.m.add_segment(make_segment("b", parent="a"))
        order = self.m.get_hierarchy_order()
        assert order.index("r") < order.index("a") < order.index("b")

    def test_validate_hierarchy_clean(self) -> None:
        self.m.add_segment(make_segment("r"))
        self.m.add_segment(make_segment("a", parent="r"))
        assert self.m.validate_hierarchy() == []

    def test_validate_hierarchy_orphan(self) -> None:
        # bypass add_segment's pre-check by injecting directly
        self.m.segments["x"] = {"name": "x", "parent": "ghost"}
        errors = self.m.validate_hierarchy()
        assert any("non-existent parent" in e for e in errors)

    def test_validate_hierarchy_cycle(self) -> None:
        # construct cycle by directly editing internal state
        self.m.segments["a"] = {"name": "a", "parent": "b"}
        self.m.segments["b"] = {"name": "b", "parent": "a"}
        self.m.hierarchy["a"] = ["b"]
        self.m.hierarchy["b"] = ["a"]
        errors = self.m.validate_hierarchy()
        assert any("Circular dependency" in e for e in errors)

    def test_create_parallel_chain(self) -> None:
        self.m.add_segment(make_segment("a"))
        self.m.add_segment(make_segment("b"))
        self.m.create_parallel_chain(
            {"name": "loop", "segments": ["a", "b"], "constraints": []}
        )
        assert len(self.m.get_parallel_chains()) == 1
        self.m.remove_parallel_chain("loop")
        assert self.m.get_parallel_chains() == []

    def test_parallel_chain_too_few_segments(self) -> None:
        self.m.add_segment(make_segment("a"))
        with pytest.raises((ValueError, AssertionError)):
            self.m.create_parallel_chain({"name": "loop", "segments": ["a"]})

    def test_parallel_chain_unknown_segment(self) -> None:
        self.m.add_segment(make_segment("a"))
        self.m.add_segment(make_segment("b"))
        with pytest.raises(ValueError, match="does not exist"):
            self.m.create_parallel_chain({"name": "loop", "segments": ["a", "ghost"]})

    def test_remove_unknown_parallel_chain(self) -> None:
        with pytest.raises(ValueError):
            self.m.remove_parallel_chain("ghost")

    def test_clear(self) -> None:
        self.m.add_segment(make_segment("a"))
        self.m.clear()
        assert self.m.get_segment_count() == 0
        assert self.m.get_parallel_chains() == []

    @pytest.mark.parametrize("engine", ["mujoco", "drake", "pinocchio"])
    def test_export_for_engine(self, engine: str) -> None:
        self.m.add_segment(make_segment("a"))
        result = self.m.export_for_engine(engine)
        assert result["engine"] == engine
        assert "segments" in result

    def test_export_for_unknown_engine_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            self.m.export_for_engine("gazebo")
