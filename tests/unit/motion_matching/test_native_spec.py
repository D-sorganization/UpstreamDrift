"""Tree construction must reject topology changes rather than dropping joints."""

import pytest

from src.shared.python.motion_matching.native_spec import order_native_tree


def test_orders_dependencies_without_dropping_any_joint() -> None:
    edges = [
        {"name": "wrist", "parent": "arm", "child": "hand"},
        {"name": "shoulder", "parent": "world", "child": "arm"},
    ]
    assert [edge["name"] for edge in order_native_tree(edges)] == ["shoulder", "wrist"]


def test_rejects_closed_or_disconnected_tree() -> None:
    with pytest.raises(ValueError, match="cycle|disconnected"):
        order_native_tree(
            [
                {"name": "a", "parent": "b", "child": "c"},
                {"name": "b", "parent": "c", "child": "b"},
            ]
        )
    with pytest.raises(ValueError, match="multiple parents"):
        order_native_tree(
            [
                {"name": "a", "parent": "world", "child": "b"},
                {"name": "b", "parent": "world", "child": "b"},
            ]
        )
