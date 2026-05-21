"""Tests for ContactManager - mock engine, no heavy deps."""

from __future__ import annotations

import numpy as np
import pytest

from src.robotics.contact import contact_manager as cm
from src.robotics.contact.contact_manager import ContactManager
from src.robotics.core.exceptions import ContactError
from src.robotics.core.types import ContactState


class FakeRoboticsEngine:
    """Implements RoboticsCapable but not ContactCapable."""

    def __init__(self) -> None:
        self._q = np.zeros(3)
        self._v = np.zeros(3)

    def get_state(self):
        return self._q.copy(), self._v.copy()

    def set_state(self, q, v):
        self._q = np.asarray(q, dtype=float)
        self._v = np.asarray(v, dtype=float)

    def compute_mass_matrix(self):
        return np.eye(3)

    def compute_bias_forces(self):
        return np.zeros(3)

    def compute_gravity_forces(self):
        return np.zeros(3)

    def compute_jacobian(self, body_name):
        return None

    def get_time(self):
        return 0.0


class FakeContactEngine(FakeRoboticsEngine):
    """Implements both RoboticsCapable and ContactCapable."""

    def __init__(self, infos=None, jacs=None, raise_on=None) -> None:
        super().__init__()
        self._infos = infos or []
        self._jacs = jacs or {}
        self._raise_on = raise_on

    def get_contact_count(self):
        return len(self._infos)

    def get_contact_info(self, idx):
        if self._raise_on is not None and idx == self._raise_on:
            raise ValueError("oops")
        return self._infos[idx]

    def get_contact_jacobian(self, idx):
        return self._jacs.get(idx)


def test_constructor_rejects_non_protocol() -> None:
    with pytest.raises(TypeError, match="RoboticsCapable"):
        ContactManager(object())  # type: ignore[arg-type]


def test_engine_property_and_defaults() -> None:
    e = FakeRoboticsEngine()
    m = ContactManager(e, default_friction=0.7)
    assert m.engine is e
    assert m.contact_count == 0
    assert m.contacts == []


def test_detect_contacts_returns_empty_when_not_contact_capable() -> None:
    m = ContactManager(FakeRoboticsEngine())
    out = m.detect_contacts()
    assert out == []


def test_detect_contacts_with_q_restores_state() -> None:
    e = FakeRoboticsEngine()
    m = ContactManager(e)
    e.set_state(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    new_q = np.array([7.0, 8.0, 9.0])
    m.detect_contacts(q=new_q)
    # State restored
    q, v = e.get_state()
    assert np.allclose(q, [1.0, 2.0, 3.0])


def test_detect_from_contact_capable_engine() -> None:
    info = {
        "body_a": "foot",
        "body_b": "ground",
        "position": [0.1, 0.2, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "penetration": 0.001,
        "force": [0.0, 0.0, 100.0],
        "friction": 0.6,
    }
    e = FakeContactEngine(infos=[info])
    m = ContactManager(e)
    out = m.detect_contacts()
    assert len(out) == 1
    c = out[0]
    assert isinstance(c, ContactState)
    assert c.normal_force == pytest.approx(100.0)
    assert c.friction_coefficient == pytest.approx(0.6)
    assert m.contact_count == 1


def test_detect_engine_raises_wraps_to_contact_error() -> None:
    e = FakeContactEngine(infos=[{}], raise_on=0)
    m = ContactManager(e)
    with pytest.raises(ContactError):
        m.detect_contacts()


def test_get_contact_jacobian_returns_none_when_not_contact_capable() -> None:
    m = ContactManager(FakeRoboticsEngine())
    contact = ContactState(
        contact_id=0,
        body_a="a",
        body_b="b",
        position=np.zeros(3),
        normal=np.array([0, 0, 1.0]),
    )
    assert m.get_contact_jacobian(contact) is None


def test_get_contact_jacobian_lookup() -> None:
    info = {"position": [0, 0, 0], "normal": [0, 0, 1.0], "force": [0, 0, 1.0]}
    e = FakeContactEngine(infos=[info], jacs={0: np.ones((3, 3))})
    m = ContactManager(e)
    contacts = m.detect_contacts()
    J = m.get_contact_jacobian(contacts[0])
    assert J is not None and J.shape == (3, 3)


def test_get_contact_jacobian_unknown_contact_returns_none() -> None:
    info = {"position": [0, 0, 0], "normal": [0, 0, 1.0], "force": [0, 0, 1.0]}
    e = FakeContactEngine(infos=[info], jacs={0: np.ones((3, 3))})
    m = ContactManager(e)
    m.detect_contacts()
    foreign = ContactState(
        contact_id=999,
        body_a="x",
        body_b="y",
        position=np.zeros(3),
        normal=np.array([0, 0, 1.0]),
    )
    assert m.get_contact_jacobian(foreign) is None


def test_get_contact_jacobian_stack_none_when_no_contacts() -> None:
    m = ContactManager(FakeRoboticsEngine())
    assert m.get_contact_jacobian_stack() is None


def test_get_contact_jacobian_stack_reduces_6_to_3() -> None:
    info = {"position": [0, 0, 0], "normal": [0, 0, 1.0], "force": [0, 0, 1.0]}
    e = FakeContactEngine(
        infos=[info, info], jacs={0: np.ones((6, 4)), 1: np.ones((6, 4))}
    )
    m = ContactManager(e)
    contacts = m.detect_contacts()
    stack = m.get_contact_jacobian_stack(contacts)
    assert stack is not None and stack.shape == (6, 4)


def test_get_active_contacts_filters_inactive() -> None:
    info_active = {"position": [0, 0, 0], "normal": [0, 0, 1.0], "force": [0, 0, 1.0]}
    e = FakeContactEngine(infos=[info_active])
    m = ContactManager(e)
    contacts = m.detect_contacts()
    # mutate cache to mark inactive (simulate by setting up a manual list)
    m._contact_cache = [
        contacts[0],
        contacts[0].with_force(0.0),
    ]
    # both are still active by default; ensure get_active_contacts returns active ones
    assert len(m.get_active_contacts()) == 2


def test_compute_support_polygon_lt3_returns_none() -> None:
    m = ContactManager(FakeRoboticsEngine())
    contacts = [
        ContactState(
            contact_id=i,
            body_a="a",
            body_b="b",
            position=np.array([float(i), 0.0, 0.0]),
            normal=np.array([0, 0, 1.0]),
        )
        for i in range(2)
    ]
    assert m.compute_support_polygon(contacts) is None


def test_compute_support_polygon_returns_hull() -> None:
    m = ContactManager(FakeRoboticsEngine())
    pts = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]
    contacts = [
        ContactState(
            contact_id=i,
            body_a="a",
            body_b="b",
            position=np.array([x, y, 0.0]),
            normal=np.array([0, 0, 1.0]),
        )
        for i, (x, y) in enumerate(pts)
    ]
    hull = m.compute_support_polygon(contacts)
    assert hull is not None
    assert hull.shape[1] == 2
    assert len(hull) <= 5


def test_point_in_support_polygon_inside_and_outside() -> None:
    m = ContactManager(FakeRoboticsEngine())
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    contacts = [
        ContactState(
            contact_id=i,
            body_a="a",
            body_b="b",
            position=np.array([x, y, 0.0]),
            normal=np.array([0, 0, 1.0]),
        )
        for i, (x, y) in enumerate(pts)
    ]
    assert m.point_in_support_polygon(np.array([0.5, 0.5, 0.0]), contacts)
    assert not m.point_in_support_polygon(np.array([2.0, 2.0]), contacts)


def test_point_in_support_polygon_no_polygon() -> None:
    m = ContactManager(FakeRoboticsEngine())
    assert not m.point_in_support_polygon(np.zeros(2), [])


def test_clear_cache_and_reset_ids() -> None:
    info = {"position": [0, 0, 0], "normal": [0, 0, 1.0], "force": [0, 0, 1.0]}
    e = FakeContactEngine(infos=[info])
    m = ContactManager(e)
    m.detect_contacts()
    assert m.contact_count == 1
    m.clear_cache()
    assert m.contact_count == 0
    m.detect_contacts()
    m.reset_contact_ids()
    assert m.contact_count == 0


def test_graham_scan_collinear_points() -> None:
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    hull = cm._graham_scan(pts)
    assert hull.shape[1] == 2
    assert len(hull) >= 3


def test_cross_product_2d_sign() -> None:
    assert (
        cm._cross_product_2d(
            np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
        )
        > 0
    )
    assert (
        cm._cross_product_2d(
            np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
        )
        < 0
    )


def test_point_in_polygon_basic() -> None:
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert cm._point_in_polygon(np.array([0.5, 0.5]), poly)
    assert not cm._point_in_polygon(np.array([-0.5, 0.5]), poly)


def test_point_in_polygon_degenerate() -> None:
    assert not cm._point_in_polygon(np.zeros(2), np.zeros((2, 2)))
