"""Unit tests for grip contact extraction and slider value mapping.

Addresses the contact-pipeline + slider test gap from issue #7724. These
methods are pure with respect to ``self`` (they only use their arguments), so
they are exercised on a bare instance created via ``object.__new__`` without
constructing the full PyQt6 tab.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.grip_modelling_tab import (
    GripModellingTab,
)

# A tiny scene: a "hand"-named body falls onto the floor, producing a real
# MuJoCo contact whose body attribution we can assert on.
_HAND_CONTACT_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="hand_palm" pos="0 0 0.04">
      <freejoint/>
      <geom name="g_hand" type="sphere" size="0.05"/>
    </body>
    <geom name="g_floor" type="plane" size="1 1 0.1"/>
  </worldbody>
</mujoco>
"""

# A scene with no hand/finger body names: contacts must be filtered out.
_NO_HAND_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="block" pos="0 0 0.04">
      <freejoint/>
      <geom name="g_block" type="sphere" size="0.05"/>
    </body>
    <geom name="g_floor" type="plane" size="1 1 0.1"/>
  </worldbody>
</mujoco>
"""


class _FakeTab:
    """Stand-in ``self`` for the pure GripModellingTab methods under test.

    The methods exercised here (``_extract_hand_contacts``,
    ``_val_to_slider``, ``_slider_to_val``) never touch instance attributes,
    so an empty object is a sufficient ``self`` and avoids constructing a real
    QWidget (which requires a running Qt app).
    """

    _extract_hand_contacts = GripModellingTab._extract_hand_contacts
    _val_to_slider = GripModellingTab._val_to_slider
    _slider_to_val = GripModellingTab._slider_to_val


def _bare_tab() -> _FakeTab:
    """Create a stand-in tab without running QWidget initialisation."""
    return _FakeTab()


def _settle(xml: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)
    return model, data


@pytest.mark.unit
def test_extract_hand_contacts_attributes_hand_body() -> None:
    """A contact involving a hand-named body is captured and attributed."""
    model, data = _settle(_HAND_CONTACT_XML)
    assert data.ncon > 0, "expected the sphere to contact the floor"

    tab = _bare_tab()
    positions, normals, forces, velocities, body_names = tab._extract_hand_contacts(
        model, data
    )

    # The hand sphere vs floor contact qualifies as a hand contact (hand_palm
    # is geom2's body), so every contact is captured.
    assert len(positions) == data.ncon
    assert len(positions) == len(normals) == len(forces) == len(body_names)
    # Behaviour note (issue #7724): attribution names *geom1's* body, which for
    # a falling-onto-floor contact is "world" -- the known body-name
    # misattribution. We pin the current behaviour rather than the ideal one.
    assert all(name == "world" for name in body_names)
    # Forces are 3-vectors; velocities default to zeros (dead slip path).
    assert all(np.asarray(f).shape == (3,) for f in forces)
    assert all(np.allclose(v, 0.0) for v in velocities)


@pytest.mark.unit
def test_extract_hand_contacts_filters_non_hand() -> None:
    """Contacts with no hand/finger body are excluded."""
    model, data = _settle(_NO_HAND_XML)
    assert data.ncon > 0, "expected the block to contact the floor"

    tab = _bare_tab()
    positions, normals, forces, velocities, body_names = tab._extract_hand_contacts(
        model, data
    )

    assert positions == []
    assert body_names == []


@pytest.mark.unit
def test_extract_hand_contacts_requires_model() -> None:
    """A None model is a contract violation."""
    tab = _bare_tab()
    with pytest.raises(ValueError, match="model must be provided"):
        tab._extract_hand_contacts(None, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# slider <-> value round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("val", [-1.0, -0.25, 0.0, 0.5, 1.25])
def test_val_to_slider_round_trip(val: float) -> None:
    """_slider_to_val(_val_to_slider(v)) recovers v within slider resolution."""
    tab = _bare_tab()
    min_v, max_v = -1.5, 1.5
    slider = tab._val_to_slider(val, min_v, max_v)
    assert 0 <= slider <= 1000
    recovered = tab._slider_to_val(slider, min_v, max_v)
    # 1000-step slider over a range of 3.0 -> resolution 0.003.
    assert recovered == pytest.approx(val, abs=0.004)


@pytest.mark.unit
def test_val_to_slider_zero_range_is_midpoint() -> None:
    """A degenerate min==max range maps to the slider midpoint (ratio 0.5)."""
    tab = _bare_tab()
    assert tab._val_to_slider(5.0, 2.0, 2.0) == 500


@pytest.mark.unit
def test_val_to_slider_bounds() -> None:
    """Min/max values map to the slider extremes."""
    tab = _bare_tab()
    assert tab._val_to_slider(-1.5, -1.5, 1.5) == 0
    assert tab._val_to_slider(1.5, -1.5, 1.5) == 1000


@pytest.mark.unit
def test_slider_to_val_extremes() -> None:
    """Slider extremes map back to min/max values."""
    tab = _bare_tab()
    assert tab._slider_to_val(0, -1.5, 1.5) == pytest.approx(-1.5)
    assert tab._slider_to_val(1000, -1.5, 1.5) == pytest.approx(1.5)


@pytest.mark.unit
def test_val_to_slider_requires_value() -> None:
    """A None value is a contract violation."""
    tab = _bare_tab()
    with pytest.raises(ValueError, match="val must be provided"):
        tab._val_to_slider(None, 0.0, 1.0)  # type: ignore[arg-type]


@pytest.mark.unit
def test_slider_to_val_requires_value() -> None:
    """A None slider value is a contract violation."""
    tab = _bare_tab()
    with pytest.raises(ValueError, match="slider_val must be provided"):
        tab._slider_to_val(None, 0.0, 1.0)  # type: ignore[arg-type]
