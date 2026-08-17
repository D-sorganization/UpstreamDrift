"""Regression tests for the MuJoCo backend wrench and kinematics (#8612).

Covers baseline findings:

- **B5** — ``frame.T @ raw[:3]`` from ``mj_contactForce`` is the world force on
  **geom2's** body (verified against ``qfrc_constraint``: a resting sphere
  reports exactly ``m g``). The force on the clubhead is therefore the negative
  of that whenever the clubhead is ``geom1``. The old code summed it unsigned,
  so oppositely-ordered pairs cancelled. MuJoCo orders a contact pair by
  *collider type*, not geom id — a box clubhead against a sphere grain puts the
  sphere first, so the shipped MJCF happened to escape the defect, but a
  box-versus-box pair (a clubhead resting on a wall, or the parametric wedge
  mesh of ADR-0032) puts the clubhead first and flips the sign.
- **B5b** — only ``raw[3:]`` (the torsional friction couple at the contact
  point) was summed. The ``(r_contact - r_CoM) x F`` moment — the reason a
  wedge digs, twists or resists opening — was absent entirely.
- **B4** — the driver wrote ``qpos`` every step and never ``qvel``, so contacts
  were solved against a clubhead whose velocity never reflected the swing.
- **B10** — a missing trajectory silently substituted 5.0 m/s.
"""

from __future__ import annotations

from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")

import numpy as np  # noqa: E402
from _bunker_fixtures_8612 import (  # noqa: E402
    write_config,
    write_straight_trajectory,
)
from bunkershot3d.backends.mpm.driver import MPMDriver  # noqa: E402

pytestmark = pytest.mark.unit

GRAVITY = 9.80665

# A "clubhead" whose geom id is 0 (declared first) with a second box resting on
# top at a +x offset. Same-type pairs are ordered by geom id, so the clubhead is
# geom1 and ``frame.T @ raw[:3]`` reports the force on the *other* body.
_GRAIN_ON_CLUBHEAD_XML = """
<mujoco model="sign_probe">
  <option gravity="0 0 -9.80665" timestep="0.001"/>
  <worldbody>
    <body name="clubhead" pos="0 0 0.5">
      <geom name="clubface" type="box" size="0.3 0.3 0.05"/>
    </body>
    <body name="grain" pos="0.15 0 0.62">
      <freejoint/>
      <geom name="grain" type="box" size="0.05 0.05 0.05" density="1019.0"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def resting_contact() -> tuple[object, object, int, float]:
    """A settled grain resting on the clubhead; returns (model, data, id, m*g)."""
    model = mujoco.MjModel.from_xml_string(_GRAIN_ON_CLUBHEAD_XML)
    data = mujoco.MjData(model)
    for _ in range(3000):
        mujoco.mj_step(model, data)
    clubhead_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "clubhead")
    grain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "grain")
    weight = float(model.body_mass[grain_id]) * GRAVITY
    return model, data, clubhead_id, weight


def _driver_with(model: object, data: object, config_path: Path) -> MPMDriver:
    driver = MPMDriver(config_path)
    driver.model = model
    driver.data = data
    return driver


class TestContactForceSign:
    """B5: the reported force must be the force *on the clubhead*."""

    def test_geom_ordering_puts_clubhead_first(self, resting_contact: tuple) -> None:
        """Pin the premise: the clubhead is geom1 in every contact here."""
        _model, data, _clubhead_id, _weight = resting_contact
        assert data.ncon >= 1
        for index in range(data.ncon):
            assert data.contact[index].geom1 == 0  # clubface
            assert data.contact[index].geom2 == 1  # grain

    def test_force_on_clubhead_points_down(
        self, resting_contact: tuple, tmp_path: Path
    ) -> None:
        """A grain resting on the clubhead pushes it DOWN by exactly m*g."""
        model, data, clubhead_id, weight = resting_contact
        config = write_config(tmp_path / "c.yaml")
        driver = _driver_with(model, data, config)

        force, _torque = driver._extract_contact_wrench(clubhead_id)

        assert force[2] == pytest.approx(-weight, rel=2e-3), (
            "force on the clubhead must be the reaction (downwards), not the "
            "force on the grain"
        )

    def test_magnitude_matches_qfrc_constraint(self, resting_contact: tuple) -> None:
        """The reference the sign convention was verified against."""
        _model, data, _clubhead_id, weight = resting_contact
        assert float(data.qfrc_constraint[2]) == pytest.approx(weight, rel=2e-3)


class TestContactMoment:
    """B5b: the r x F moment about the clubhead CoM must be reported."""

    def test_moment_arm_term_is_present(
        self, resting_contact: tuple, tmp_path: Path
    ) -> None:
        model, data, clubhead_id, weight = resting_contact
        config = write_config(tmp_path / "c.yaml")
        driver = _driver_with(model, data, config)

        _force, torque = driver._extract_contact_wrench(clubhead_id)

        assert float(np.linalg.norm(torque)) > 0.1, (
            "the (r_contact - r_CoM) x F moment is missing: only the torsional "
            "friction couple was summed"
        )
        # A load of m*g applied 0.15 m ahead of the CoM tips the head about +y.
        assert torque[1] == pytest.approx(0.15 * weight, rel=0.02)
        assert abs(torque[0]) < 1e-3
        assert abs(torque[2]) < 1e-3

    def test_moment_vanishes_for_a_centred_contact(self, tmp_path: Path) -> None:
        """Placed over the CoM, the moment arm term is zero."""
        xml = _GRAIN_ON_CLUBHEAD_XML.replace('pos="0.15 0 0.62"', 'pos="0 0 0.62"')
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        for _ in range(3000):
            mujoco.mj_step(model, data)
        clubhead_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "clubhead")
        driver = _driver_with(model, data, write_config(tmp_path / "c.yaml"))

        _force, torque = driver._extract_contact_wrench(clubhead_id)
        assert float(np.linalg.norm(torque)) < 1e-2


class TestOppositelyOrderedPairsDoNotCancel:
    """B5: two contacts with opposite geom ordering used to cancel to zero."""

    # The floor is a plane, so the plane-box collider puts the floor first and
    # the clubhead second; the grain is a box, so the box-box collider orders by
    # geom id and puts the clubhead first. One contact of each ordering.
    _XML = """
    <mujoco model="ordering">
      <option gravity="0 0 -9.80665" timestep="0.002"/>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1"/>
        <body name="clubhead" pos="0 0 0.05">
          <freejoint/>
          <geom name="clubface" type="box" size="0.3 0.3 0.05" density="100"/>
        </body>
        <body name="grain" pos="0 0 0.155">
          <freejoint/>
          <geom name="grain" type="box" size="0.05 0.05 0.05" density="3000"/>
        </body>
      </worldbody>
    </mujoco>
    """

    def test_floor_and_grain_contacts_do_not_cancel(self, tmp_path: Path) -> None:
        model = mujoco.MjModel.from_xml_string(self._XML)
        data = mujoco.MjData(model)
        for _ in range(4000):
            mujoco.mj_step(model, data)
        clubhead_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "clubhead")
        grain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "grain")
        driver = _driver_with(model, data, write_config(tmp_path / "c.yaml"))

        force, _torque = driver._extract_contact_wrench(clubhead_id)

        club_mass = float(model.body_mass[clubhead_id])
        grain_mass = float(model.body_mass[grain_id])
        assert grain_mass > 0.5 * club_mass, "grain must be heavy enough to matter"

        # Floor pushes up with (m_club + m_grain) g; the grain pushes down with
        # m_grain g. Summing unsigned would report (m_club + 2 m_grain) g.
        assert force[2] == pytest.approx(club_mass * GRAVITY, rel=0.05)


class TestPrescribedClubheadVelocity:
    """B4: qvel must be prescribed alongside qpos."""

    def test_clubhead_dof_velocity_matches_trajectory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        speed = 1.0
        traj = write_straight_trajectory(
            tmp_path / "swing.csv", speed=speed, duration=0.02
        )
        config = write_config(
            tmp_path / "c.yaml",
            grain_count=8,
            diameter_mean=0.05,
            duration=0.01,
            trajectory_file=traj.name,
        )

        driver = MPMDriver(config)
        driver.setup()
        assert driver.model is not None

        clubhead_id = mujoco.mj_name2id(
            driver.model, mujoco.mjtObj.mjOBJ_BODY, "clubhead"
        )
        dof_adr = next(
            int(driver.model.jnt_dofadr[j])
            for j in range(driver.model.njnt)
            if driver.model.jnt_bodyid[j] == clubhead_id
        )

        seen: list[float] = []
        real_step = mujoco.mj_step

        def spy(model: object, data: object, *args: object, **kwargs: object) -> None:
            seen.append(float(data.qvel[dof_adr]))
            real_step(model, data, *args, **kwargs)

        monkeypatch.setattr(mujoco, "mj_step", spy)
        driver.run(tmp_path / "out.h5")

        assert seen, "driver never stepped"
        assert max(seen) == pytest.approx(speed, rel=0.05), (
            "clubhead qvel was never set: contacts are solved against a body "
            "whose velocity does not reflect the swing"
        )


class TestTrajectoryIsMandatory:
    """B10: a missing trajectory must fail loudly, not become 5 m/s."""

    def test_missing_trajectory_file_raises(self, tmp_path: Path) -> None:
        config = write_config(
            tmp_path / "c.yaml", trajectory_file="definitely_absent.csv"
        )
        driver = MPMDriver(config)
        with pytest.raises(FileNotFoundError) as excinfo:
            driver.run(tmp_path / "out.h5")
        assert "definitely_absent.csv" in str(excinfo.value)

    def test_no_hard_coded_fallback_velocity_in_source(self) -> None:
        import bunkershot3d.backends.mpm.driver as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "vel = 5.0" not in source
        assert "impact_velocity = 5.0" not in source
