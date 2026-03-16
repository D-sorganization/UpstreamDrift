from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.spatial_algebra.indexed_acceleration import (
    AccelerationClosureError,
    IndexedAcceleration,
    compute_indexed_acceleration_from_engine,
)


class TestIndexedAcceleration:
    def test_total(self):
        ia = IndexedAcceleration(
            gravity=np.array([1.0]),
            coriolis=np.array([2.0]),
            applied_torque=np.array([3.0]),
            constraint=np.array([4.0]),
            external=np.array([5.0]),
        )
        np.testing.assert_array_equal(ia.total, np.array([15.0]))

        ia.centrifugal = np.array([6.0])
        np.testing.assert_array_equal(ia.total, np.array([21.0]))

    def test_assert_closure(self):
        ia = IndexedAcceleration(
            gravity=np.array([1.0]),
            coriolis=np.array([1.0]),
            applied_torque=np.array([1.0]),
            constraint=np.array([1.0]),
            external=np.array([1.0]),
        )

        # total is 5.0
        # This should pass:
        ia.assert_closure(np.array([5.0]))

        # This should raise:
        with pytest.raises(AccelerationClosureError):
            ia.assert_closure(np.array([4.0]))

    def test_get_contribution_percentages(self):
        ia = IndexedAcceleration(
            gravity=np.array([2.0]),
            coriolis=np.array([0.0]),
            applied_torque=np.array([0.0]),
            constraint=np.array([0.0]),
            external=np.array([0.0]),
        )

        pct = ia.get_contribution_percentages()
        assert pct["gravity"] == 100.0
        assert pct["coriolis"] == 0.0

        # very small magnitude
        ia2 = IndexedAcceleration(
            gravity=np.array([1e-15]),
            coriolis=np.array([0.0]),
            applied_torque=np.array([0.0]),
            constraint=np.array([0.0]),
            external=np.array([0.0]),
        )
        pct2 = ia2.get_contribution_percentages()
        assert pct2["gravity"] == 0.0

    def test_compute_indexed_acceleration_from_engine(self):
        engine = MagicMock()
        engine.compute_drift_acceleration.return_value = np.array([5.0])
        engine.compute_control_acceleration.return_value = np.array([2.0])
        engine.compute_mass_matrix.return_value = np.array([[2.0]])
        engine.compute_gravity_forces.return_value = np.array([6.0])

        # gravity acc = M_inv @ g_forces = 0.5 * 6.0 = 3.0
        # coriolis acc = drift - gravity = 5.0 - 3.0 = 2.0

        tau = np.array([1.0])
        ia = compute_indexed_acceleration_from_engine(engine, tau)

        np.testing.assert_array_equal(ia.gravity, np.array([3.0]))
        np.testing.assert_array_equal(ia.coriolis, np.array([2.0]))
        np.testing.assert_array_equal(ia.applied_torque, np.array([2.0]))
        np.testing.assert_array_equal(ia.constraint, np.array([0.0]))
        np.testing.assert_array_equal(ia.external, np.array([0.0]))
