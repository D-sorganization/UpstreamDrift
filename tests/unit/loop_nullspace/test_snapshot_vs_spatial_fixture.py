"""NSX-13: ``snapshot_from_adapter`` equals the NSX-12 spatial fixture (#12558).

The fixture (``spatial_fixture``) builds its snapshot with frame Jacobians and
classical accelerations on a hand-built Pinocchio chain.  Here the *same*
Pinocchio model is driven through the real ``PinocchioForceAdapter`` methods
(constraint-model Jacobian, ``constraintDynamics`` drift, frame protocol,
canonical conversion) via a ``NativePinocchioModel`` whose tree is the fixture's
model and whose weld is a ``RigidConstraintModel`` between the two grip frames.
The spec-driven club inertia split is not derivable from a hand-built chain, so
it is supplied from the fixture's snapshot; everything else is compared.
"""

from __future__ import annotations

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from src.engines.physics_engines.pinocchio.python.force_adapter import (  # noqa: E402
    PinocchioForceAdapter,
)
from src.engines.physics_engines.pinocchio.python.native_model import (  # noqa: E402
    NativePinocchioModel,
)
from src.shared.python.loop_nullspace import spatial_fixture  # noqa: E402
from src.shared.python.loop_nullspace.contracts import InputProvenance  # noqa: E402
from src.shared.python.loop_nullspace.snapshot import (  # noqa: E402
    snapshot_from_adapter,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

TOL = 1e-9


class _FixtureNativeModel(NativePinocchioModel):
    """Real ``NativePinocchioModel`` methods over the fixture's Pinocchio tree."""

    def __init__(self, sp, names: tuple[str, ...]) -> None:
        self._pin = pin
        self.model = sp.model
        self._coordinates = {n: i for i, n in enumerate(names)}
        self._velocity_indices = dict(self._coordinates)
        frames = sp.model.frames
        trail, lead = frames[sp.trail_grip], frames[sp.lead_grip]
        constraint = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D,
            sp.model,
            trail.parentJoint,
            trail.placement,
            lead.parentJoint,
            lead.placement,
            pin.ReferenceFrame.LOCAL,
        )
        self.constraints = [constraint]
        self.constraint_data = [constraint.createData()]
        self.data = sp.model.createData()
        pin.initConstraintDynamics(
            self.model, self.data, self.constraints, self.constraint_data
        )


def _adapter(sp, case) -> PinocchioForceAdapter:
    topo = case.snapshot.topology
    ad = PinocchioForceAdapter.__new__(PinocchioForceAdapter)
    ad._plant = _FixtureNativeModel(sp, topo.coordinate_order)
    ad._pin, ad._model, ad._data = pin, sp.model, sp.model.createData()
    ad._names = topo.coordinate_order
    ad._actuated = topo.actuated_indices
    ad._contact_names, ad._contact_ids = (), ()
    bare = case.snapshot.bare_club
    hands = case.snapshot.welded_hand_inertias
    ad.club_inertia_split = lambda: (bare, hands)  # spec-driven; see docstring
    return ad


@pytest.mark.parametrize("floating_root", [False, True])
def test_adapter_snapshot_equals_nsx12_spatial_fixture(floating_root) -> None:
    case = spatial_fixture.build_case(floating_root=floating_root)
    sp = spatial_fixture._build_closed(floating_root, False)
    ref = case.snapshot
    ad = _adapter(sp, case)
    snap = snapshot_from_adapter(
        ad,
        ref.q,
        ref.v,
        ref.a,
        input_provenance=InputProvenance("test", "nsx-12 spatial fixture"),
    )
    assert snap.J_c.shape == ref.J_c.shape == (6, ref.topology.n)
    for name in ("M", "h", "J_c", "Jdot_v"):
        got, want = getattr(snap, name), getattr(ref, name)
        scale = max(1.0, float(np.max(np.abs(want))))
        assert np.max(np.abs(got - want)) / scale < TOL, name
    np.testing.assert_allclose(
        snap.grip_frame.rotation, ref.grip_frame.rotation, atol=1e-12
    )
    np.testing.assert_allclose(
        snap.grip_frame.origin, ref.grip_frame.origin, atol=1e-12
    )
    # the fixture state is admissible: the adapter snapshot also closes the loop
    assert np.max(np.abs(snap.J_c @ snap.v)) < 1e-10
    assert np.max(np.abs(snap.J_c @ snap.a + snap.Jdot_v)) < 1e-9
