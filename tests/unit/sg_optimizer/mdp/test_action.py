import numpy as np

from src.shared.python.sg_optimizer.mdp.action import ActionSet


def test_action_set_default_aim_grid_is_float64() -> None:
    actions = ActionSet(clubs=("7_iron",))

    assert actions.aim_grid_deg.dtype == np.float64
    assert actions.aim_grid_deg.shape == (31,)
