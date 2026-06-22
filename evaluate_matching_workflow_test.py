import numpy as np

def test_workflow():
    joint_power = np.random.rand(10, 5)

    net_power = np.sum(joint_power, axis=1)
    net_power_einsum = np.einsum('ij->i', joint_power)

    assert np.allclose(net_power, net_power_einsum)

test_workflow()
