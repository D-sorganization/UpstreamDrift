"""Tests for drake spatial_algebra helpers."""

import numpy as np

def test_spatial_algebra_imports():
    """Test that all proxies from spatial_algebra are correctly exposed."""
    # Test __init__
    import src.engines.physics_engines.drake.python.src.spatial_algebra as sa
    assert sa is not None

    # Test inertia
    from src.engines.physics_engines.drake.python.src.spatial_algebra.inertia import (
        mcI,
        transform_spatial_inertia,
    )
    assert callable(mcI)
    assert callable(transform_spatial_inertia)

    # Test spatial_vectors
    from src.engines.physics_engines.drake.python.src.spatial_algebra.spatial_vectors import (
        crf,
        crm,
        skew,
        spatial_cross,
    )
    assert callable(crf)
    assert callable(crm)
    assert callable(skew)
    assert callable(spatial_cross)

    # Test transforms
    from src.engines.physics_engines.drake.python.src.spatial_algebra.transforms import (
        inv_xtrans,
        xlt,
        xrot,
        xtrans,
    )
    assert callable(inv_xtrans)
    assert callable(xlt)
    assert callable(xrot)
    assert callable(xtrans)

    # Test joints
    from src.engines.physics_engines.drake.python.src.spatial_algebra.joints import (
        S_PX,
        S_PY,
        S_PZ,
        S_RX,
        S_RY,
        S_RZ,
        jcalc,
    )
    assert S_PX is not None
    assert S_PY is not None
    assert S_PZ is not None
    assert S_RX is not None
    assert S_RY is not None
    assert S_RZ is not None
    assert callable(jcalc)


def test_jcalc_proxy():
    """Test the proxy jcalc implementation in drake spatial_algebra.joints."""
    from src.engines.physics_engines.drake.python.src.spatial_algebra.joints import jcalc
    
    # Test with a revolute joint around X axis (assuming jtype "rx" or "Rx")
    # According to featherstone, 'Rx' returns X_J and S
    xj_transform, s_subspace = jcalc("Rx", 0.0)
    
    assert isinstance(xj_transform, np.ndarray)
    assert isinstance(s_subspace, np.ndarray)
    assert xj_transform.shape == (6, 6)
    assert s_subspace.shape == (6, 1) or s_subspace.shape == (6,)
