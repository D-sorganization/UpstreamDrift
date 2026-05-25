"""C3D loader tests; integration tests rely on the cluster-marker C3D files."""

from __future__ import annotations

import pytest
from src.shared.python.motion_matching import (
    AlignOptions,
    load_club_target_c3d,
)
from src.shared.python.motion_matching.loaders.synthetic import (
    synthesize_target_from_coefficients,
)

from ._fixtures import repo_root

C3D_DIR_RELATIVE = (
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/Data/Mocap C3D Files"
)


def _first_c3d():
    base = repo_root() / C3D_DIR_RELATIVE
    if not base.is_dir():
        return None
    files = sorted(base.glob("*.c3d"))
    return files[0] if files else None


def test_c3d_rejects_missing_file() -> None:
    import numpy as np
    from src.shared.python.core.contracts import PreconditionError

    with pytest.raises((FileNotFoundError, ValueError, PreconditionError)):
        load_club_target_c3d("does/not/exist.c3d", AlignOptions())
    # Synthetic dispatcher requires an explicit engine name and raises
    # ValueError when none is provided (replaced the original
    # NotImplementedError stub once #4122 / #4166 landed engine backends).
    with pytest.raises(ValueError, match="no engine specified"):
        synthesize_target_from_coefficients(np.zeros(3), AlignOptions())
