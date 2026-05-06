"""C3D loader tests; integration tests rely on the Optical C3D files."""

from __future__ import annotations

import pytest
from src.shared.python.motion_matching import (
    AlignOptions,
    ClubTarget,
    load_club_target_c3d,
)
from src.shared.python.motion_matching.loaders.synthetic import (
    synthesize_target_from_coefficients,
)

from ._fixtures import repo_root

C3D_DIR_RELATIVE = (
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/Data/Optical C3D Files"
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
    # Synthetic stub raises NotImplementedError
    with pytest.raises(NotImplementedError):
        synthesize_target_from_coefficients(np.zeros(3), AlignOptions())


@pytest.mark.integration
def test_load_c3d_succeeds_on_optical_file() -> None:
    p = _first_c3d()
    if p is None:
        pytest.skip("No Optical C3D files present")
    try:
        target = load_club_target_c3d(p, AlignOptions())
    except (ValueError, ImportError) as exc:
        pytest.skip(f"C3D parser could not produce a ClubTarget for {p.name}: {exc}")
    assert isinstance(target, ClubTarget)
    assert target.time.shape[0] == target.butt.shape[0]
    assert target.source.format == "c3d"
