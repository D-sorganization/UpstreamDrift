"""Regression test for canonical OpenSim FK extractors (issue #4191).

This test guards against the regression that motivated issue #4191:
the original ``opensim_golf/fk.py`` looked for body names ``hand_left`` /
``hand_right`` that **do not exist** in the canonical
``golf_humanoid.osim``. The shipped model exposes the grip and clubhead
as ``PhysicalOffsetFrame`` objects on the ``hand_r_to_club`` weld joint
(``hand_r_grip_offset`` and ``club_head_offset``).

The canonical FK module under test is
``src/engines/physics_engines/opensim/python/opensim_golf/fk.py``.

Layers:

1. **Pure-Python structural assertions** (always run): import the FK
   module, check the canonical landmark catalogue and frame-path
   constants are present and well-formed. These run on every CI matrix
   leg even when the OpenSim wheel is not installed.
2. **OpenSim binding extraction** (``requires_opensim`` marker): load
   the committed ``golf_humanoid.osim`` via ``osim.Model``, call
   ``initSystem()``, and assert the canonical extractors return finite,
   sane-magnitude poses at the neutral pose. Skipped automatically when
   the OpenSim Python bindings are not installed.

Acceptance per issue #4191:
    - ``grip`` and ``clubhead`` positions are finite NumPy arrays.
    - Both lie within 5 m of origin (sane-magnitude sanity bound).
    - Quaternions are unit-norm.
    - The frames resolved are the canonical ``hand_r_grip_offset`` /
      ``club_head_offset`` (not the legacy ``hand_left`` / ``hand_right``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "opensim"
    / "models"
    / "golf_humanoid.osim"
)


def _opensim_available() -> bool:
    """Detect a real OpenSim install without tripping mocked sys.modules entries.

    Some sibling test modules patch ``sys.modules['opensim']`` with a
    ``MagicMock`` that lacks ``__spec__``. ``find_spec`` raises
    ``ValueError`` on that case during *collection* (i.e. before any
    fixture cleanup has run), so we swallow it and treat the binding as
    absent for the purposes of layer-2 gating.
    """
    try:
        return importlib.util.find_spec("opensim") is not None
    except (ValueError, ModuleNotFoundError):  # pragma: no cover - test pollution
        return False


_OPENSIM_AVAILABLE = _opensim_available()


# ---------------------------------------------------------------------------
# Layer 1: pure-Python structural assertions (no opensim dependency).
# ---------------------------------------------------------------------------


def test_fk_module_exposes_canonical_frame_constants() -> None:
    """The canonical frame names must match the .osim contract."""
    from src.engines.physics_engines.opensim.python.opensim_golf import fk

    assert fk.GRIP_FRAME_NAME == "hand_r_grip_offset"
    assert fk.CLUBHEAD_FRAME_NAME == "club_head_offset"
    assert fk.GRIP_FRAME_PATH == "/jointset/hand_r_to_club/hand_r_grip_offset"
    assert fk.CLUBHEAD_FRAME_PATH == "/jointset/hand_r_to_club/club_head_offset"
    # Catalogue keys are the cross-engine SimOut landmark names.
    assert set(fk.CANONICAL_LANDMARKS) == {"grip", "clubhead"}
    assert fk.CANONICAL_LANDMARKS["grip"] == fk.GRIP_FRAME_PATH
    assert fk.CANONICAL_LANDMARKS["clubhead"] == fk.CLUBHEAD_FRAME_PATH


def test_fk_module_exposes_canonical_extractors() -> None:
    """Public canonical API must be importable and callable."""
    from src.engines.physics_engines.opensim.python.opensim_golf import fk

    for name in ("extract_grip_pose", "extract_clubhead_pose", "extract_full_pose"):
        assert hasattr(fk, name), f"fk module missing {name!r}"
        assert callable(getattr(fk, name))


def test_legacy_extractors_emit_deprecation_warning_and_validate() -> None:
    """Legacy entry points still raise their documented errors.

    Ensures the deprecation wrappers do not break pre-existing callers
    that catch ValueError / TypeError / NotImplementedError.
    """
    from src.engines.physics_engines.opensim.python.opensim_golf.fk import (
        compute_skeleton_fk,
    )

    sentinel_model = object()
    # Empty list -> ValueError after the deprecation warning fires.
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="empty"):
        compute_skeleton_fk(sentinel_model, [])

    # Wrong type -> TypeError.
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError):
        compute_skeleton_fk(sentinel_model, "not a list")  # type: ignore[arg-type]

    # ndarray -> NotImplementedError (matches legacy behaviour).
    with pytest.warns(DeprecationWarning), pytest.raises(NotImplementedError):
        compute_skeleton_fk(sentinel_model, np.zeros((10, 23)))


def test_canonical_extractors_reject_none_inputs() -> None:
    """Pre-condition: state and model must be non-None."""
    from src.engines.physics_engines.opensim.python.opensim_golf.fk import (
        extract_clubhead_pose,
        extract_full_pose,
        extract_grip_pose,
    )

    for func in (extract_grip_pose, extract_clubhead_pose, extract_full_pose):
        with pytest.raises(ValueError, match="state and model"):
            func(None, object())
        with pytest.raises(ValueError, match="state and model"):
            func(object(), None)


# ---------------------------------------------------------------------------
# Layer 2: actual OpenSim load + extraction (binding required).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_model_and_state():
    """Load the canonical model and return ``(model, state)``.

    Module-scoped to amortise the SWIG load across all extractor tests.
    """
    import opensim as osim

    assert MODEL_PATH.is_file(), (
        f"golf_humanoid.osim missing at {MODEL_PATH}. "
        "Run `python3 scripts/build_humanoid_osim.py` to regenerate."
    )
    model = osim.Model(str(MODEL_PATH))
    state = model.initSystem()
    return model, state
