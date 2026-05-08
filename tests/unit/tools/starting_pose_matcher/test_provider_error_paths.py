"""Error / availability paths for physics-engine providers.

Each provider's import fails to a clean ``*NotAvailableError`` when the
engine wheel isn't installed; ``*ProviderError`` is raised on missing
or invalid configuration. These tests do not need the real engine
because the conftest replaces them with ``MagicMock``.

Test-only; no production code changes (issue #4673).
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Drake                                                                       #
# --------------------------------------------------------------------------- #


def test_drake_constants_and_classes_importable():
    from src.tools.starting_pose_matcher.providers import drake

    assert hasattr(drake, "DRAKE_TO_MATCHER_VOCAB")
    assert hasattr(drake, "MATCHER_TO_DRAKE")
    assert "ch" in drake.MATCHER_TO_DRAKE
    assert drake.MATCHER_TO_DRAKE["ch"] == "clubhead"


def test_drake_provider_with_neither_raises_provider_error():
    from src.tools.starting_pose_matcher.providers.drake import (
        DrakeNotAvailableError,
        DrakeProviderError,
        DrakeSkeletonProvider,
    )

    # Either DrakeNotAvailableError (real Drake missing) or the
    # DrakeProviderError (our explicit "neither path nor xml" message).
    with pytest.raises((DrakeNotAvailableError, DrakeProviderError)):
        DrakeSkeletonProvider(model_path=None, model_xml=None)


def test_drake_create_provider_factory():
    from src.tools.starting_pose_matcher.providers import drake

    # The factory is a thin wrapper — ensure it forwards to the constructor.
    with pytest.raises((drake.DrakeNotAvailableError, drake.DrakeProviderError)):
        drake.create_provider(model_path=None, model_xml=None)


# --------------------------------------------------------------------------- #
# OpenSim                                                                     #
# --------------------------------------------------------------------------- #


def test_opensim_constants_and_classes_importable():
    from src.tools.starting_pose_matcher.providers import opensim

    assert hasattr(opensim, "OPENSIM_TO_MATCHER_VOCAB")
    assert opensim.MATCHER_TO_OPENSIM["ls"] == "left_shoulder"


def test_opensim_provider_with_neither_raises():
    from src.tools.starting_pose_matcher.providers.opensim import (
        OpenSimNotAvailableError,
        OpenSimProviderError,
        OpenSimSkeletonProvider,
    )

    with pytest.raises((OpenSimNotAvailableError, OpenSimProviderError)):
        OpenSimSkeletonProvider(model_path=None, model_xml=None)


def test_opensim_create_provider_factory():
    from src.tools.starting_pose_matcher.providers import opensim

    with pytest.raises(
        (opensim.OpenSimNotAvailableError, opensim.OpenSimProviderError)
    ):
        opensim.create_provider(model_path=None, model_xml=None)


# --------------------------------------------------------------------------- #
# MuJoCo                                                                      #
# --------------------------------------------------------------------------- #


def test_mujoco_constants_and_classes_importable():
    from src.tools.starting_pose_matcher.providers import mujoco

    assert hasattr(mujoco, "MUJOCO_TO_MATCHER_VOCAB")
    # "hip" in matcher vocab maps to either "hip" or "pelvis" in MuJoCo —
    # both are accepted by MUJOCO_TO_MATCHER_VOCAB.
    assert mujoco.MATCHER_TO_MUJOCO["hip"] in ("hip", "pelvis")


def test_mujoco_provider_with_neither_raises():
    from src.tools.starting_pose_matcher.providers.mujoco import (
        MuJoCoNotAvailableError,
        MuJoCoProviderError,
        MuJoCoSkeletonProvider,
    )

    with pytest.raises((MuJoCoNotAvailableError, MuJoCoProviderError)):
        MuJoCoSkeletonProvider(model_path=None, model_xml=None)


def test_mujoco_create_provider_factory():
    from src.tools.starting_pose_matcher.providers import mujoco

    with pytest.raises((mujoco.MuJoCoNotAvailableError, mujoco.MuJoCoProviderError)):
        mujoco.create_provider(model_path=None, model_xml=None)


# --------------------------------------------------------------------------- #
# Pinocchio                                                                   #
# --------------------------------------------------------------------------- #


def test_pinocchio_constants_and_classes_importable():
    from src.tools.starting_pose_matcher.providers import pinocchio

    assert hasattr(pinocchio, "PINOCCHIO_TO_MATCHER_VOCAB")
    assert pinocchio.MATCHER_TO_PINOCCHIO["mp"] == "midpoint"


def test_pinocchio_provider_without_path_raises():
    from src.tools.starting_pose_matcher.providers.pinocchio import (
        PinocchioNotAvailableError,
        PinocchioProviderError,
        PinocchioSkeletonProvider,
    )

    with pytest.raises((PinocchioNotAvailableError, PinocchioProviderError)):
        PinocchioSkeletonProvider(urdf_path=None)


def test_pinocchio_create_provider_factory_with_no_path():
    from src.tools.starting_pose_matcher.providers import pinocchio

    with pytest.raises(
        (pinocchio.PinocchioNotAvailableError, pinocchio.PinocchioProviderError)
    ):
        pinocchio.create_provider(urdf_path=None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Vocabulary completeness — every required short name maps somewhere.        #
# --------------------------------------------------------------------------- #


REQUIRED_VOCAB = [
    "hip",
    "spine",
    "torso",
    "hub",
    "ls",
    "rs",
    "le",
    "re",
    "lw",
    "rw",
    "mp",
    "ch",
]


@pytest.mark.parametrize(
    "module_name",
    [
        "src.tools.starting_pose_matcher.providers.drake",
        "src.tools.starting_pose_matcher.providers.mujoco",
        "src.tools.starting_pose_matcher.providers.opensim",
        "src.tools.starting_pose_matcher.providers.pinocchio",
    ],
)
def test_provider_vocabulary_is_complete(module_name: str):
    import importlib

    mod = importlib.import_module(module_name)
    matcher_to_engine = next(
        getattr(mod, name) for name in dir(mod) if name.startswith("MATCHER_TO_")
    )
    for vocab in REQUIRED_VOCAB:
        assert vocab in matcher_to_engine, (
            f"{module_name}: missing vocabulary mapping for {vocab}"
        )
