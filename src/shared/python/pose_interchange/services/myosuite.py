"""MyoSuite :class:`LiveKinematicsService` implementation.

MyoSuite (https://sites.google.com/view/myosuite) is built on top of
MuJoCo and consumes MJCF (``.xml``) models. The byte-level kinematics
surface is therefore the same as MuJoCo's: ``qpos`` is laid out as
``[x, y, z, qw, qx, qy, qz, joint_0, ...]`` and ``mj_forward`` /
``mj_step`` advance the world. This service mirrors
:class:`MuJoCoKinematicsService` exactly (per issue #6091 the design
explicitly says "mirror the closest sibling, don't innovate").

Wheel-availability logic:

- The real service is returned only when **both** :mod:`mujoco` AND
  :mod:`myosuite` are importable. If only :mod:`mujoco` is installed
  we still fall back to the mock so users who asked for MyoSuite get
  a MyoSuite-named mock rather than silently being upgraded to a
  MuJoCo service.
- The fall-back :class:`MockKinematicsService` is initialised with
  ``engine_name="myosuite"`` so the registry contract holds.

:meth:`step` advances the underlying ``mujoco.mj_step`` — MyoSuite
itself just composes Gym-style environments on top of that primitive,
so we sidestep the heavy gym/env machinery for the live-kinematics
path. Live MyoSuite-env stepping (with muscle activations etc.) is
explicitly **out of scope** for this PR; tracked under #6091.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.pose_interchange.adapters._base import (
    build_default_joint_layout,
    encode_joint_angles,
    euler_xyz_deg_to_quat_wxyz,
)
from src.shared.python.pose_interchange.canonical import CanonicalPose
from src.shared.python.pose_interchange.live_kinematics import (
    LiveKinematicsService,
    ServiceCapabilities,
)
from src.shared.python.pose_interchange.protocol import JointSlot
from src.shared.python.pose_interchange.services._mock import (
    MockKinematicsService,
)

logger = get_logger(__name__)

ENGINE_NAME = "myosuite"


_MYOSUITE_CAPABILITIES = ServiceCapabilities(
    supports_dynamics_step=True,
    supports_collision_query=True,
    supports_realtime=True,
)


def _module_is_importable(name: str) -> bool:
    """Return ``True`` if *name* can be imported.

    Resilient to test conftests that stub the module without setting
    ``__spec__`` (``find_spec`` raises ``ValueError`` in that case).
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _myosuite_is_importable() -> bool:
    """Return ``True`` only when both :mod:`mujoco` and :mod:`myosuite` import.

    MyoSuite always pulls in MuJoCo, but we check both explicitly so a
    bare ``mujoco`` install does not accidentally satisfy the
    MyoSuite registry slot.
    """
    return _module_is_importable("mujoco") and _module_is_importable("myosuite")


def _configured_joint_layout(model: Any) -> Mapping[str, JointSlot] | None:
    if hasattr(model, "joint_layout") and isinstance(model.joint_layout, Mapping):
        return model.joint_layout
    if isinstance(model, Mapping) and isinstance(model.get("joint_layout"), Mapping):
        return model["joint_layout"]
    return None


def _discover_joint_layout(
    model: Any,
    mujoco_module: Any,
) -> Mapping[str, JointSlot]:
    configured_layout = _configured_joint_layout(model)
    if configured_layout is not None:
        return configured_layout

    # _TODO_review_convention: MyoSuite environments use environment-
    # specific joint names (e.g. ``hip_flexion_r``, ``elbow_flexion``)
    # rather than our canonical ones. Until MyoSuite-specific mapping
    # tables are added (tracked under #6091), we reuse the canonical
    # name set with a ``myo_`` prefix so the default mock layout still
    # round-trips. Real MyoSuite models must pass an explicit
    # ``joint_layout`` via the model handle.
    layout_by_engine_name = {
        slot.engine_name: canonical_name
        for canonical_name, slot in build_default_joint_layout(
            base_offset=0, units="rad", sign=1, name_prefix="myo_"
        ).items()
    }
    layout_by_engine_name.update(
        {
            canonical_name: canonical_name
            for canonical_name in layout_by_engine_name.values()
        }
    )

    discovered: dict[str, JointSlot] = {}
    njnt = int(getattr(model, "njnt", 0))
    jnt_qposadr = getattr(model, "jnt_qposadr", ())
    jnt_type = getattr(model, "jnt_type", ())
    free_joint_kind = int(mujoco_module.mjtJoint.mjJNT_FREE)
    for joint_index in range(njnt):
        joint_name = mujoco_module.mj_id2name(
            model,
            mujoco_module.mjtObj.mjOBJ_JOINT,
            joint_index,
        )
        if not joint_name:
            continue
        canonical_name = layout_by_engine_name.get(str(joint_name))
        if canonical_name is None:
            continue
        if int(jnt_type[joint_index]) == free_joint_kind:
            continue
        discovered[canonical_name] = JointSlot(
            canonical_name=canonical_name,
            engine_name=str(joint_name),
            start_index=int(jnt_qposadr[joint_index]),
            length=1,
            units="rad",
            sign=1,
        )
    return discovered


def _free_joint_qpos_address(model: Any, mujoco_module: Any) -> int | None:
    njnt = int(getattr(model, "njnt", 0))
    jnt_qposadr = getattr(model, "jnt_qposadr", ())
    jnt_type = getattr(model, "jnt_type", ())
    free_joint_kind = int(mujoco_module.mjtJoint.mjJNT_FREE)
    for joint_index in range(njnt):
        if int(jnt_type[joint_index]) == free_joint_kind:
            return int(jnt_qposadr[joint_index])
    return None


def _canonical_pose_to_qpos(
    model: Any,
    pose: CanonicalPose,
    mujoco_module: Any,
) -> npt.NDArray[np.float64]:
    nq = int(model.nq)
    qpos = np.zeros(nq, dtype=np.float64)

    free_joint_qpos_address = _free_joint_qpos_address(model, mujoco_module)
    if free_joint_qpos_address is not None:
        free_joint_stop = free_joint_qpos_address + 7
        if free_joint_stop > nq:
            raise RuntimeError(
                "MyoSuiteKinematicsService.set_pose: free joint overruns qpos "
                f"(start={free_joint_qpos_address}, nq={nq})."
            )
        qpos[free_joint_qpos_address : free_joint_qpos_address + 3] = (
            pose.pelvis_translation_m
        )
        qpos[free_joint_qpos_address + 3 : free_joint_stop] = (
            euler_xyz_deg_to_quat_wxyz(pose.pelvis_rotation_xyz_deg)
        )
    elif np.any(pose.pelvis_translation_m) or np.any(pose.pelvis_rotation_xyz_deg):
        logger.debug(
            "MyoSuiteKinematicsService.set_pose: model has no free joint; "
            "ignoring canonical pelvis pose."
        )

    joint_layout = _discover_joint_layout(model, mujoco_module)
    if joint_layout:
        encode_joint_angles(pose.joint_angles_deg, joint_layout, qpos)
    return qpos


class MyoSuiteKinematicsService:
    """Real-engine :class:`LiveKinematicsService` backed by MyoSuite/MuJoCo.

    MyoSuite is MJCF-backed and delegates physics to :mod:`mujoco`, so
    we load the model via :func:`mujoco.MjModel.from_xml_path` directly
    rather than spinning up a full MyoSuite gym environment. This keeps
    the live-kinematics surface fast (no env reset overhead) and
    independent of the heavier MyoSuite muscle/reward machinery; live
    MyoSuite env-step support is tracked separately under #6091.
    """

    engine_name: str = ENGINE_NAME

    def __init__(self) -> None:
        self._model: Any = None
        self._data: Any = None
        self._pose: CanonicalPose | None = None

    def _require_loaded(self, op: str) -> None:
        if self._model is None or self._data is None:
            raise RuntimeError(
                f"MyoSuiteKinematicsService.{op}: model not loaded; call load() first."
            )

    def load(self, model_path: Path) -> None:
        if not isinstance(model_path, Path):
            raise TypeError(
                f"model_path must be a pathlib.Path, got {type(model_path).__name__}"
            )
        import mujoco  # type: ignore[import-not-found]

        # _TODO_review_convention: MyoSuite environments are usually
        # constructed via ``gym.make("myoChallengeXxx-v0")`` rather than
        # by direct MJCF load. For raw kinematics evaluation we bypass
        # the env wrapper — confirm this is acceptable for the Pose
        # Studio use case before promoting to production. Tracked
        # under #6091.
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)
        logger.debug(
            "MyoSuiteKinematicsService loaded model_path=%s nq=%d nbody=%d",
            model_path,
            getattr(self._model, "nq", -1),
            getattr(self._model, "nbody", -1),
        )

    def set_pose(self, pose: CanonicalPose) -> None:
        if not isinstance(pose, CanonicalPose):
            raise TypeError(f"pose must be a CanonicalPose, got {type(pose).__name__}")
        self._require_loaded("set_pose")
        self._pose = pose
        import mujoco  # type: ignore[import-not-found]

        qpos = _canonical_pose_to_qpos(self._model, pose, mujoco)
        self._data.qpos[:] = qpos  # type: ignore[union-attr]
        mujoco.mj_forward(self._model, self._data)

    def get_link_transforms(self) -> dict[str, npt.NDArray[np.float64]]:
        self._require_loaded("get_link_transforms")
        import mujoco  # type: ignore[import-not-found]

        transforms: dict[str, npt.NDArray[np.float64]] = {}
        nbody = int(self._model.nbody)  # type: ignore[union-attr]
        for i in range(nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name is None or name == "":
                name = f"body_{i}"
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = np.asarray(
                self._data.xmat[i],  # type: ignore[union-attr]
                dtype=np.float64,
            ).reshape(3, 3)
            transform[:3, 3] = np.asarray(
                self._data.xpos[i],  # type: ignore[union-attr]
                dtype=np.float64,
            )
            transforms[name] = transform
        return transforms

    def step(self, dt: float) -> None:
        """Advance the underlying MuJoCo simulation by ``dt`` seconds.

        Notes
        -----
        This delegates to :func:`mujoco.mj_step` on the loaded MJCF and
        deliberately bypasses MyoSuite's gym-env step (muscle
        activations, reward shaping). Full MyoSuite env stepping is
        intentionally out of scope for issue #6091; tracked there for
        a follow-up.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        self._require_loaded("step")
        import mujoco  # type: ignore[import-not-found]

        self._model.opt.timestep = float(dt)  # type: ignore[union-attr]
        mujoco.mj_step(self._model, self._data)

    def reset(self) -> None:
        self._pose = None
        if self._model is None or self._data is None:
            return
        import mujoco  # type: ignore[import-not-found]

        mujoco.mj_resetData(self._model, self._data)

    def capabilities(self) -> ServiceCapabilities:
        return _MYOSUITE_CAPABILITIES


def create_myosuite_service() -> LiveKinematicsService:
    """Return a MyoSuite service if MyoSuite + MuJoCo install, else mock."""
    if _myosuite_is_importable():
        logger.debug(
            "create_myosuite_service: myosuite importable, returning real service."
        )
        return MyoSuiteKinematicsService()
    logger.info(
        "create_myosuite_service: myosuite not importable, "
        "falling back to MockKinematicsService(engine_name=%r).",
        ENGINE_NAME,
    )
    mock: LiveKinematicsService = MockKinematicsService(engine_name=ENGINE_NAME)
    return mock


__all__ = [
    "ENGINE_NAME",
    "MyoSuiteKinematicsService",
    "create_myosuite_service",
]
