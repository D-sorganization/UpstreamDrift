"""Pluggable skeleton extractor for model joint positions.

This is the seam that issue #4367 will exploit to make the matcher work
with all four physics engines (MuJoCo / Drake / Pinocchio / OpenSim) and
not just Simscape.  The matcher's GUI talks to a ``SkeletonExtractor`` —
each extractor knows how to enumerate the model's poses and return their
joint positions in the matcher's vocabulary (``hip``, ``spine``,
``torso``, ``hub``, ``ls``, ``rs``, ``le``, ``re``, ``lw``, ``rw``,
``mp``, ``ch``).

Today only the JSON extractor is wired in (it consumes the
``simscape_skeleton_<pose>.json`` files produced by
``export_default_skeleton.m`` next to the legacy MATLAB tree).  Future
extractors will dispatch to engine-native FK:

     Engine      | Source              | Implementation hint
    -----------:|:--------------------|:---------------------
     Simscape    | JSON file          | this module (``JsonSkeletonExtractor``)
     MuJoCo      | MJCF + qpos        | ``mujoco.MjData``; read ``xpos``
     Drake       | URDF/SDF + plant   | ``MultibodyPlant.SetPositions`` then ``EvalBodyPoseInWorld``
     Pinocchio   | URDF + q           | ``pin.forwardKinematics``; ``data.oMi[id].translation``
     OpenSim     | OSIM + state       | ``Model.realizePosition`` then ``Body.getPositionInGround``

See #4367 for the full plan; this module currently exposes the abstract
base class + the JSON extractor so the matcher's GUI is already engine-
agnostic at the call sites.

Note: This module is named ``skeleton_extractor`` (not ``skeleton_provider``)
to disambiguate from ``FitSwingProvider`` in the motion_matching package.
The naming distinction is documented in docs/adr/XXXX-skeleton-vs-fit-swing-naming.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.tools.starting_pose_matcher.core import (
    Skeleton,
    fallback_skeleton,
    load_skeleton,
)


class SkeletonExtractor(ABC):
    """Abstract interface for sources of model skeleton poses."""

    @abstractmethod
    def list_poses(self) -> list[str]:
        """Return the names of the poses this extractor can produce."""

    @abstractmethod
    def get_skeleton(self, pose_name: str) -> Skeleton:
        """Return the :class:`Skeleton` for the named pose."""


class JsonSkeletonExtractor(SkeletonExtractor):
    """Reads ``simscape_skeleton_<pose>.json`` files from a directory.

    These files are produced by ``export_default_skeleton.m`` (MATLAB-side
    helper next to the legacy Motion Capture Plotter tree).  When a file
    is missing, falls back to :func:`core.fallback_skeleton` (which
    derives the skeleton from the shared
    :func:`reference_golfer_setup` + :func:`forward_kinematics`).
    """

    def __init__(
        self,
        json_dir: str | Path,
        poses: tuple[str, ...] = ("TopofBackswing", "Impact"),
    ) -> None:
        self._dir = Path(json_dir)
        self._poses = tuple(poses)

    def list_poses(self) -> list[str]:
        return list(self._poses)

    def get_skeleton(self, pose_name: str) -> Skeleton:
        path = self._dir / f"simscape_skeleton_{pose_name}.json"
        return load_skeleton(path, fallback_pose=pose_name)


__all__ = ["SkeletonExtractor", "JsonSkeletonExtractor", "fallback_skeleton"]
