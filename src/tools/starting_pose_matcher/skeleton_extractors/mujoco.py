"""MuJoCo skeleton provider for starting-pose matcher.

This module provides a MuJoCo-based skeleton provider that maps MuJoCo
model bodies/sites to the shared matcher skeleton vocabulary.

Required vocabulary:
    hip, spine, torso, hub, ls, rs, le, re, lw, rw, mp, ch
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Body/site name mapping from MuJoCo to matcher vocabulary
# These are the standard names expected by the starting-pose matcher
MUJOCO_TO_MATCHER_VOCAB: dict[str, str] = {
    # Lower body
    "hip": "hip",
    "pelvis": "hip",
    # Spine
    "spine": "spine",
    "torso": "torso",
    # Upper body
    "hub": "hub",
    # Shoulders
    "left_shoulder": "ls",
    "right_shoulder": "rs",
    # Elbows
    "left_elbow": "le",
    "right_elbow": "re",
    # Wrists/Hands
    "left_wrist": "lw",
    "right_wrist": "rw",
    # Mid-point (between hands)
    "midpoint": "mp",
    # Clubhead
    "clubhead": "ch",
}

# Reverse mapping for lookup
MATCHER_TO_MUJOCO: dict[str, str] = {v: k for k, v in MUJOCO_TO_MATCHER_VOCAB.items()}


class MuJoCoNotAvailableError(Exception):
    """Raised when MuJoCo is not installed but a MuJoCo provider is requested."""


class MuJoCoProviderError(Exception):
    """Raised when there's an error with the MuJoCo provider configuration."""


class MuJoCoSkeletonProvider:
    """Provides skeleton data from a MuJoCo model.

    This provider loads a MuJoCo MJCF/XML model and extracts body/site
    positions to map them to the shared matcher skeleton vocabulary.

    Attributes:
        model_path: Path to the MJCF/XML model file.
        model: The MuJoCo MjModel instance.
        data: The MuJoCo MjData instance.
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_xml: str | None = None,
    ):
        """Initialize the MuJoCo skeleton provider.

        Args:
            model_path: Path to the MJCF/XML model file.
            model_xml: Optional XML string for the model (alternative to path).

        Raises:
            MuJoCoNotAvailableError: If MuJoCo is not installed.
            MuJoCoProviderError: If neither model_path nor model_xml is provided.
        """
        try:
            import mujoco
        except ImportError as e:
            raise MuJoCoNotAvailableError(
                "MuJoCo is not installed. Install with: pip install mujoco"
            ) from e

        self._mujoco = mujoco

        if model_path is None and model_xml is None:
            raise MuJoCoProviderError("Either model_path or model_xml must be provided")

        if model_xml is not None:
            self.model = self._mujoco.MjModel.from_xml_string(model_xml)
        else:
            self.model = self._mujoco.MjModel.from_xml_path(model_path)

        self.data = self._mujoco.MjData(self.model)

        # Build body name to ID mapping
        self._body_name_to_id: dict[str, int] = {
            self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_BODY, i): i
            for i in range(self.model.nbody)
        }

        # Validate that required vocabulary is available
        self._validate_vocabulary()

    def _validate_vocabulary(self) -> None:
        """Validate that the model has bodies for the required vocabulary."""
        missing = []
        for matcher_name, mujoco_name in MATCHER_TO_MUJOCO.items():
            if mujoco_name not in self._body_name_to_id:
                missing.append(f"{matcher_name} (mapped from '{mujoco_name}')")

        if missing:
            raise MuJoCoProviderError(
                f"Missing required body mappings in MuJoCo model: {', '.join(missing)}"
            )

    def _get_body_position(self, name: str) -> tuple[float, float, float]:
        """Get the position of a body in world coordinates.

        Args:
            name: The MuJoCo body name.

        Returns:
            Tuple of (x, y, z) coordinates in meters.
        """
        body_id = self._body_name_to_id.get(name)
        if body_id is None:
            raise MuJoCoProviderError(f"Body '{name}' not found in model")

        # Get the body position from xipos (inertial frame position)
        pos = self.data.xipos[body_id]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def get_skeleton(
        self, qpos: NDArray[np.float64] | None = None
    ) -> dict[str, NDArray[np.float64]]:
        """Get skeleton joint positions from MuJoCo model.

        Args:
            qpos: Optional pose vector in MuJoCo qpos order. If provided,
                  the model state will be updated before extracting positions.

        Returns:
            Dictionary mapping matcher vocabulary names to 3D positions (in meters).
        """
        import numpy as np

        if qpos is not None:
            self.data.qpos[:] = qpos
            self._mujoco.mj_forward(self.model, self.data)

        skeleton: dict[str, NDArray[np.float64]] = {}

        for matcher_name, mujoco_name in MATCHER_TO_MUJOCO.items():
            if mujoco_name in self._body_name_to_id:
                pos = self._get_body_position(mujoco_name)
                skeleton[matcher_name] = np.array(pos, dtype=np.float64)

        return skeleton

    def get_available_bodies(self) -> list[str]:
        """Get list of available body names in the model."""
        return list(self._body_name_to_id.keys())


def create_provider(
    model_path: str | None = None,
    model_xml: str | None = None,
) -> MuJoCoSkeletonProvider:
    """Create a MuJoCo skeleton provider.

    This is the factory function used by the starting-pose matcher
    registry to instantiate a MuJoCo provider.

    Args:
        model_path: Path to the MJCF/XML model file.
        model_xml: Optional XML string for the model.

    Returns:
        A configured MuJoCoSkeletonProvider instance.
    """
    return MuJoCoSkeletonProvider(model_path=model_path, model_xml=model_xml)
