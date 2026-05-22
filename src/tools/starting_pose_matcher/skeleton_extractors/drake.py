"""Drake skeleton provider for starting-pose matcher.

This module provides a Drake-based skeleton provider that maps Drake
model bodies/frames to the shared matcher skeleton vocabulary.

Required vocabulary:
    hip, spine, torso, hub, ls, rs, le, re, lw, rw, mp, ch
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Body/frame name mapping from Drake to matcher vocabulary
# These are the standard names expected by the starting-pose matcher
DRAKE_TO_MATCHER_VOCAB: dict[str, str] = {
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
MATCHER_TO_DRAKE: dict[str, str] = {v: k for k, v in DRAKE_TO_MATCHER_VOCAB.items()}


class DrakeNotAvailableError(Exception):
    """Raised when Drake is not installed but a Drake provider is requested."""


class DrakeProviderError(Exception):
    """Raised when there's an error with the Drake provider configuration."""


class DrakeSkeletonProvider:
    """Provides skeleton data from a Drake MultibodyPlant.

    This provider loads a Drake URDF/SDF model and extracts body/frame
    positions to map them to the shared matcher skeleton vocabulary.

    Attributes:
        model_path: Path to the URDF/SDF model file.
        plant: The Drake MultibodyPlant instance.
        diagram: The Drake DiagramBuilder instance.
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_xml: str | None = None,
    ):
        """Initialize the Drake skeleton provider.

        Args:
            model_path: Path to the URDF/SDF model file.
            model_xml: Optional XML string for the model (alternative to path).

        Raises:
            DrakeNotAvailableError: If Drake is not installed.
            DrakeProviderError: If neither model_path nor model_xml is provided.
        """
        try:
            from pydrake.multibody.plant import MultibodyPlant
            from pydrake.systems.framework import DiagramBuilder
        except ImportError as e:
            raise DrakeNotAvailableError(
                "Drake is not installed. Install with: pip install drake"
            ) from e

        self._MultibodyPlant = MultibodyPlant
        self._DiagramBuilder = DiagramBuilder

        if model_path is None and model_xml is None:
            raise DrakeProviderError("Either model_path or model_xml must be provided")

        # Create plant with default time step
        self.plant = MultibodyPlant(0.0)

        # Load model
        if model_xml is not None:
            from pydrake.multibody.parser import Parser
            import defusedxml.ElementTree as ET  # noqa: S314  # Security: defusedxml prevents XML attacks

            parser = Parser(self.plant)
            # Detect SDF vs URDF by parsing the XML root tag
            # This is more robust than regex which can match comments or miss
            # the root tag if it appears after a long prolog
            try:
                root = ET.fromstring(model_xml.strip())
                is_sdf = root.tag == "sdf"
            except ET.ParseError:
                # If XML parsing fails, fall back to URDF (Drake's default)
                is_sdf = False
            if is_sdf:
                parser.SetPackageMapAutoMerge(True)
                parser.AddModelFromString(model_xml, "sdf")
            else:
                parser.SetPackageMapAutoMerge(True)
                parser.AddModelFromString(model_xml, "urdf")
        else:
            from pydrake.multibody.parser import Parser

            parser = Parser(self.plant)
            parser.SetPackageMapAutoMerge(True)
            parser.AddModelFromFile(model_path)

        # Finalize the plant
        self.plant.Finalize()

        # Build body name to index mapping
        self._body_name_to_index: dict[str, int] = {}
        for i in range(self.plant.num_bodies()):
            body = self.plant.get_body(i)  # type: ignore[arg-type]
            self._body_name_to_index[body.name()] = i

        # Validate that required vocabulary is available
        self._validate_vocabulary()

    def _validate_vocabulary(self) -> None:
        """Validate that the model has bodies for the required vocabulary."""
        missing = []
        for matcher_name, drake_name in MATCHER_TO_DRAKE.items():
            if drake_name not in self._body_name_to_index:
                missing.append(f"{matcher_name} (mapped from '{drake_name}')")

        if missing:
            raise DrakeProviderError(
                f"Missing required body mappings in Drake model: {', '.join(missing)}"
            )

    def _get_body_position(
        self, body_index: int, context
    ) -> tuple[float, float, float]:
        """Get the position of a body in world coordinates.

        Args:
            body_index: The Drake body index.
            context: The Drake context.

        Returns:
            Tuple of (x, y, z) coordinates in meters.
        """
        body = self.plant.get_body(body_index)  # type: ignore[arg-type]
        pose = body.EvalBodyPoseInWorld(context)  # type: ignore[attr-defined]
        position = pose.translation()
        return (float(position[0]), float(position[1]), float(position[2]))

    def get_skeleton(
        self, positions: NDArray[np.float64] | None = None
    ) -> dict[str, NDArray[np.float64]]:
        """Get skeleton joint positions from Drake model.

        Args:
            positions: Optional position vector in Drake model order. If provided,
                      the model state will be updated before extracting positions.

        Returns:
            Dictionary mapping matcher vocabulary names to 3D positions (in meters).
        """
        import numpy as np

        # Create context
        context = self.plant.CreateDefaultContext()

        if positions is not None:
            # Set plant positions
            self.plant.SetPositions(context, positions)

        skeleton: dict[str, NDArray[np.float64]] = {}

        for matcher_name, drake_name in MATCHER_TO_DRAKE.items():
            if drake_name in self._body_name_to_index:
                body_index = self._body_name_to_index[drake_name]
                pos = self._get_body_position(body_index, context)
                skeleton[matcher_name] = np.array(pos, dtype=np.float64)

        return skeleton

    def get_available_bodies(self) -> list[str]:
        """Get list of available body names in the model."""
        return list(self._body_name_to_index.keys())


def create_provider(
    model_path: str | None = None,
    model_xml: str | None = None,
) -> DrakeSkeletonProvider:
    """Create a Drake skeleton provider.

    This is the factory function used by the starting-pose matcher
    registry to instantiate a Drake provider.

    Args:
        model_path: Path to the URDF/SDF model file.
        model_xml: Optional XML string for the model.

    Returns:
        A configured DrakeSkeletonProvider instance.
    """
    return DrakeSkeletonProvider(model_path=model_path, model_xml=model_xml)
