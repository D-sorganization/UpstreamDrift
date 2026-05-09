"""OpenSim skeleton provider for starting-pose matcher.

This module provides an OpenSim-based skeleton provider that maps OpenSim
model bodies/frames/markers to the shared matcher skeleton vocabulary.

Required vocabulary:
    hip, spine, torso, hub, ls, rs, le, re, lw, rw, mp, ch
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Body/frame/marker name mapping from OpenSim to matcher vocabulary
# These are the standard names expected by the starting-pose matcher
OPENSIM_TO_MATCHER_VOCAB: dict[str, str] = {
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
MATCHER_TO_OPENSIM: dict[str, str] = {v: k for k, v in OPENSIM_TO_MATCHER_VOCAB.items()}


class OpenSimNotAvailableError(Exception):
    """Raised when OpenSim is not installed but an OpenSim provider is requested."""


class OpenSimProviderError(Exception):
    """Raised when there's an error with the OpenSim provider configuration."""


class OpenSimSkeletonProvider:
    """Provides skeleton data from an OpenSim model.

    This provider loads an OpenSim .osim model and extracts body/frame/marker
    positions to map them to the shared matcher skeleton vocabulary.

    Attributes:
        model_path: Path to the .osim model file.
        model: The OpenSim Model instance.
        state: The OpenSim State instance.
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_xml: str | None = None,
    ):
        """Initialize the OpenSim skeleton provider.

        Args:
            model_path: Path to the .osim model file.
            model_xml: Optional XML string for the model (alternative to path).

        Raises:
            OpenSimNotAvailableError: If OpenSim is not installed.
            OpenSimProviderError: If neither model_path nor model_xml is provided.
        """
        try:
            import opensim as osim
        except ImportError as e:
            raise OpenSimNotAvailableError(
                "OpenSim is not installed. Install with: pip install opensim"
            ) from e

        self._osim = osim

        if model_path is None and model_xml is None:
            raise OpenSimProviderError(
                "Either model_path or model_xml must be provided"
            )

        # Load model
        if model_xml is not None:
            # Write XML to temp file and load
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".osim", delete=False
            ) as f:
                f.write(model_xml)
                temp_path = f.name
            self.model = self._osim.Model(temp_path)
            import os

            os.unlink(temp_path)
        else:
            self.model = self._osim.Model(model_path)

        # Initialize the model
        self.model.initSystem()
        self.state = self.model.getState()

        # Build body name mapping
        self._body_name_to_index: dict[str, int] = {}
        for i in range(self.model.getBodySet().getSize()):
            body = self.model.getBodySet().get(i)
            self._body_name_to_index[body.getName()] = i

        # Build marker name mapping
        self._marker_name_to_index: dict[str, int] = {}
        for i in range(self.model.getMarkerSet().getSize()):
            marker = self.model.getMarkerSet().get(i)
            self._marker_name_to_index[marker.getName()] = i

        # Validate that required vocabulary is available
        self._validate_vocabulary()

    def _validate_vocabulary(self) -> None:
        """Validate that the model has bodies/markers for the required vocabulary."""
        missing = []
        for matcher_name, opensim_name in MATCHER_TO_OPENSIM.items():
            # Check both bodies and markers
            found = (
                opensim_name in self._body_name_to_index
                or opensim_name in self._marker_name_to_index
            )
            if not found:
                missing.append(f"{matcher_name} (mapped from '{opensim_name}')")

        if missing:
            raise OpenSimProviderError(
                f"Missing required body/marker mappings in OpenSim model: {', '.join(missing)}"
            )

    def _get_body_position(self, body_index: int) -> tuple[float, float, float]:
        """Get the position of a body in ground coordinates.

        Args:
            body_index: The OpenSim body index.

        Returns:
            Tuple of (x, y, z) coordinates in meters.
        """
        body = self.model.getBodySet().get(body_index)
        transform = body.getTransformInGround(self.state)
        position = transform.p
        return (float(position[0]), float(position[1]), float(position[2]))

    def _get_marker_position(self, marker_index: int) -> tuple[float, float, float]:
        """Get the position of a marker in ground coordinates.

        Args:
            marker_index: The OpenSim marker index.

        Returns:
            Tuple of (x, y, z) coordinates in meters.
        """
        marker = self.model.getMarkerSet().get(marker_index)
        position = marker.getLocationInGround(self.state)
        return (float(position[0]), float(position[1]), float(position[2]))

    def get_skeleton(
        self, coordinates: dict[str, float] | None = None
    ) -> dict[str, NDArray[np.float64]]:
        """Get skeleton joint positions from OpenSim model.

        Args:
            coordinates: Optional dictionary of coordinate name to value mappings.
                        If provided, the model state will be updated before extracting positions.

        Returns:
            Dictionary mapping matcher vocabulary names to 3D positions (in meters).
        """
        import numpy as np

        if coordinates is not None:
            # Apply coordinate values
            for coord_name, value in coordinates.items():
                try:
                    coord = self.model.getCoordinateSet().get(coord_name)
                    coord.setValue(self.state, value)
                except (RuntimeError, KeyError):
                    # Coordinate not found, skip it
                    pass

            # Realize to position stage
            self.model.realizePosition(self.state)

        skeleton: dict[str, NDArray[np.float64]] = {}

        for matcher_name, opensim_name in MATCHER_TO_OPENSIM.items():
            # Try body first, then marker
            if opensim_name in self._body_name_to_index:
                body_index = self._body_name_to_index[opensim_name]
                pos = self._get_body_position(body_index)
                skeleton[matcher_name] = np.array(pos, dtype=np.float64)
            elif opensim_name in self._marker_name_to_index:
                marker_index = self._marker_name_to_index[opensim_name]
                pos = self._get_marker_position(marker_index)
                skeleton[matcher_name] = np.array(pos, dtype=np.float64)

        return skeleton

    def get_available_bodies(self) -> list[str]:
        """Get list of available body names in the model."""
        return list(self._body_name_to_index.keys())

    def get_available_markers(self) -> list[str]:
        """Get list of available marker names in the model."""
        return list(self._marker_name_to_index.keys())


def create_provider(
    model_path: str | None = None,
    model_xml: str | None = None,
) -> OpenSimSkeletonProvider:
    """Create an OpenSim skeleton provider.

    This is the factory function used by the starting-pose matcher
    registry to instantiate an OpenSim provider.

    Args:
        model_path: Path to the .osim model file.
        model_xml: Optional XML string for the model.

    Returns:
        A configured OpenSimSkeletonProvider instance.
    """
    return OpenSimSkeletonProvider(model_path=model_path, model_xml=model_xml)
