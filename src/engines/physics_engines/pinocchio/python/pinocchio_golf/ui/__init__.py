"""UI components for Pinocchio GUI."""

from __future__ import annotations

# Check meshcat availability
try:
    import meshcat.geometry as g
    import meshcat.visualizer as viz

    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False
    g = None  # type: ignore
    viz = None  # type: ignore

if MESHCAT_AVAILABLE:
    from pinocchio.visualize import MeshcatVisualizer
else:
    MeshcatVisualizer = object  # Dummy class if missing
