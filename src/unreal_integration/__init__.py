"""Unreal Engine Integration Module for Golf Modeling Suite.

This module provides a bridge between the Python physics backend and Unreal Engine
visualization frontend, enabling real-time streaming of simulation data, mesh loading,
skeleton mapping, and VR interaction support.

Architecture:
    Physics Backend (Python) → REST API/WebSocket → Unreal Frontend (Visualization)

Design Principles:
    - Design by Contract (DbC): Preconditions, postconditions, invariants
    - DRY: Reusable components and shared utilities
    - Orthogonal: Independent modules that can be mixed and matched
    - Reversible: Feature flags and configuration for flexibility

Modules:
    - data_models: Core data structures for Unreal communication
    - streaming: WebSocket and REST API streaming
    - mesh_loader: Multi-format mesh loading (GLTF/GLB/FBX/OBJ)
    - skeleton_mapper: Gaming skeleton to physics model mapping
    - visualization: Force vectors, trajectories, HUD data
    - vr_interaction: VR-specific controls and interactions
    - viewer_backends: Unified viewer backend abstraction

Usage:
    from src.unreal_integration import (
        UnrealDataFrame,
        UnrealStreamingServer,
        MeshLoader,
        SkeletonMapper,
    )

    # Create a streaming server
    server = UnrealStreamingServer(host="localhost", port=8765)

    # Stream simulation data
    async with server:
        for frame in simulation:
            await server.broadcast(UnrealDataFrame.from_physics_state(frame))
"""

# Import with graceful fallback for missing dependencies
try:
    from src.unreal_integration.data_models import (
        ClubState,
        ForceVector,
        JointState,
        SwingMetrics,
        UnrealDataFrame,
        Vector3,
        Quaternion,
        TrajectoryPoint,
        BallState,
        EnvironmentState,
    )
except ImportError as e:
    raise ImportError(f"Failed to import data_models: {e}") from e

try:
    from src.unreal_integration.streaming import (
        StreamingConfig,
        UnrealStreamingServer,
        StreamingState,
        ControlMessage,
        ControlAction,
        FrameBuffer,
    )
except ImportError as e:
    raise ImportError(f"Failed to import streaming: {e}") from e

try:
    from src.unreal_integration.mesh_loader import (
        LoadedMesh,
        MeshLoader,
        MeshFormat,
        MeshVertex,
        MeshFace,
        MeshMaterial,
        MeshSkeleton,
        MeshBone,
    )
except ImportError as e:
    raise ImportError(f"Failed to import mesh_loader: {e}") from e

try:
    from src.unreal_integration.skeleton_mapper import (
        BoneMapping,
        SkeletonMapper,
        SkeletonType,
        MappingProfile,
        PoseTransform,
    )
except ImportError as e:
    raise ImportError(f"Failed to import skeleton_mapper: {e}") from e

try:
    from src.unreal_integration.visualization import (
        ForceVectorRenderer,
        TrajectoryRenderer,
        HUDDataProvider,
        VisualizationConfig,
        RenderData,
        VisualizationType,
    )
except ImportError as e:
    raise ImportError(f"Failed to import visualization: {e}") from e

try:
    from src.unreal_integration.vr_interaction import (
        VRControllerState,
        VRHeadsetState,
        VRInteractionManager,
        VRLocomotionMode,
        VRControllerHand,
        VRInteractionMode,
        VRGesture,
    )
except ImportError as e:
    raise ImportError(f"Failed to import vr_interaction: {e}") from e

try:
    from src.unreal_integration.viewer_backends import (
        ViewerBackend,
        MeshcatBackend,
        MockBackend,
        ViewerConfig,
        BackendType,
        create_viewer,
    )
except ImportError as e:
    raise ImportError(f"Failed to import viewer_backends: {e}") from e

__all__ = [
    # Data Models
    "UnrealDataFrame",
    "JointState",
    "ForceVector",
    "ClubState",
    "SwingMetrics",
    "Vector3",
    "Quaternion",
    "TrajectoryPoint",
    "BallState",
    "EnvironmentState",
    # Streaming
    "UnrealStreamingServer",
    "StreamingConfig",
    "StreamingState",
    "ControlMessage",
    "ControlAction",
    "FrameBuffer",
    # Mesh Loading
    "MeshLoader",
    "LoadedMesh",
    "MeshFormat",
    "MeshVertex",
    "MeshFace",
    "MeshMaterial",
    "MeshSkeleton",
    "MeshBone",
    # Skeleton Mapping
    "SkeletonMapper",
    "SkeletonType",
    "BoneMapping",
    "MappingProfile",
    "PoseTransform",
    # Visualization
    "ForceVectorRenderer",
    "TrajectoryRenderer",
    "HUDDataProvider",
    "VisualizationConfig",
    "RenderData",
    "VisualizationType",
    # VR Interaction
    "VRInteractionManager",
    "VRControllerState",
    "VRHeadsetState",
    "VRLocomotionMode",
    "VRControllerHand",
    "VRInteractionMode",
    "VRGesture",
    # Viewer Backends
    "ViewerBackend",
    "MeshcatBackend",
    "MockBackend",
    "ViewerConfig",
    "BackendType",
    "create_viewer",
]

__version__ = "1.0.0"
