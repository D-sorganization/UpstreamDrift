"""Version information for Golf Modeling Suite."""

__version__ = "1.0.0"
__version_info__ = (1, 0, 0)

__title__ = "Golf Modeling Suite"
__description__ = "Professional biomechanical analysis platform for golf swing modeling"
__author__ = "Golf Modeling Suite Team"
__author_email__ = "support@golfmodelingsuite.com"
__license__ = "MIT"
__url__ = "https://github.com/D-sorganization/Golf_Modeling_Suite"

# Build information
__build_date__ = "2026-01-12"
__python_requires__ = ">=3.11"

# Feature flags for professional version
FEATURES = {
    "video_pose_estimation": True,
    "ball_flight_physics": True,
    "api_server": True,
    "multi_engine_support": True,
    "professional_visualization": True,
    "cloud_integration": True,
    "enterprise_features": False,  # Reserved for enterprise license
}

# Supported physics engines
SUPPORTED_ENGINES = ["mujoco", "drake", "pinocchio", "myosuite", "opensim"]

# Professional edition features
PROFESSIONAL_FEATURES = [
    "Cross-engine validation and comparison",
    "Video-based pose estimation with MediaPipe",
    "Ball flight physics with Magnus effect",
    "REST API for cloud integration",
    "Standardized model library",
    "Professional visualization suite",
    "Batch processing capabilities",
    "Export to multiple formats",
]
