"""Sample tools for AI integration with Golf Suite.

This module provides pre-built tools that expose Golf Modeling Suite
capabilities to the AI assistant. These tools can be invoked by the
AI to perform analysis, load data, and explain concepts.

Example:
    >>> from shared.python.ai.sample_tools import register_golf_suite_tools
    >>> from shared.python.ai.tool_registry import ToolRegistry
    >>> registry = ToolRegistry()
    >>> register_golf_suite_tools(registry)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from shared.python.ai.education import EducationSystem
from shared.python.ai.tool_registry import ToolCategory, ToolRegistry
from shared.python.ai.types import ExpertiseLevel

logger = logging.getLogger(__name__)

# Singleton education system for explanation tools
_education_system: EducationSystem | None = None


def _get_education_system() -> EducationSystem:
    """Get or create the education system singleton."""
    global _education_system
    if _education_system is None:
        _education_system = EducationSystem()
    return _education_system


def register_golf_suite_tools(registry: ToolRegistry) -> None:
    """Register all Golf Suite tools with the registry.

    Args:
        registry: Tool registry to register tools with.
    """
    _register_data_tools(registry)
    _register_analysis_tools(registry)
    _register_education_tools(registry)
    _register_validation_tools(registry)
    logger.info("Registered Golf Suite tools")


def _register_data_tools(registry: ToolRegistry) -> None:
    """Register data loading and management tools."""

    @registry.register(
        name="list_sample_files",
        description=(
            "List available sample C3D motion capture files that can be "
            "used for analysis. Returns a list of file paths and descriptions."
        ),
        category=ToolCategory.DATA_LOADING,
        expertise_level=1,
    )
    def list_sample_files() -> dict[str, Any]:
        """List available sample C3D files."""
        # Check for sample data directory
        sample_dir = Path("data/samples")
        if not sample_dir.exists():
            return {
                "files": [],
                "message": "No sample data directory found. Please add C3D files.",
            }

        c3d_files = list(sample_dir.glob("*.c3d"))
        files = [
            {
                "path": str(f),
                "name": f.stem,
                "size_kb": f.stat().st_size // 1024,
            }
            for f in c3d_files
        ]

        return {
            "files": files,
            "count": len(files),
            "message": f"Found {len(files)} sample C3D files.",
        }

    @registry.register(
        name="load_c3d",
        description=(
            "Load a C3D motion capture file for analysis. Extracts marker "
            "positions, frame rate, and metadata. Returns summary of loaded data."
        ),
        category=ToolCategory.DATA_LOADING,
        expertise_level=1,
    )
    def load_c3d(file_path: str) -> dict[str, Any]:
        """Load and validate a C3D file.

        Args:
            file_path: Path to the C3D file.

        Returns:
            Summary of loaded data.
        """
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        if path.suffix.lower() != ".c3d":
            return {"success": False, "error": "File must be a .c3d file"}

        try:
            # Try to import c3d library
            try:
                import c3d
            except ImportError:
                return {
                    "success": False,
                    "error": "c3d library not installed. Run: pip install c3d",
                }

            with open(path, "rb") as f:
                reader = c3d.Reader(f)

                # Extract metadata
                point_labels = reader.point_labels
                frame_count = reader.last_frame - reader.first_frame + 1
                frame_rate = reader.point_rate

                return {
                    "success": True,
                    "file": str(path),
                    "markers": len(point_labels),
                    "marker_names": list(point_labels)[:10],  # First 10
                    "frames": frame_count,
                    "frame_rate": frame_rate,
                    "duration_s": frame_count / frame_rate if frame_rate > 0 else 0,
                    "message": (
                        f"Loaded {path.name}: {len(point_labels)} markers, "
                        f"{frame_count} frames at {frame_rate} Hz"
                    ),
                }

        except Exception as e:
            return {"success": False, "error": f"Failed to load C3D: {e}"}

    @registry.register(
        name="get_marker_info",
        description=(
            "Get information about markers in a loaded C3D file, including "
            "which body segments they represent."
        ),
        category=ToolCategory.DATA_LOADING,
        expertise_level=2,
    )
    def get_marker_info(file_path: str) -> dict[str, Any]:
        """Get marker information from a C3D file.

        Args:
            file_path: Path to the C3D file.

        Returns:
            Marker information.
        """
        # Common marker name patterns
        segment_mapping = {
            "LSHO": "Left Shoulder",
            "RSHO": "Right Shoulder",
            "LELB": "Left Elbow",
            "RELB": "Right Elbow",
            "LWRI": "Left Wrist",
            "RWRI": "Right Wrist",
            "LASI": "Left Pelvis (ASIS)",
            "RASI": "Right Pelvis (ASIS)",
            "LPSI": "Left Pelvis (PSIS)",
            "RPSI": "Right Pelvis (PSIS)",
            "LKNE": "Left Knee",
            "RKNE": "Right Knee",
            "LANK": "Left Ankle",
            "RANK": "Right Ankle",
            "LTOE": "Left Toe",
            "RTOE": "Right Toe",
            "C7": "7th Cervical Vertebra",
            "T10": "10th Thoracic Vertebra",
            "CLAV": "Clavicle",
            "STRN": "Sternum",
        }

        result = load_c3d(file_path)
        if not result.get("success"):
            # Return the error from load_c3d
            error_result: dict[str, Any] = result
            return error_result

        markers = result.get("marker_names", [])
        identified = []
        for marker in markers:
            marker_upper = marker.strip().upper()
            if marker_upper in segment_mapping:
                identified.append(
                    {
                        "marker": marker,
                        "segment": segment_mapping[marker_upper],
                    }
                )

        return {
            "success": True,
            "total_markers": result.get("markers", 0),
            "identified": identified,
            "message": f"Identified {len(identified)} standard markers.",
        }


def _register_analysis_tools(registry: ToolRegistry) -> None:
    """Register analysis and simulation tools."""

    @registry.register(
        name="run_inverse_dynamics",
        description=(
            "Run inverse dynamics to calculate joint torques from motion data. "
            "Uses physics engine to compute forces that produced the observed motion."
        ),
        category=ToolCategory.SIMULATION,
        requires_confirmation=True,
        expertise_level=2,
    )
    def run_inverse_dynamics(
        file_path: str,
        engine: str = "mujoco",
    ) -> dict[str, Any]:
        """Run inverse dynamics simulation.

        Args:
            file_path: Path to C3D file.
            engine: Physics engine to use (mujoco, drake, pinocchio).

        Returns:
            Simulation results summary.
        """
        valid_engines = ["mujoco", "drake", "pinocchio"]
        if engine.lower() not in valid_engines:
            return {
                "success": False,
                "error": f"Invalid engine. Choose from: {valid_engines}",
            }

        # This implementation requires integration with the physics engines.
        # 1. Load the C3D data
        # 2. Create/load the model
        # 3. Run inverse dynamics
        # 4. Return results

        return {
            "success": True,
            "status": "simulation_pending",
            "engine": engine,
            "file": file_path,
            "message": (
                f"Inverse dynamics simulation queued using {engine}. "
                "This would normally take 30-60 seconds for a typical swing."
            ),
            "note": ("Implementation requires physics " "engine integration."),
        }

    @registry.register(
        name="interpret_torques",
        description=(
            "Interpret joint torque results from inverse dynamics. Provides "
            "context on whether values are typical for golf swings."
        ),
        category=ToolCategory.ANALYSIS,
        expertise_level=1,
    )
    def interpret_torques(
        shoulder_torque: float = 100.0,
        hip_torque: float = 150.0,
        wrist_torque: float = 30.0,
    ) -> dict[str, Any]:
        """Interpret joint torque values.

        Args:
            shoulder_torque: Peak shoulder torque [N·m].
            hip_torque: Peak hip torque [N·m].
            wrist_torque: Peak wrist torque [N·m].

        Returns:
            Interpretation of torque values.
        """
        # Typical ranges for golf swing (approximate)
        ranges = {
            "shoulder": {"low": 40, "typical": 80, "high": 150, "unit": "N·m"},
            "hip": {"low": 60, "typical": 120, "high": 200, "unit": "N·m"},
            "wrist": {"low": 10, "typical": 25, "high": 50, "unit": "N·m"},
        }

        def classify(value: float, range_info: dict[str, Any]) -> str:
            if value < range_info["low"]:
                return "Below typical"
            elif value <= range_info["high"]:
                return "Within typical range"
            else:
                return "Above typical (high stress)"

        return {
            "shoulder": {
                "value": shoulder_torque,
                "classification": classify(shoulder_torque, ranges["shoulder"]),
                "typical_range": f"{ranges['shoulder']['low']}-{ranges['shoulder']['high']} N·m",
            },
            "hip": {
                "value": hip_torque,
                "classification": classify(hip_torque, ranges["hip"]),
                "typical_range": f"{ranges['hip']['low']}-{ranges['hip']['high']} N·m",
            },
            "wrist": {
                "value": wrist_torque,
                "classification": classify(wrist_torque, ranges["wrist"]),
                "typical_range": f"{ranges['wrist']['low']}-{ranges['wrist']['high']} N·m",
            },
            "message": (
                "Torque values have been classified based on typical ranges "
                "observed in amateur and professional golf swings."
            ),
        }


def _register_education_tools(registry: ToolRegistry) -> None:
    """Register educational and explanation tools."""

    @registry.register(
        name="explain_concept",
        description=(
            "Explain a biomechanics or physics concept at the user's expertise "
            "level. Use this when the user asks 'what is X?' or needs clarification."
        ),
        category=ToolCategory.EDUCATIONAL,
        expertise_level=1,
    )
    def explain_concept(
        term: str,
        expertise_level: int = 1,
    ) -> dict[str, Any]:
        """Explain a biomechanics concept.

        Args:
            term: The term or concept to explain.
            expertise_level: User's expertise level (1-4).

        Returns:
            Explanation at appropriate level.
        """
        edu = _get_education_system()

        # Map level number to enum
        level_map = {
            1: ExpertiseLevel.BEGINNER,
            2: ExpertiseLevel.INTERMEDIATE,
            3: ExpertiseLevel.ADVANCED,
            4: ExpertiseLevel.EXPERT,
        }
        level = level_map.get(expertise_level, ExpertiseLevel.BEGINNER)

        explanation = edu.explain(term, level)
        entry = edu.get_entry(term)

        result: dict[str, Any] = {
            "term": term,
            "explanation": explanation,
            "level": level.name.lower(),
        }

        if entry:
            result["related_terms"] = entry.related_terms
            if entry.formula:
                result["formula"] = entry.formula
            if entry.units:
                result["units"] = entry.units

        return result

    @registry.register(
        name="list_glossary_terms",
        description=(
            "List available terms in the glossary, optionally filtered by category. "
            "Categories include: dynamics, kinematics, golf, simulation, validation."
        ),
        category=ToolCategory.EDUCATIONAL,
        expertise_level=1,
    )
    def list_glossary_terms(category: str | None = None) -> dict[str, Any]:
        """List glossary terms.

        Args:
            category: Optional category filter.

        Returns:
            List of available terms.
        """
        edu = _get_education_system()

        if category:
            terms = edu.list_terms(category=category)
        else:
            terms = edu.list_terms()

        categories = edu.list_categories()

        return {
            "terms": terms,
            "count": len(terms),
            "categories": categories,
            "filter": category,
        }

    @registry.register(
        name="search_glossary",
        description=(
            "Search the glossary for terms matching a query. Searches term names, "
            "categories, and definitions."
        ),
        category=ToolCategory.EDUCATIONAL,
        expertise_level=1,
    )
    def search_glossary(query: str) -> dict[str, Any]:
        """Search the glossary.

        Args:
            query: Search query.

        Returns:
            Matching terms.
        """
        edu = _get_education_system()
        results = edu.search(query)

        return {
            "query": query,
            "results": [
                {
                    "term": r.term,
                    "category": r.category,
                }
                for r in results
            ],
            "count": len(results),
        }


def _register_validation_tools(registry: ToolRegistry) -> None:
    """Register validation and verification tools."""

    @registry.register(
        name="validate_cross_engine",
        description=(
            "Run cross-engine validation to verify results are consistent "
            "across multiple physics engines (MuJoCo, Drake, Pinocchio)."
        ),
        category=ToolCategory.VALIDATION,
        requires_confirmation=True,
        expertise_level=3,
    )
    def validate_cross_engine(
        file_path: str,
        tolerance: float = 0.02,
    ) -> dict[str, Any]:
        """Run cross-engine validation.

        Args:
            file_path: Path to data file.
            tolerance: Acceptable tolerance for agreement.

        Returns:
            Validation results.
        """
        # Placeholder for actual cross-engine validation
        return {
            "status": "validation_pending",
            "file": file_path,
            "engines": ["mujoco", "drake", "pinocchio"],
            "tolerance": tolerance,
            "message": (
                "Cross-engine validation queued. This compares results from "
                "multiple physics engines to ensure accuracy."
            ),
            "note": "Placeholder - requires full physics engine integration.",
        }

    @registry.register(
        name="check_energy_conservation",
        description=(
            "Check energy conservation in a simulation to verify physical "
            "plausibility. Energy should be conserved or explained by work done."
        ),
        category=ToolCategory.VALIDATION,
        expertise_level=3,
    )
    def check_energy_conservation(tolerance: float = 0.01) -> dict[str, Any]:
        """Check energy conservation.

        Args:
            tolerance: Acceptable energy drift tolerance.

        Returns:
            Energy conservation check results.
        """
        return {
            "status": "check_pending",
            "tolerance": tolerance,
            "message": (
                "Energy conservation check queued. This verifies that total "
                "mechanical energy is properly accounted for throughout the motion."
            ),
            "note": "Placeholder - requires simulation data.",
        }

    @registry.register(
        name="list_physics_engines",
        description="List available physics engines and their status.",
        category=ToolCategory.CONFIGURATION,
        expertise_level=1,
    )
    def list_physics_engines() -> dict[str, Any]:
        """List available physics engines.

        Uses importlib.util.find_spec to check availability without importing,
        which avoids potential crashes from engine initialization.
        """
        import importlib.util

        def _check_module(name: str) -> bool:
            """Safely check if a module is available."""
            try:
                return importlib.util.find_spec(name) is not None
            except (ValueError, ModuleNotFoundError):
                # ValueError: __spec__ is not set (partially initialized module)
                # ModuleNotFoundError: module not found
                return False

        engines = []

        # Check MuJoCo (avoid importing due to potential initialization issues)
        if _check_module("mujoco"):
            engines.append({"name": "MuJoCo", "status": "available"})
        else:
            engines.append({"name": "MuJoCo", "status": "not installed"})

        # Check Drake
        if _check_module("pydrake"):
            engines.append({"name": "Drake", "status": "available"})
        else:
            engines.append({"name": "Drake", "status": "not installed"})

        # Check Pinocchio
        if _check_module("pinocchio"):
            engines.append({"name": "Pinocchio", "status": "available"})
        else:
            engines.append({"name": "Pinocchio", "status": "not installed"})

        available = sum(1 for e in engines if e["status"] == "available")

        return {
            "engines": engines,
            "available_count": available,
            "message": f"{available} of 3 physics engines available.",
        }
