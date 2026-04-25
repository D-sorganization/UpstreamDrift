# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

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

from pathlib import Path
from typing import Any

from src.shared.python.ai.education import EducationSystem
from src.shared.python.ai.tool_registry import ToolCategory, ToolRegistry
from src.shared.python.ai.types import ExpertiseLevel
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Singleton holder for education system (avoids 'global' keyword)
_education_holder: dict[str, EducationSystem | None] = {"instance": None}


def _get_education_system() -> EducationSystem:
    """Get or create the education system singleton."""
    if _education_holder["instance"] is None:
        _education_holder["instance"] = EducationSystem()

    system = _education_holder["instance"]
    if system is None:  # Ensure it is not None for mypy
        raise ValueError("DbC Blocked: Precondition failed.")
    return system


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


def _register_list_sample_files_tool(registry: ToolRegistry) -> None:
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


def _register_load_c3d_tool(registry: ToolRegistry):
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

        except ImportError as e:
            return {"success": False, "error": f"Failed to load C3D: {e}"}

    return load_c3d


def _register_marker_info_tool(registry: ToolRegistry, load_c3d_fn) -> None:
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

        result = load_c3d_fn(file_path)
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


def _register_data_tools(registry: ToolRegistry) -> None:
    """Register data loading and management tools."""
    _register_list_sample_files_tool(registry)
    load_c3d_fn = _register_load_c3d_tool(registry)
    _register_marker_info_tool(registry, load_c3d_fn)


def _pendulum_demo_trajectory(
    duration_s: float = 1.0,
    dt: float = 0.02,
    length: float = 1.0,
    mass: float = 1.0,
    gravity: float = 9.81,
) -> dict[str, list[float]]:
    """Build a closed-form 1-DoF pendulum trajectory.

    Uses ``q(t) = sin(2 pi t / duration)`` (small-angle friendly) so that
    ``q``, ``qdot``, ``qddot`` are available analytically for inverse
    dynamics unit tests independent of any installed physics engine.

    Returns a dict of parallel lists: ``t``, ``q``, ``qdot``, ``qddot``,
    and the torque sequence ``tau`` produced by the equation
    ``tau = m L^2 qddot + m g L sin(q)``.
    """
    import math

    if duration_s <= 0 or dt <= 0:
        raise ValueError("duration_s and dt must be positive")
    steps = int(duration_s / dt) + 1
    two_pi = 2.0 * math.pi
    t_list: list[float] = []
    q_list: list[float] = []
    qdot_list: list[float] = []
    qddot_list: list[float] = []
    tau_list: list[float] = []
    for i in range(steps):
        t = i * dt
        omega = two_pi / duration_s
        q = math.sin(omega * t)
        qdot = omega * math.cos(omega * t)
        qddot = -(omega * omega) * math.sin(omega * t)
        # tau = I qddot + m g L sin(q), I = m L^2 (point mass at end of rod)
        tau = mass * length * length * qddot + mass * gravity * length * math.sin(q)
        t_list.append(t)
        q_list.append(q)
        qdot_list.append(qdot)
        qddot_list.append(qddot)
        tau_list.append(tau)
    return {
        "t": t_list,
        "q": q_list,
        "qdot": qdot_list,
        "qddot": qddot_list,
        "tau": tau_list,
    }


def _available_engines() -> list[str]:
    """Return the names of physics engines importable in this process."""
    import importlib.util

    names = []
    for mod_name, label in (
        ("mujoco", "mujoco"),
        ("pydrake", "drake"),
        ("pinocchio", "pinocchio"),
    ):
        try:
            if importlib.util.find_spec(mod_name) is not None:
                names.append(label)
        except (ValueError, ModuleNotFoundError):
            continue
    return names


def _register_inverse_dynamics_tool(registry: ToolRegistry) -> None:
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
        file_path: str | None = None,
        engine: str = "mujoco",
    ) -> dict[str, Any]:
        """Run inverse dynamics simulation.

        If ``file_path`` is omitted (or not a real C3D file on disk), a
        closed-form single-DoF pendulum trajectory is synthesized and its
        torques are returned as a demo fixture. This makes the tool
        immediately useful without requiring a motion-capture file or a
        specific physics engine install.

        Args:
            file_path: Optional path to a C3D file.
            engine: Physics engine to use (mujoco, drake, pinocchio).

        Returns:
            Dict with ``success`` and either torques or an error.
        """
        valid_engines = ["mujoco", "drake", "pinocchio"]
        engine_lower = (engine or "mujoco").lower()
        if engine_lower not in valid_engines:
            return {
                "success": False,
                "error": f"Invalid engine. Choose from: {valid_engines}",
            }

        # Real IK/ID pipeline is not yet wired - return honest not-implemented
        # response rather than fake data.  Tracked in issue #3163.
        return {
            "success": False,
            "error": "not implemented",
            "issue": "#3163",
            "tool": "run_inverse_dynamics",
        }


def _register_interpret_torques_tool(registry: ToolRegistry) -> None:
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
        # Typical ranges for golf swing (approximate).
        # NOTE: these ranges are heuristic estimates; see issue #3163 for citation tracking
        # Sources: shoulder ranges follow Nesbit & Serrano (2005) J.Sports Sci. Med.;
        # hip ranges adapted from MacKenzie & Sprigings (2009); wrist ranges from
        # Kwon et al. (2012) kinetic analysis of driver swings. Values are
        # peak-magnitude order-of-magnitude references, not diagnostic thresholds.
        if shoulder_torque is None:
            raise ValueError("shoulder_torque must be provided")
        ranges = {
            "shoulder": {
                "low": 40,
                "typical": 80,
                "high": 150,
                "unit": "N·m",
                "source": "Nesbit & Serrano (2005)",
            },
            "hip": {
                "low": 60,
                "typical": 120,
                "high": 200,
                "unit": "N·m",
                "source": "MacKenzie & Sprigings (2009)",
            },
            "wrist": {
                "low": 10,
                "typical": 25,
                "high": 50,
                "unit": "N·m",
                "source": "Kwon et al. (2012)",
            },
        }

        def classify(value: float, range_info: dict[str, Any]) -> str:
            """Classify a torque value relative to its typical range."""
            if value is None:
                raise ValueError("value must be provided")
            if value < range_info["low"]:
                return "Below typical"
            if value <= range_info["high"]:
                return "Within typical range"
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


def _register_analysis_tools(registry: ToolRegistry) -> None:
    """Register analysis and simulation tools."""
    _register_inverse_dynamics_tool(registry)
    _register_interpret_torques_tool(registry)


def _register_explain_concept_tool(registry: ToolRegistry) -> None:
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
        if term is None:
            raise ValueError("term must be provided")
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


def _register_list_glossary_terms_tool(registry: ToolRegistry) -> None:
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

        terms = edu.list_terms(category=category) if category else edu.list_terms()

        categories = edu.list_categories()

        return {
            "terms": terms,
            "count": len(terms),
            "categories": categories,
            "filter": category,
        }


def _register_search_glossary_tool(registry: ToolRegistry) -> None:
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


def _register_education_tools(registry: ToolRegistry) -> None:
    """Register educational and explanation tools."""
    _register_explain_concept_tool(registry)
    _register_list_glossary_terms_tool(registry)
    _register_search_glossary_tool(registry)


def _register_cross_engine_validation_tool(registry: ToolRegistry) -> None:
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
        file_path: str | None = None,
        tolerance: float = 0.02,
    ) -> dict[str, Any]:
        """Run cross-engine validation on the demo pendulum trajectory.

        Compares the closed-form analytic torque against the identical
        trajectory evaluated symbolically for each available engine. For
        this demo, the "engine" computations are all the analytic form
        (since no physics engine is assumed installed), so max torque
        deltas are zero. Real engine-backed validation is performed when
        the corresponding physics engine is importable.

        Args:
            file_path: Reserved for future C3D input.
            tolerance: Acceptable max torque delta [N*m].

        Returns:
            Dict describing per-engine results and a pass/fail flag.
        """
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")

        # Real cross-engine validation requires invoking each available engine
        # and comparing its output against a reference. Returning success:True
        # with a hardcoded delta of 0.0 would falsely report PASS for engines
        # that were never actually run. Until per-engine result extraction is
        # wired, return an honest not-implemented response. See issue #3163.
        available = _available_engines()
        return {
            "success": False,
            "error": "not implemented",
            "issue": "#3163",
            "tool": "validate_cross_engine",
            "available_engines": available,
            "message": (
                "Cross-engine validation is not yet implemented. "
                "Each physics engine must be invoked and results compared; "
                "returning fake deltas of 0.0 would misreport as PASS."
            ),
        }


def _register_energy_conservation_tool(registry: ToolRegistry) -> None:
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
        """Compute energy drift on the demo pendulum trajectory.

        Kinetic energy: ``KE = 0.5 * m * L^2 * qdot^2``.
        Potential energy: ``PE = m * g * L * (1 - cos(q))`` (zero at q=0).

        For a driven pendulum the total mechanical energy oscillates along
        with the applied torque's work. This routine returns the peak and
        RMS deviation of ``E = KE + PE`` relative to its initial value and
        reports whether the drift fits within ``tolerance`` (as a fraction
        of the peak total energy).

        Args:
            tolerance: Acceptable energy drift as a fraction of peak E.

        Returns:
            Dict with success, energy stats and pass/fail.
        """
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")

        # Energy conservation check requires a real simulation trajectory from
        # a physics engine - not a synthetic analytic fixture.  Returning
        # honest not-implemented rather than misleading drift=0 on a
        # hand-crafted pendulum.  Tracked in issue #3163.
        return {
            "success": False,
            "error": "not implemented",
            "issue": "#3163",
            "tool": "check_energy_conservation",
        }


def _register_list_physics_engines_tool(registry: ToolRegistry) -> None:
    @registry.register(
        name="list_physics_engines",
        description="List available physics engines and their status.",
        category=ToolCategory.CONFIGURATION,
        expertise_level=1,
    )
    def list_physics_engines() -> dict[str, Any]:
        """List available physics engines.

        Introspects via :func:`_available_engines` (importlib.util.find_spec),
        so results reflect the current process environment rather than a
        hardcoded list.
        """
        available = set(_available_engines())
        engines = []
        for mod_key, display in (
            ("mujoco", "MuJoCo"),
            ("drake", "Drake"),
            ("pinocchio", "Pinocchio"),
        ):
            engines.append(
                {
                    "name": display,
                    "status": "available" if mod_key in available else "not installed",
                }
            )
        return {
            "engines": engines,
            "available_count": len(available),
            "message": f"{len(available)} of 3 physics engines available.",
        }


def _register_validation_tools(registry: ToolRegistry) -> None:
    """Register validation and verification tools."""
    _register_cross_engine_validation_tool(registry)
    _register_energy_conservation_tool(registry)
    _register_list_physics_engines_tool(registry)
