"""Workflow definitions for the AI assistant.

Pre-built workflow definitions for common analysis tasks.
Each workflow is defined by factory functions that assemble
``WorkflowStep`` objects into a ``Workflow``.

Workflows defined here:
- ``first_analysis`` — beginner tutorial for first-time users
- ``c3d_import`` — motion capture data import and validation
- ``inverse_dynamics`` — full inverse dynamics analysis
- ``cross_engine_validation`` — multi-engine result comparison
- ``drift_control_decomposition`` — advanced decomposition analysis
"""

from __future__ import annotations

from src.shared.python.ai.types import ExpertiseLevel
from src.shared.python.ai.workflow_engine import (
    RecoveryStrategy,
    Workflow,
    WorkflowStep,
)

# ── First Analysis (beginner) ─────────────────────────────────────


def _first_analysis_welcome_step() -> WorkflowStep:
    return WorkflowStep(
        id="welcome",
        name="Welcome",
        description="Introduction to golf swing analysis",
        educational_content={
            "beginner": (
                "Welcome to the Golf Modeling Suite! 🏌️\n\n"
                "This workflow will guide you through analyzing a golf swing "
                "using physics simulations. By the end, you'll understand:\n"
                "- How to load motion capture data\n"
                "- What inverse dynamics tells us\n"
                "- How to interpret joint torques"
            ),
            "intermediate": (
                "This workflow covers inverse dynamics analysis using "
                "your choice of physics engine (MuJoCo, Drake, or Pinocchio)."
            ),
        },
    )


def _first_analysis_select_file_step() -> WorkflowStep:
    return WorkflowStep(
        id="select_file",
        name="Select Motion Data",
        description="Choose a C3D file containing golf swing motion capture",
        tool_name="list_sample_files",
        educational_content={
            "beginner": (
                "C3D files contain motion capture data - 3D positions of "
                "markers placed on the body during a golf swing.\n\n"
                "Think of it like a detailed recording of exactly how "
                "the body moved during the swing."
            ),
        },
    )


def _first_analysis_load_step() -> WorkflowStep:
    return WorkflowStep(
        id="load_data",
        name="Load Motion Data",
        description="Load and validate the C3D file",
        tool_name="load_c3d",
        educational_content={
            "beginner": (
                "When we load the C3D file, we're extracting:\n"
                "- Marker positions over time\n"
                "- Frame rate (usually 100-500 Hz)\n"
                "- Duration of the recording\n\n"
                "The system will also check that the data is valid."
            ),
        },
    )


def _first_analysis_simulation_step() -> WorkflowStep:
    return WorkflowStep(
        id="run_simulation",
        name="Run Physics Simulation",
        description="Compute joint torques using inverse dynamics",
        tool_name="run_inverse_dynamics",
        timeout=600.0,
        educational_content={
            "beginner": (
                "Inverse dynamics answers the question:\n"
                "'What forces caused this motion?'\n\n"
                "By knowing how the body moved, we can calculate the "
                "torques (rotational forces) at each joint.\n\n"
                "This helps us understand muscle contribution and "
                "potential injury risks."
            ),
            "advanced": (
                "Using the equation τ = M(q)q̈ + C(q,q̇) + g(q), "
                "we solve for joint torques given the measured motion."
            ),
        },
    )


def _first_analysis_interpret_step() -> WorkflowStep:
    return WorkflowStep(
        id="interpret_results",
        name="Interpret Results",
        description="Understand what the torque data means",
        tool_name="interpret_torques",
        educational_content={
            "beginner": (
                "The results show torques at each joint:\n"
                "- Shoulder: typically 50-150 N·m during downswing\n"
                "- Hip: often the highest torques (100-200 N·m)\n"
                "- Wrist: lower but important for club control\n\n"
                "Higher torques mean more muscle effort at that joint."
            ),
        },
    )


def create_first_analysis_workflow() -> Workflow:
    """Create the 'first_analysis' beginner workflow.

    This workflow guides beginners through their first
    complete analysis of a golf swing.

    Returns:
        Complete Workflow object.
    """
    workflow = Workflow(
        id="first_analysis",
        name="Your First Golf Swing Analysis",
        description=(
            "A step-by-step guide to analyzing your first golf swing. "
            "This workflow will help you load data, run simulations, "
            "and interpret results."
        ),
        expertise_level=ExpertiseLevel.BEGINNER,
        estimated_duration=30,
        tags=["beginner", "tutorial", "inverse-dynamics"],
    )

    workflow.add_step(_first_analysis_welcome_step())
    workflow.add_step(_first_analysis_select_file_step())
    workflow.add_step(_first_analysis_load_step())
    workflow.add_step(_first_analysis_simulation_step())
    workflow.add_step(_first_analysis_interpret_step())

    return workflow


# ── C3D Import ─────────────────────────────────────────────────────


def _c3d_import_intro_step() -> WorkflowStep:
    return WorkflowStep(
        id="intro",
        name="C3D Import Introduction",
        description="Overview of the C3D import process",
        educational_content={
            "beginner": (
                "C3D (Coordinate 3D) is the standard format for motion "
                "capture data. It contains 3D marker positions recorded "
                "during movement.\n\n"
                "This workflow will help you:\n"
                "- Load a C3D file\n"
                "- Inspect its contents\n"
                "- Verify data quality"
            ),
            "intermediate": (
                "C3D files contain both analog and point data streams. "
                "We'll extract marker trajectories, check for gaps, and "
                "prepare the data for physics simulation."
            ),
        },
    )


def _c3d_import_list_files_step() -> WorkflowStep:
    return WorkflowStep(
        id="list_files",
        name="Browse Available Files",
        description="List available C3D files for import",
        tool_name="list_sample_files",
        educational_content={
            "beginner": (
                "Sample files are included to help you learn. "
                "You can also import your own C3D files."
            ),
        },
    )


def _c3d_import_load_step() -> WorkflowStep:
    return WorkflowStep(
        id="load_file",
        name="Load C3D File",
        description="Load the selected C3D file",
        tool_name="load_c3d",
        on_failure=RecoveryStrategy.ASK_USER,
        educational_content={
            "beginner": (
                "Loading extracts marker positions, frame rate, "
                "and other metadata from the file."
            ),
            "intermediate": (
                "The loader validates file structure, checks for "
                "corrupted frames, and converts units if necessary."
            ),
        },
    )


def _c3d_import_inspect_step() -> WorkflowStep:
    return WorkflowStep(
        id="inspect_markers",
        name="Inspect Marker Configuration",
        description="Examine the markers present in the data",
        tool_name="get_marker_info",
        educational_content={
            "beginner": (
                "Markers are reflective balls placed on the body. "
                "Each marker has a name and represents a specific "
                "body location."
            ),
            "intermediate": (
                "The marker set determines which body segments can "
                "be tracked. Common sets include Plug-in Gait and "
                "Cleveland Clinic marker sets."
            ),
        },
    )


def create_c3d_import_workflow() -> Workflow:
    """Create the 'c3d_import' workflow for importing motion capture data.

    This workflow guides users through importing and validating
    C3D motion capture files.

    Returns:
        Complete Workflow object.
    """
    workflow = Workflow(
        id="c3d_import",
        name="C3D Motion Capture Import",
        description=(
            "Import and validate C3D motion capture data. This workflow "
            "guides you through loading, inspecting, and preparing motion "
            "data for analysis."
        ),
        expertise_level=ExpertiseLevel.INTERMEDIATE,
        estimated_duration=15,
        tags=["c3d", "import", "motion-capture", "data-loading"],
    )

    workflow.add_step(_c3d_import_intro_step())
    workflow.add_step(_c3d_import_list_files_step())
    workflow.add_step(_c3d_import_load_step())
    workflow.add_step(_c3d_import_inspect_step())

    return workflow


# ── Inverse Dynamics ───────────────────────────────────────────────


def _id_workflow_steps() -> list[WorkflowStep]:
    return [
        WorkflowStep(
            id="intro",
            name="Inverse Dynamics Overview",
            description="Introduction to inverse dynamics analysis",
            educational_content={
                "beginner": (
                    "Inverse dynamics calculates the forces that caused "
                    "a movement. Given how the body moved, we determine "
                    "what muscle forces were required."
                ),
                "intermediate": (
                    "Using Newton-Euler equations and measured kinematics, "
                    "we solve for joint torques. This requires accurate "
                    "segment inertial properties and kinematic data."
                ),
                "advanced": (
                    "The inverse dynamics problem solves τ = M(q)q̈ + C(q,q̇)q̇ + g(q) "
                    "where M is the mass matrix, C represents Coriolis/centrifugal "
                    "effects, and g is the gravity vector."
                ),
            },
        ),
        WorkflowStep(
            id="select_engine",
            name="Select Physics Engine",
            description="Choose the physics engine for simulation",
            tool_name="list_physics_engines",
            educational_content={
                "intermediate": (
                    "Available engines:\n"
                    "- MuJoCo: Fast, GPU-accelerated, good for muscle models\n"
                    "- Drake: Robust, precise, good for contacts\n"
                    "- Pinocchio: Efficient, analytical derivatives"
                ),
            },
        ),
        WorkflowStep(
            id="load_data",
            name="Load Motion Data",
            description="Load C3D data for analysis",
            tool_name="load_c3d",
            on_failure=RecoveryStrategy.ASK_USER,
        ),
        WorkflowStep(
            id="run_id",
            name="Run Inverse Dynamics",
            description="Execute inverse dynamics calculation",
            tool_name="run_inverse_dynamics",
            timeout=600.0,
            on_failure=RecoveryStrategy.RETRY,
            educational_content={
                "beginner": (
                    "The simulation is now calculating joint torques. "
                    "This may take a few minutes depending on data length."
                ),
            },
        ),
        WorkflowStep(
            id="check_energy",
            name="Verify Energy Conservation",
            description="Check physical plausibility of results",
            tool_name="check_energy_conservation",
            educational_content={
                "intermediate": (
                    "Energy conservation is a key validation metric. "
                    "Large energy violations indicate simulation errors."
                ),
            },
        ),
        WorkflowStep(
            id="interpret",
            name="Interpret Torque Results",
            description="Analyze and interpret the calculated torques",
            tool_name="interpret_torques",
            educational_content={
                "beginner": (
                    "Joint torques tell us about muscle effort:\n"
                    "- High torques = high muscle demands\n"
                    "- Timing patterns reveal movement strategy\n"
                    "- Asymmetries may indicate technique issues"
                ),
            },
        ),
    ]


def create_inverse_dynamics_workflow() -> Workflow:
    """Create the 'inverse_dynamics' workflow for full ID analysis.

    This workflow performs complete inverse dynamics analysis
    with detailed configuration options.

    Returns:
        Complete Workflow object.
    """
    workflow = Workflow(
        id="inverse_dynamics",
        name="Inverse Dynamics Analysis",
        description=(
            "Complete inverse dynamics analysis workflow. Calculate joint "
            "torques, muscle contributions, and analyze movement patterns."
        ),
        expertise_level=ExpertiseLevel.INTERMEDIATE,
        estimated_duration=45,
        tags=["inverse-dynamics", "torques", "analysis", "biomechanics"],
    )

    for step in _id_workflow_steps():
        workflow.add_step(step)

    return workflow


# ── Cross-Engine Validation ────────────────────────────────────────


def _cross_engine_validation_steps() -> list[WorkflowStep]:
    return [
        WorkflowStep(
            id="intro",
            name="Cross-Engine Validation Overview",
            description="Why cross-engine validation matters",
            educational_content={
                "intermediate": (
                    "Different physics engines use different algorithms. "
                    "Comparing results helps identify numerical issues and "
                    "ensures scientific validity."
                ),
                "advanced": (
                    "Cross-validation detects:\n"
                    "- Numerical instabilities\n"
                    "- Integration method artifacts\n"
                    "- Contact model discrepancies\n"
                    "- Solver convergence issues"
                ),
            },
        ),
        WorkflowStep(
            id="check_engines",
            name="Check Available Engines",
            description="Verify which physics engines are available",
            tool_name="list_physics_engines",
            educational_content={
                "intermediate": (
                    "At least two engines are needed for cross-validation. "
                    "Results are compared based on:\n"
                    "- Joint torque magnitudes\n"
                    "- Peak timing\n"
                    "- Energy conservation"
                ),
            },
        ),
        WorkflowStep(
            id="load_data",
            name="Load Motion Data",
            description="Load C3D data for multi-engine analysis",
            tool_name="load_c3d",
        ),
        WorkflowStep(
            id="run_validation",
            name="Run Cross-Engine Validation",
            description="Execute simulations on multiple engines and compare",
            tool_name="validate_cross_engine",
            timeout=900.0,
            on_failure=RecoveryStrategy.ASK_USER,
            educational_content={
                "advanced": (
                    "Validation metrics:\n"
                    "- RMS error between engines\n"
                    "- Correlation coefficient\n"
                    "- Peak torque agreement\n"
                    "- Energy drift comparison\n\n"
                    "Acceptable agreement: RMS < 5%, r > 0.99"
                ),
            },
        ),
        WorkflowStep(
            id="energy_check",
            name="Energy Conservation Analysis",
            description="Verify energy conservation across engines",
            tool_name="check_energy_conservation",
            educational_content={
                "advanced": (
                    "Energy conservation violations should be:\n"
                    "- < 1% for constrained systems\n"
                    "- < 5% for systems with contacts\n"
                    "- Consistent across engines"
                ),
            },
        ),
    ]


def create_cross_engine_validation_workflow() -> Workflow:
    """Create the 'cross_engine_validation' workflow.

    This workflow validates results by comparing across multiple
    physics engines to ensure robustness.

    Returns:
        Complete Workflow object.
    """
    workflow = Workflow(
        id="cross_engine_validation",
        name="Cross-Engine Validation",
        description=(
            "Validate simulation results by comparing outputs from multiple "
            "physics engines. Ensures results are robust and not artifacts "
            "of a specific engine implementation."
        ),
        expertise_level=ExpertiseLevel.ADVANCED,
        estimated_duration=60,
        tags=["validation", "cross-engine", "robustness", "quality-assurance"],
    )

    for step in _cross_engine_validation_steps():
        workflow.add_step(step)

    return workflow


# ── Drift-Control Decomposition ────────────────────────────────────


def _drift_control_steps() -> list[WorkflowStep]:
    return [
        WorkflowStep(
            id="intro",
            name="Drift-Control Decomposition Overview",
            description="Introduction to drift-control decomposition",
            educational_content={
                "advanced": (
                    "Drift-control decomposition separates torques into:\n"
                    "- Passive drift: gravity, inertial coupling\n"
                    "- Active control: intentional muscle activation\n\n"
                    "This reveals the control strategy used during movement."
                ),
                "expert": (
                    "The decomposition solves:\n"
                    "τ_total = τ_drift + τ_control\n\n"
                    "where τ_drift = M(q)q̈_free + C(q,q̇)q̇ + g(q)\n"
                    "represents the torques needed to allow natural motion, "
                    "and τ_control represents active corrections."
                ),
            },
        ),
        WorkflowStep(
            id="load_data",
            name="Load Motion Data",
            description="Load high-quality motion capture data",
            tool_name="load_c3d",
            on_failure=RecoveryStrategy.ASK_USER,
            educational_content={
                "expert": (
                    "Drift-control analysis requires:\n"
                    "- High sampling rate (≥200 Hz)\n"
                    "- Low marker noise\n"
                    "- Complete marker visibility\n"
                    "- Accurate segment properties"
                ),
            },
        ),
        WorkflowStep(
            id="run_id",
            name="Compute Total Joint Torques",
            description="Calculate total torques via inverse dynamics",
            tool_name="run_inverse_dynamics",
            timeout=600.0,
        ),
        WorkflowStep(
            id="compute_drift",
            name="Calculate Drift Component",
            description="Compute passive drift torques",
            tool_name="run_inverse_dynamics",
            tool_arguments={"mode": "drift_only"},
            timeout=600.0,
            educational_content={
                "expert": (
                    "Drift torques represent what happens if the nervous "
                    "system provides no active control - the natural dynamics "
                    "of the linked segment system under gravity."
                ),
            },
        ),
        WorkflowStep(
            id="verify_energy",
            name="Verify Energy Conservation",
            description="Check energy conservation in decomposition",
            tool_name="check_energy_conservation",
            educational_content={
                "expert": (
                    "The control component should:\n"
                    "- Add energy during acceleration phases\n"
                    "- Remove energy during deceleration\n"
                    "- Show minimal energy when coasting"
                ),
            },
        ),
        WorkflowStep(
            id="cross_validate",
            name="Cross-Engine Validation",
            description="Validate decomposition across physics engines",
            tool_name="validate_cross_engine",
            timeout=900.0,
            on_failure=RecoveryStrategy.SKIP,
            educational_content={
                "expert": (
                    "Cross-engine validation ensures the decomposition "
                    "is robust to numerical implementation details."
                ),
            },
        ),
    ]


def create_drift_control_decomposition_workflow() -> Workflow:
    """Create the 'drift_control_decomposition' workflow.

    This advanced workflow decomposes control contributions
    into drift and corrective components.

    Returns:
        Complete Workflow object.
    """
    workflow = Workflow(
        id="drift_control_decomposition",
        name="Drift-Control Decomposition Analysis",
        description=(
            "Decompose joint torques into passive (drift) and active (control) "
            "components. Advanced analysis for understanding movement control "
            "strategies and neuromuscular coordination."
        ),
        expertise_level=ExpertiseLevel.EXPERT,
        estimated_duration=90,
        tags=["drift", "control", "decomposition", "expert", "neuromuscular"],
    )

    for step in _drift_control_steps():
        workflow.add_step(step)

    return workflow
