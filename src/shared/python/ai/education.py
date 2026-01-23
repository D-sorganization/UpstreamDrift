"""Educational content system for the AI Assistant.

This module provides multi-level explanations of biomechanics concepts,
enabling the AI to adjust complexity based on user expertise.

Example:
    >>> from shared.python.ai.education import EducationSystem, ExpertiseLevel
    >>> edu = EducationSystem()
    >>> explanation = edu.explain("inverse_dynamics", ExpertiseLevel.BEGINNER)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.shared.python.ai.types import ExpertiseLevel

logger = logging.getLogger(__name__)


@dataclass
class GlossaryEntry:
    """A glossary entry with multi-level explanations.

    Attributes:
        term: The term being defined.
        category: Topic category (e.g., "dynamics", "kinematics").
        definitions: Explanations by expertise level.
        related_terms: Related glossary terms.
        formula: Mathematical formula (if applicable).
        units: Physical units (if applicable).
        see_also: References or further reading.
    """

    term: str
    category: str
    definitions: dict[ExpertiseLevel, str] = field(default_factory=dict)
    related_terms: list[str] = field(default_factory=list)
    formula: str | None = None
    units: str | None = None
    see_also: list[str] = field(default_factory=list)

    def get_definition(self, level: ExpertiseLevel) -> str:
        """Get definition at or below the given level.

        Args:
            level: Maximum expertise level.

        Returns:
            Definition string.
        """
        # Try exact level
        if level in self.definitions:
            return self.definitions[level]

        # Fall back to lower levels
        for level_check in reversed(list(ExpertiseLevel)):
            if level_check <= level and level_check in self.definitions:
                return self.definitions[level_check]

        return f"Definition for '{self.term}' not available."


def _build_default_glossary() -> dict[str, GlossaryEntry]:
    """Build the default glossary with core biomechanics terms.

    Returns:
        Dictionary mapping term names to GlossaryEntry objects.
    """
    glossary: dict[str, GlossaryEntry] = {}

    # === DYNAMICS ===

    glossary["inverse_dynamics"] = GlossaryEntry(
        term="Inverse Dynamics",
        category="dynamics",
        definitions={
            ExpertiseLevel.BEGINNER: (
                "A way to figure out what forces caused a movement. "
                "By watching how someone moved, we can calculate the "
                "muscle forces they must have used. Think of it like "
                "being a detective - seeing the result and working backwards "
                "to find the cause."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "A computational method that calculates joint torques from "
                "measured kinematics. Given positions, velocities, and accelerations, "
                "inverse dynamics solves for the net torques at each joint."
            ),
            ExpertiseLevel.ADVANCED: (
                "The solution of τ = M(q)q̈ + C(q,q̇)q̇ + g(q) for joint torques τ, "
                "given measured generalized coordinates q and their derivatives. "
                "Requires accurate segment inertial properties."
            ),
            ExpertiseLevel.EXPERT: (
                "Recursive Newton-Euler formulation: outward kinematics sweep "
                "followed by inward dynamics sweep. O(n) complexity for n bodies. "
                "Sensitive to noise amplification from numerical differentiation."
            ),
        },
        formula="τ = M(q)q̈ + C(q,q̇) + g(q)",
        units="N·m",
        related_terms=["forward_dynamics", "joint_torque", "equations_of_motion"],
    )

    glossary["forward_dynamics"] = GlossaryEntry(
        term="Forward Dynamics",
        category="dynamics",
        definitions={
            ExpertiseLevel.BEGINNER: (
                "Predicting how something will move when you apply forces. "
                "Like pushing a swing - you know how hard you pushed, "
                "and forward dynamics tells you how the swing will move."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "Calculating motion from applied forces. Given joint torques, "
                "forward dynamics computes the resulting accelerations, which "
                "can be integrated to get velocities and positions."
            ),
            ExpertiseLevel.ADVANCED: (
                "Solving q̈ = M(q)⁻¹ [τ - C(q,q̇)q̇ - g(q)] for accelerations. "
                "Requires mass matrix inversion, typically using Cholesky decomposition."
            ),
        },
        formula="q̈ = M(q)⁻¹ [τ - C(q,q̇) - g(q)]",
        units="rad/s²",
        related_terms=["inverse_dynamics", "simulation", "equations_of_motion"],
    )

    glossary["joint_torque"] = GlossaryEntry(
        term="Joint Torque",
        category="dynamics",
        definitions={
            ExpertiseLevel.BEGINNER: (
                "The rotational force at a joint, like your elbow or knee. "
                "When you curl a weight, your bicep creates torque at your elbow. "
                "More torque means more rotational power."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "The net rotational force acting about a joint axis. "
                "Represents the sum of all muscle, ligament, and contact forces "
                "causing rotation at that joint."
            ),
            ExpertiseLevel.ADVANCED: (
                "Generalized force corresponding to a rotational degree of freedom. "
                "In inverse dynamics, it's the torque required to produce "
                "the observed angular acceleration given the inertial properties."
            ),
        },
        units="N·m (Newton-meters)",
        related_terms=["inverse_dynamics", "muscle_force", "moment_of_inertia"],
    )

    # === KINEMATICS ===

    glossary["kinematics"] = GlossaryEntry(
        term="Kinematics",
        category="kinematics",
        definitions={
            ExpertiseLevel.BEGINNER: (
                "The study of motion without worrying about forces. "
                "It's about describing HOW things move - positions, speeds, "
                "and how fast things speed up or slow down."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "The branch of mechanics describing motion through degrees of "
                "freedom (joints) without considering forces. Includes position, "
                "velocity, and acceleration analysis."
            ),
            ExpertiseLevel.ADVANCED: (
                "Configuration space analysis: generalized coordinates q(t), "
                "generalized velocities q̇(t), and accelerations q̈(t). "
                "Foundation for dynamics computations."
            ),
        },
        related_terms=["dynamics", "motion_capture", "joint_angles"],
    )

    glossary["c3d_file"] = GlossaryEntry(
        term="C3D File",
        category="data",
        definitions={
            ExpertiseLevel.BEGINNER: (
                "A file format for storing motion capture data. "
                "Contains the 3D positions of markers placed on a person's body "
                "during movement, recorded many times per second."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "The standard binary format for 3D biomechanics data. "
                "Stores marker trajectories, analog data (like force plates), "
                "and metadata about the recording."
            ),
            ExpertiseLevel.ADVANCED: (
                "Binary format per Motion Lab Systems specification. "
                "Contains header (512 bytes), parameter section, and data section. "
                "Supports up to 65535 frames and multiple point/analog channels."
            ),
        },
        related_terms=["motion_capture", "marker", "force_plate"],
    )

    glossary["marker"] = GlossaryEntry(
        term="Marker",
        category="data",
        definitions={
            ExpertiseLevel.BEGINNER: (
                "Small reflective balls placed on the body during motion capture. "
                "Special cameras track these markers to record exactly how "
                "the body moved in 3D space."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "Retro-reflective spheres (typically 10-25mm) placed at anatomical "
                "landmarks. Their 3D positions are tracked by infrared cameras "
                "to reconstruct body motion."
            ),
        },
        related_terms=["c3d_file", "motion_capture", "anatomical_landmark"],
    )

    # === GOLF-SPECIFIC ===

    glossary["kinetic_chain"] = GlossaryEntry(
        term="Kinetic Chain",
        category="golf",
        definitions={
            ExpertiseLevel.BEGINNER: (
                "How energy flows through your body during a golf swing. "
                "Power starts from your legs and ground contact, moves through "
                "your hips, torso, arms, and finally to the club head."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "The sequential activation and energy transfer from proximal "
                "to distal segments. In golf: lower body → trunk → arms → club. "
                "Proper timing maximizes club head speed."
            ),
            ExpertiseLevel.ADVANCED: (
                "Proximal-to-distal kinetic sequence enabling efficient "
                "angular momentum transfer. Peak angular velocities occur "
                "sequentially: pelvis → thorax → arm → club. "
                "Timing disruption reduces performance ~20%."
            ),
        },
        related_terms=["x_factor", "ground_reaction_force", "club_head_speed"],
    )

    glossary["x_factor"] = GlossaryEntry(
        term="X-Factor",
        category="golf",
        definitions={
            ExpertiseLevel.BEGINNER: (
                "The twist between your hips and shoulders at the top of the swing. "
                "A bigger X-Factor means you've stored more rotational energy, "
                "which can translate to more power."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "The difference between thorax and pelvis rotation angles "
                "at the top of backswing. Associated with increased club head speed, "
                "though individual optimal values vary."
            ),
            ExpertiseLevel.ADVANCED: (
                "Angular separation between thorax and pelvis in the transverse plane. "
                "Elite PGA: 45-55°. X-Factor Stretch (additional separation into "
                "downswing) correlates r=0.7 with ball speed."
            ),
        },
        units="degrees",
        related_terms=["kinetic_chain", "ground_reaction_force", "club_head_speed"],
    )

    glossary["ground_reaction_force"] = GlossaryEntry(
        term="Ground Reaction Force (GRF)",
        category="forces",
        definitions={
            ExpertiseLevel.BEGINNER: (
                "The push-back from the ground when you push against it. "
                "In golf, you push down and the ground pushes back up and "
                "sideways, helping you rotate powerfully."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "Forces measured by force plates during ground contact. "
                "Components: vertical (support weight + acceleration), "
                "anterior-posterior, and medio-lateral. Key for understanding "
                "how power originates from the ground."
            ),
            ExpertiseLevel.ADVANCED: (
                "Vector sum of distributed ground pressures. Analyzed as: "
                "vertical Fz (typically 1.2-1.5 BW at impact), "
                "shear forces Fx/Fy, and center of pressure trajectory. "
                "Moment about vertical axis indicates rotational demand."
            ),
        },
        units="N or BW (Body Weight)",
        related_terms=["kinetic_chain", "force_plate", "center_of_pressure"],
    )

    # === PHYSICS ENGINES ===

    glossary["physics_engine"] = GlossaryEntry(
        term="Physics Engine",
        category="simulation",
        definitions={
            ExpertiseLevel.BEGINNER: (
                "Software that simulates how things move and interact, "
                "like a virtual physics lab. The Golf Modeling Suite uses "
                "several engines to cross-check results for accuracy."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "Software library implementing rigid body dynamics. "
                "Handles contact, constraints, and integration. "
                "Examples: MuJoCo, Drake, Pinocchio."
            ),
            ExpertiseLevel.ADVANCED: (
                "Numerical solver for constrained multibody dynamics. "
                "Core algorithms: articulated body (Featherstone), "
                "projected Gauss-Seidel for contacts. Time-stepping via "
                "symplectic Euler or semi-implicit RK."
            ),
        },
        related_terms=["mujoco", "drake", "pinocchio", "simulation"],
    )

    glossary["mujoco"] = GlossaryEntry(
        term="MuJoCo",
        category="simulation",
        definitions={
            ExpertiseLevel.BEGINNER: (
                "One of the physics engines we use. It's especially good at "
                "handling contacts (like feet on ground or hands on club)."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "Multi-Joint dynamics with Contact. Physics engine optimized "
                "for robotics and biomechanics. Uses convex contact dynamics "
                "and implicit integration."
            ),
            ExpertiseLevel.ADVANCED: (
                "Recursive dynamics with complementarity contact model. "
                "PGS solver for contact forces. GPU-accelerated with MJX. "
                "Muscle models: first-order activation dynamics."
            ),
        },
        see_also=["https://mujoco.org"],
        related_terms=["physics_engine", "drake", "pinocchio"],
    )

    glossary["drake"] = GlossaryEntry(
        term="Drake",
        category="simulation",
        definitions={
            ExpertiseLevel.BEGINNER: (
                "Another physics engine we use. It's especially good at "
                "optimization and ensuring physical consistency."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "Model-Based Design for Robotics. MIT-developed toolkit for "
                "dynamics, control, and optimization. Uses symbolic computation "
                "for model analysis."
            ),
            ExpertiseLevel.ADVANCED: (
                "AutoDiff-enabled multibody dynamics. Direct collocation "
                "and SNOPT integration for trajectory optimization. "
                "Focuses on mathematical rigor and guarantees."
            ),
        },
        see_also=["https://drake.mit.edu"],
        related_terms=["physics_engine", "mujoco", "pinocchio"],
    )

    glossary["pinocchio"] = GlossaryEntry(
        term="Pinocchio",
        category="simulation",
        definitions={
            ExpertiseLevel.BEGINNER: (
                "A lightweight physics engine we use for fast calculations. "
                "Named after the famous puppet, it handles articulated bodies."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "Efficient rigid body dynamics library. Implements Featherstone's "
                "algorithms with analytical derivatives. Focus on speed and "
                "differentiability."
            ),
            ExpertiseLevel.ADVANCED: (
                "O(n) spatial algebra implementation. Supports CasADi and "
                "Ceres for automatic differentiation. Second-order derivatives "
                "for optimal control."
            ),
        },
        see_also=["https://github.com/stack-of-tasks/pinocchio"],
        related_terms=["physics_engine", "mujoco", "drake"],
    )

    # === VALIDATION ===

    glossary["cross_engine_validation"] = GlossaryEntry(
        term="Cross-Engine Validation",
        category="validation",
        definitions={
            ExpertiseLevel.BEGINNER: (
                "Checking our results are correct by running the same analysis "
                "on multiple physics engines. If they all agree, we can be "
                "confident the results are accurate."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "Verification method comparing results across MuJoCo, Drake, "
                "and Pinocchio. Agreement within tolerance indicates reliable "
                "results. Disagreement signals potential issues."
            ),
            ExpertiseLevel.ADVANCED: (
                "Ensemble validation using heterogeneous solvers. "
                "Tolerance thresholds: τ ± 2%, KE ± 0.5%, position ± 1e-6. "
                "Discrepancies often indicate contact model differences."
            ),
        },
        related_terms=["physics_engine", "tolerance", "validation"],
    )

    glossary["drift_control_decomposition"] = GlossaryEntry(
        term="Drift-Control Decomposition",
        category="analysis",
        definitions={
            ExpertiseLevel.BEGINNER: (
                "A way to understand which part of motion is 'passive' "
                "(would happen without muscles) versus 'controlled' "
                "(actively driven by muscles). Helps identify when "
                "muscles are really working versus just going along for the ride."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "Separating motion into: drift (gravity, inertia, passive dynamics) "
                "and control (active muscle contribution). The Drift-Control Ratio "
                "indicates how much active control a movement requires."
            ),
            ExpertiseLevel.ADVANCED: (
                "Affine decomposition: q̈ = drift(q, q̇) + B(q)u. "
                "Control contribution from null-space of drift solutions. "
                "DCR = ||control|| / (||drift|| + ||control||). "
                "Low DCR → gravity-assisted; High DCR → muscle-dominated."
            ),
        },
        formula="DCR = ||control|| / (||drift|| + ||control||)",
        related_terms=["inverse_dynamics", "muscle_contribution", "energy"],
    )

    return glossary


class EducationSystem:
    """Educational content delivery system.

    Provides multi-level explanations and glossary access
    based on user expertise level.

    Example:
        >>> edu = EducationSystem()
        >>> explanation = edu.explain("inverse_dynamics", ExpertiseLevel.BEGINNER)
        >>> related = edu.get_related_terms("inverse_dynamics")
    """

    def __init__(self) -> None:
        """Initialize education system with default glossary."""
        self._glossary = _build_default_glossary()
        logger.info("Initialized EducationSystem with %d terms", len(self._glossary))

    def explain(
        self,
        term: str,
        level: ExpertiseLevel = ExpertiseLevel.BEGINNER,
    ) -> str:
        """Get explanation for a term at the given level.

        Args:
            term: Term to explain (case-insensitive, underscores accepted).
            level: User's expertise level.

        Returns:
            Explanation string.
        """
        # Normalize term
        normalized = term.lower().replace(" ", "_").replace("-", "_")

        entry = self._glossary.get(normalized)
        if entry is None:
            return f"Term '{term}' not found in glossary."

        definition = entry.get_definition(level)

        # Add formula for advanced users
        if level >= ExpertiseLevel.ADVANCED and entry.formula:
            definition += f"\n\nFormula: {entry.formula}"

        # Add units for intermediate+
        if level >= ExpertiseLevel.INTERMEDIATE and entry.units:
            definition += f"\n\nUnits: {entry.units}"

        return definition

    def get_entry(self, term: str) -> GlossaryEntry | None:
        """Get full glossary entry.

        Args:
            term: Term to look up.

        Returns:
            GlossaryEntry if found, None otherwise.
        """
        normalized = term.lower().replace(" ", "_").replace("-", "_")
        return self._glossary.get(normalized)

    def get_related_terms(self, term: str) -> list[str]:
        """Get terms related to the given term.

        Args:
            term: Term to find relatives for.

        Returns:
            List of related term names.
        """
        entry = self.get_entry(term)
        if entry is None:
            return []
        return entry.related_terms

    def search(self, query: str) -> list[GlossaryEntry]:
        """Search glossary for matching terms.

        Args:
            query: Search query (partial match).

        Returns:
            List of matching GlossaryEntry objects.
        """
        query_lower = query.lower()
        results: list[GlossaryEntry] = []

        for entry in self._glossary.values():
            # Match term name
            if query_lower in entry.term.lower():
                results.append(entry)
                continue

            # Match category
            if query_lower in entry.category.lower():
                results.append(entry)
                continue

            # Match definitions
            for definition in entry.definitions.values():
                if query_lower in definition.lower():
                    results.append(entry)
                    break

        return results

    def list_categories(self) -> list[str]:
        """List all glossary categories.

        Returns:
            List of category names.
        """
        categories = {e.category for e in self._glossary.values()}
        return sorted(categories)

    def list_terms(self, category: str | None = None) -> list[str]:
        """List all terms, optionally filtered by category.

        Args:
            category: Filter by category (None for all).

        Returns:
            List of term names.
        """
        if category is None:
            return sorted(self._glossary.keys())

        return sorted(
            name for name, entry in self._glossary.items() if entry.category == category
        )

    def add_entry(self, entry: GlossaryEntry) -> None:
        """Add or update a glossary entry.

        Args:
            entry: Entry to add.
        """
        key = entry.term.lower().replace(" ", "_").replace("-", "_")
        self._glossary[key] = entry
        logger.debug("Added glossary entry: %s", entry.term)

    def __len__(self) -> int:
        """Return number of glossary entries."""
        return len(self._glossary)

    def __contains__(self, term: str) -> bool:
        """Check if term is in glossary."""
        normalized = term.lower().replace(" ", "_").replace("-", "_")
        return normalized in self._glossary
