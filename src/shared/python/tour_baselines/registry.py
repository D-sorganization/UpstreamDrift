"""Authoritative registry and alias resolution for golf models.

Enforces unambiguous model identity, separates topologies and backends,
and provides context-aware alias resolution to preserve saved sessions without
conflating distinct mechanisms.
"""

from __future__ import annotations

import logging
from typing import Optional

from .models import (
    BackendType,
    GolfModelIdentity,
    ModelTopology,
    SourceOwner,
)

logger = logging.getLogger(__name__)


class AmbiguousModelError(KeyError):
    """Raised when an alias maps to multiple distinct models and no context was supplied."""


_REGISTRY: dict[str, GolfModelIdentity] = {}
_ALIASES: dict[tuple[str, str | None], str] = {}
_REVERSE_ALIASES: dict[str, list[str]] = {}


def register_golf_model(identity: GolfModelIdentity) -> None:
    """Register a golf model identity and its aliases.

    Preconditions:
    - identity.model_id must not already be registered.
    """
    if identity.model_id in _REGISTRY:
        raise ValueError(f"Model ID '{identity.model_id}' is already registered")

    _REGISTRY[identity.model_id] = identity
    _REVERSE_ALIASES[identity.model_id] = list(identity.aliases)

    # Register canonical id
    _ALIASES[(identity.model_id, None)] = identity.model_id

    # Register aliases
    for alias in identity.aliases:
        _ALIASES[(alias, identity.topology.value)] = identity.model_id
        _ALIASES[(alias, identity.backend.value)] = identity.model_id

        # Also register specific convenience contexts
        if identity.topology == ModelTopology.KINEMATIC_RECONSTRUCTION:
            _ALIASES[(alias, "reconstruction")] = identity.model_id
            _ALIASES[(alias, "mocap")] = identity.model_id
        elif identity.topology == ModelTopology.PLANAR_DRIVEN_PENDULUM:
            _ALIASES[(alias, "pendulum_simulator")] = identity.model_id
            _ALIASES[(alias, "simulation")] = identity.model_id
            _ALIASES[(alias, "dynamic")] = identity.model_id
        elif identity.topology == ModelTopology.CONSTRAINED_UPPER_BODY:
            _ALIASES[(alias, "upper_body")] = identity.model_id
            _ALIASES[(alias, "pendulum_simulator")] = identity.model_id
        elif identity.topology == ModelTopology.FULL_BODY_MULTIBODY:
            _ALIASES[(alias, "full_body")] = identity.model_id
            _ALIASES[(alias, "matched_swing")] = identity.model_id


def get_golf_model(
    model_id_or_alias: str, context: str | None = None
) -> GolfModelIdentity:
    """Retrieve a model identity by ID or alias with optional disambiguating context.

    Raises:
        AmbiguousModelError: If the alias maps to multiple models and context was None.
        KeyError: If not found.
    """
    canonical_id = resolve_model_alias(model_id_or_alias, context=context)
    return _REGISTRY[canonical_id]


def list_golf_models() -> list[GolfModelIdentity]:
    """Return all registered model identities, sorted by canonical model_id."""
    return sorted(_REGISTRY.values(), key=lambda m: m.model_id)


def resolve_model_alias(alias: str, context: str | None = None) -> str:
    """Resolve an alias to a canonical model ID.

    Parameters:
        alias: Exact model ID or legacy alias (e.g. 'double_pendulum', 'double').
        context: Optional namespace / subsystem hint (e.g. 'reconstruction', 'pendulum_simulator').

    Returns:
        Canonical model_id string.
    """
    if not alias:
        raise ValueError("alias must be non-empty string")

    # 1. Direct canonical hit
    if alias in _REGISTRY:
        return alias

    # 2. Context-specific hit
    if context is not None:
        if (alias, context) in _ALIASES:
            return _ALIASES[(alias, context)]

    # 3. Check for exact alias across all models
    matching = [
        m.model_id
        for m in _REGISTRY.values()
        if alias in m.aliases or m.model_id == alias
    ]

    if len(matching) == 1:
        return matching[0]

    if len(matching) > 1:
        raise AmbiguousModelError(
            f"Alias '{alias}' is ambiguous across multiple models: {matching}. "
            f"Specify context (e.g. context='reconstruction' or context='pendulum_simulator')."
        )

    raise KeyError(f"No model registered for alias '{alias}' (context={context})")


def detect_provider_mismatch(provider_name: str, model_id: str) -> bool:
    """Check if an engine/provider adapter is compatible with the specified model.

    Returns:
        True if there is a mismatch (incompatible), False if valid.
    """
    try:
        model = get_golf_model(model_id)
    except KeyError:
        return True

    prov = provider_name.lower().strip()

    # Pendulum / tools providers require driven pendulums or upper-body golfer
    if prov in ("pendulum", "tools", "pendulum_simulator"):
        return model.topology not in (
            ModelTopology.PLANAR_DRIVEN_PENDULUM,
            ModelTopology.CONSTRAINED_UPPER_BODY,
        )

    # Kinematic reconstruct provider
    if prov in ("reconstruct", "reconstruction"):
        return model.topology != ModelTopology.KINEMATIC_RECONSTRUCTION

    # Engine-specific matching
    engine_map = {
        "mujoco": BackendType.MUJOCO,
        "pinocchio": BackendType.PINOCCHIO,
        "drake": BackendType.DRAKE,
        "opensim": BackendType.OPENSIM,
        "simscape": BackendType.SIMSCAPE,
        "myosuite": BackendType.MYOSUITE,
    }

    if prov in engine_map:
        return model.backend != engine_map[prov]

    return False


def clear_golf_model_registry() -> None:
    """Clear all registered models. Primarily for testing."""
    _REGISTRY.clear()
    _ALIASES.clear()
    _REVERSE_ALIASES.clear()


def init_default_registry() -> None:
    """Initialize canonical model registrations."""
    clear_golf_model_registry()

    # 1. Kinematic Reconstruction Models (omitting simulated club)
    register_golf_model(
        GolfModelIdentity(
            model_id="reconstruction_golfer",
            display_name="Reconstruction Golfer",
            topology=ModelTopology.KINEMATIC_RECONSTRUCTION,
            backend=BackendType.RECONSTRUCT_SOLVER,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.motion_capture.reconstruct.model.golfer",
            source_file="src/motion_capture/reconstruct/model/golfer.py",
            dof=17,
            independent_dof=17,
            constraint_count=0,
            constraint_description="Unconstrained anatomical kinematic tree",
            has_simulated_club=False,
            club_representation="None (hands = mean of wrists)",
            aliases=("golfer",),
            notes="Articulated scapula golfer model from #9709/#9914; omits simulated club",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="reconstruction_double_pendulum",
            display_name="Reconstruction Double Pendulum",
            topology=ModelTopology.KINEMATIC_RECONSTRUCTION,
            backend=BackendType.RECONSTRUCT_SOLVER,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.motion_capture.reconstruct.model.registry",
            source_file="src/motion_capture/reconstruct/model/registry.py",
            dof=4,
            independent_dof=4,
            constraint_count=0,
            constraint_description="3D pivot rotation + 1 planar hinge to hands",
            has_simulated_club=False,
            club_representation="None (hands = mean of wrists)",
            aliases=("double_pendulum", "double-pendulum/1.0"),
            notes="Kinematic shoulder-to-hands fit from #9730/#9914; omits simulated club",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="reconstruction_triple_pendulum",
            display_name="Reconstruction Triple Pendulum",
            topology=ModelTopology.KINEMATIC_RECONSTRUCTION,
            backend=BackendType.RECONSTRUCT_SOLVER,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.motion_capture.reconstruct.model.registry",
            source_file="src/motion_capture/reconstruct/model/registry.py",
            dof=5,
            independent_dof=5,
            constraint_count=0,
            constraint_description="3D pivot rotation + 2 planar hinges (elbow, wrist) to hands",
            has_simulated_club=False,
            club_representation="None (hands = mean of wrists)",
            aliases=("triple_pendulum", "triple-pendulum/1.0"),
            notes="Kinematic shoulder-to-elbow-to-hands fit from #9730/#9914; omits simulated club",
        )
    )

    # 2. Torque-Driven Reduced Pendulum Models (with explicitly simulated club)
    register_golf_model(
        GolfModelIdentity(
            model_id="driven_double_pendulum",
            display_name="Driven Double Pendulum",
            topology=ModelTopology.PLANAR_DRIVEN_PENDULUM,
            backend=BackendType.SCIPY_ODE,
            source_owner=SourceOwner.TOOLS,
            import_path="src.shared.python.pendulum_simulator.physics",
            source_file="src/shared/python/pendulum_simulator/physics.py",
            dof=2,
            independent_dof=2,
            constraint_count=0,
            constraint_description="Planar 2-DOF mechanism in inclined plane (theta1 arm, theta2 wrist)",
            has_simulated_club=True,
            club_representation="Rigid composite shaft + clubhead point mass",
            aliases=(
                "double",
                "double_pendulum",
                "double_pendulum_golf",
                "double_pendulum_analytical",
            ),
            notes="2-DOF torque-driven model; owned by Tools double_pendulum_golf package",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="driven_triple_pendulum",
            display_name="Driven Triple Pendulum",
            topology=ModelTopology.PLANAR_DRIVEN_PENDULUM,
            backend=BackendType.SCIPY_ODE,
            source_owner=SourceOwner.TOOLS,
            import_path="src.shared.python.pendulum_simulator.physics_triple",
            source_file="src/shared/python/pendulum_simulator/physics_triple.py",
            dof=3,
            independent_dof=3,
            constraint_count=0,
            constraint_description="Planar 3-DOF mechanism (hub theta0, arm theta1, club theta2)",
            has_simulated_club=True,
            club_representation="Rigid shaft + clubhead point mass",
            aliases=("triple", "triple_pendulum", "triple_pendulum_golf"),
            notes="3-DOF torque-driven model with moving hub; owned by Tools repo",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="constrained_upper_body_golfer",
            display_name="Constrained Upper Body Golfer",
            topology=ModelTopology.CONSTRAINED_UPPER_BODY,
            backend=BackendType.SCIPY_ODE,
            source_owner=SourceOwner.TOOLS,
            import_path="src.shared.python.pendulum_simulator.physics_golfer",
            source_file="src/shared/python/pendulum_simulator/physics_golfer.py",
            dof=8,
            independent_dof=5,
            constraint_count=4,
            constraint_description="4 loop closure equations (rank 3 constraint Jacobian) yielding 5 independent DOFs",
            has_simulated_club=True,
            club_representation="Rigid shaft + head point mass gripped by both hands",
            aliases=("golfer", "golfer_upper_body", "upper_body_golfer", "golfer_sim"),
            notes="8 generalized coordinates with closed bilateral grip kinematic loop",
        )
    )

    # 3. Flagship Full-Body Models in All Six Engines
    register_golf_model(
        GolfModelIdentity(
            model_id="full_body_mujoco",
            display_name="MuJoCo Full-Body Golfer",
            topology=ModelTopology.FULL_BODY_MULTIBODY,
            backend=BackendType.MUJOCO,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf",
            source_file="src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/__main__.py",
            dof=38,
            independent_dof=35,
            constraint_count=3,
            constraint_description="Bilateral hand grip weld constraint + ground polygon contact",
            has_simulated_club=True,
            club_representation="Driver / 7-Iron CAD + inertia models (club_models.py)",
            aliases=("mujoco_golf", "mujoco_full_body"),
            notes="Primary kinematic and forward-dynamics reference (governed by #10363, #10378)",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="full_body_pinocchio",
            display_name="Pinocchio / Crocoddyl Full-Body Golfer",
            topology=ModelTopology.FULL_BODY_MULTIBODY,
            backend=BackendType.PINOCCHIO,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.shared.python.motion_matching.crocoddyl_polynomial",
            source_file="src/shared/python/motion_matching/crocoddyl_polynomial.py",
            dof=46,
            independent_dof=40,
            constraint_count=6,
            constraint_description="Finite weld Jacobian contact closure + support cone",
            has_simulated_club=True,
            club_representation="Rigid club body with inertia tensors",
            aliases=("pinocchio_golf", "crocoddyl_golf"),
            notes="Analytical rigid-body dynamics and optimal control (#10377, #10378)",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="full_body_drake",
            display_name="Drake Full-Body Golfer",
            topology=ModelTopology.FULL_BODY_MULTIBODY,
            backend=BackendType.DRAKE,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.engines.physics_engines.drake.python.src.drake_gui_app",
            source_file="src/engines/physics_engines/drake/python/src/drake_gui_app.py",
            dof=40,
            independent_dof=34,
            constraint_count=6,
            constraint_description="Hydroelastic contact + bilateral grip weld",
            has_simulated_club=True,
            club_representation="MultibodyPlant club element with visual geometry",
            aliases=("drake_golf",),
            notes="High-fidelity contact & trajectory optimization (#10375, #10378)",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="full_body_opensim",
            display_name="OpenSim Moco Full-Body Golfer",
            topology=ModelTopology.FULL_BODY_MULTIBODY,
            backend=BackendType.OPENSIM,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.engines.physics_engines.opensim.python.tour_matching",
            source_file="src/engines/physics_engines/opensim/python/tour_matching/__init__.py",
            dof=35,
            independent_dof=29,
            constraint_count=6,
            constraint_description="Coordinate couplers, muscle activations, and grip constraints",
            has_simulated_club=True,
            club_representation="OpenSim club body welded to hand bodies",
            aliases=("opensim_golf", "moco_golf"),
            notes="Musculoskeletal validation and muscle actuation (#10376, #10414)",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="full_body_simscape",
            display_name="Simscape 3D Golf Model",
            topology=ModelTopology.FULL_BODY_MULTIBODY,
            backend=BackendType.SIMSCAPE,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.engines.Simscape_Multibody_Models.3D_Golf_Model",
            source_file="src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/",
            dof=26,
            independent_dof=20,
            constraint_count=6,
            constraint_description="Simscape physical joints, hard stops, and grip coupling",
            has_simulated_club=True,
            club_representation="Simscape flexible/rigid shaft + clubhead block",
            aliases=("simscape_golf", "matlab_golf"),
            notes="Historical tour capture authority pinned to MATLAB R2025b (#9921)",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="full_body_myosuite",
            display_name="MyoSuite Neural Golf Model",
            topology=ModelTopology.FULL_BODY_MULTIBODY,
            backend=BackendType.MYOSUITE,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.engines.physics_engines.myosuite.python.gui",
            source_file="src/engines/physics_engines/myosuite/python/gui.py",
            dof=30,
            independent_dof=24,
            constraint_count=6,
            constraint_description="Neural muscle activation constraints (currently fail-closed)",
            has_simulated_club=True,
            club_representation="MyoSuite club asset (pending retarget under MS-51)",
            aliases=("myosuite_golf",),
            notes="Experimental neural control model; fail-closed per MS-50 (#9478)",
        )
    )

    # 4. Catalog URDF / MJCF Models (#9914)
    register_golf_model(
        GolfModelIdentity(
            model_id="reference_pinocchio_urdf",
            display_name="Pinocchio Golfer URDF",
            topology=ModelTopology.REFERENCE_CATALOG_URDF,
            backend=BackendType.PINOCCHIO,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.motion_capture.reference.fit_catalog",
            source_file="src/engines/physics_engines/pinocchio/models/generated/golfer.urdf",
            dof=46,
            independent_dof=46,
            constraint_count=0,
            constraint_description="Joint limits from URDF",
            has_simulated_club=True,
            club_representation="Rigid club attached to hands in URDF",
            aliases=("pinocchio_golfer",),
            notes="Catalog URDF variant from #9914",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="reference_pinocchio_urdf_ik",
            display_name="Pinocchio Golfer IK URDF",
            topology=ModelTopology.REFERENCE_CATALOG_URDF,
            backend=BackendType.PINOCCHIO,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.motion_capture.reference.fit_catalog",
            source_file="src/engines/physics_engines/pinocchio/models/generated/golfer_ik.urdf",
            dof=46,
            independent_dof=46,
            constraint_count=0,
            constraint_description="IK-optimized joint limits from URDF",
            has_simulated_club=True,
            club_representation="Rigid club attached to hands in URDF",
            aliases=("pinocchio_golfer_ik",),
            notes="Catalog IK URDF variant from #9914",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="reference_drake_urdf",
            display_name="Drake Golfer URDF",
            topology=ModelTopology.REFERENCE_CATALOG_URDF,
            backend=BackendType.DRAKE,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.motion_capture.reference.fit_catalog",
            source_file="src/engines/physics_engines/drake/models/generated/golfer.urdf",
            dof=40,
            independent_dof=40,
            constraint_count=0,
            constraint_description="Joint limits from URDF",
            has_simulated_club=True,
            club_representation="Rigid club attached to hands in URDF",
            aliases=("drake_golfer",),
            notes="Catalog Drake URDF variant from #9914",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="reference_simple_humanoid",
            display_name="Simple Humanoid URDF",
            topology=ModelTopology.REFERENCE_CATALOG_URDF,
            backend=BackendType.RECONSTRUCT_SOLVER,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.motion_capture.reference.fit_catalog",
            source_file="src/shared/python/model_generation/library/bundled/simple_humanoid/humanoid.urdf",
            dof=28,
            independent_dof=28,
            constraint_count=0,
            constraint_description="Generic humanoid kinematic tree",
            has_simulated_club=False,
            club_representation="None",
            aliases=("simple_humanoid",),
            notes="Bundled simple humanoid URDF from #9914",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="reference_human_subject",
            display_name="Human Subject with Meshes",
            topology=ModelTopology.REFERENCE_CATALOG_URDF,
            backend=BackendType.RECONSTRUCT_SOLVER,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.motion_capture.reference.fit_catalog",
            source_file="src/tools/model_explorer/bundled_assets/human_models/human_subject_with_meshes/model.urdf",
            dof=32,
            independent_dof=32,
            constraint_count=0,
            constraint_description="Human anatomical joint limits",
            has_simulated_club=False,
            club_representation="None",
            aliases=("human_subject",),
            notes="Bundled human subject model from #9914",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="reference_mujoco_humanoid",
            display_name="MuJoCo Humanoid MJCF",
            topology=ModelTopology.REFERENCE_CATALOG_URDF,
            backend=BackendType.MUJOCO,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.motion_capture.reference.fit_catalog",
            source_file="src/shared/python/model_generation/library/bundled/mujoco_humanoid/humanoid.xml",
            dof=21,
            independent_dof=21,
            constraint_count=0,
            constraint_description="Joint limits in MJCF",
            has_simulated_club=False,
            club_representation="None",
            aliases=("mujoco_humanoid",),
            notes="Bundled MJCF humanoid from #9914",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="myosuite_body",
            display_name="MyoSuite Body Placeholder",
            topology=ModelTopology.REFERENCE_CATALOG_URDF,
            backend=BackendType.MYOSUITE,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.motion_capture.reference.fit_catalog",
            source_file="vendor/myosuite/placeholder",
            dof=0,
            independent_dof=0,
            constraint_count=0,
            constraint_description="Unavailable",
            has_simulated_club=False,
            club_representation="None",
            aliases=("myobody",),
            notes="Bundled myobody/myoupperbody are labeled placeholder models, not MyoSuite anatomy",
        )
    )

    register_golf_model(
        GolfModelIdentity(
            model_id="opensim_golfer",
            display_name="OpenSim Golfer Placeholder",
            topology=ModelTopology.REFERENCE_CATALOG_URDF,
            backend=BackendType.OPENSIM,
            source_owner=SourceOwner.UPSTREAM_DRIFT,
            import_path="src.motion_capture.reference.fit_catalog",
            source_file="src/engines/physics_engines/opensim/models/golfer.osim",
            dof=0,
            independent_dof=0,
            constraint_count=0,
            constraint_description="Custom-joint/muscle constraints require OpenSim adapter",
            has_simulated_club=False,
            club_representation="None",
            aliases=(),
            notes="Native OpenSim constraints require an OpenSim adapter; no surrogate fit substituted",
        )
    )


# Auto-initialize default registry on import
init_default_registry()
