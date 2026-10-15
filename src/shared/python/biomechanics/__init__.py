"""Muscle models, kinematics, and swing analysis.

This module provides a backward-compatible lazy facade over biomechanics kernels,
submodules, and optional visualization/integration tools. Importing the top-level
package or individual core kernels does not eagerly load unrelated optional
dependencies or visualization tools.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .activation_dynamics import ActivationDynamics
    from .biomechanics_data import BiomechanicalData
    from .club_solids_partition import (
        ClubSolidsPartition,
        SolidSpatialInertia,
        bare_club_kinematics_from_body_frame,
        club_solids_partition,
    )
    from .drift_control_transfer import (
        AttributionShares,
        ForcePathProjection,
        ImpulseDecomposition,
        JointTransferTrajectory,
        PathFrame,
        PathWeightedMeanForce,
        PhaseTransferSummary,
        PowerWorkDecomposition,
        SwingPhase,
        attribution_shares,
        build_phase_masks,
        compute_impulses,
        compute_path_frame,
        compute_path_weighted_mean_force,
        compute_power_and_work,
        project_forces_onto_path,
        summarize_phases,
    )
    from .dynamic_com import BiomechanicalModel, SegmentDefinition
    from .fsp_integration import (
        FspResult,
        compute_swing_fsp,
        detect_md_mf_window,
        extract_clubhead_trajectory,
    )
    from .grf_visualization import plot_grf_and_com_3d
    from .hill_muscle import HillMuscleModel, MuscleParameters, MuscleState
    from .humanoid_urdf_contracts import ContractViolation, ValidationReport
    from .interaction_evidence import (
        EvidenceManifest,
        InterfaceDescriptor,
        NumericalTolerance,
        PredictionRecord,
        SpatialWrenchTrajectory,
        calibrate_convergence_tolerance,
        load_evidence_manifest,
        spatial_from_planar,
    )
    from .kinematic_sequence import SegmentPeak, SegmentTimingResult
    from .multi_muscle import MuscleAttachment, MuscleGroup
    from .muscle_allocation import (
        MuscleAllocationResult,
        MuscleRecruitmentPath,
        allocate_muscle_excitations,
    )
    from .muscle_analysis import MuscleSynergyAnalyzer, SynergyResult
    from .muscle_equilibrium import EquilibriumSolver
    from .muscle_force_evaluation import (
        QuasistaticMuscleForceEvaluation,
        evaluate_quasistatic_muscle_force,
    )
    from .myoconverter_integration import MyoConverter
    from .myosuite_adapter import MuscleDrivenEnv
    from .rust_muscle import f_l, is_rust_available
    from .swing_comparison import ComparisonMetric, DTWResult, SwingComparator
    from .swing_plane_analysis import SwingPlaneAnalyzer, SwingPlaneMetrics
    from .swing_plane_visualization import SwingPlaneVisualization
    from .ztcf import ZTCFResult

_UNCONDITIONAL_EXPORTS: dict[str, str] = {
    # fsp_integration
    "FspResult": ".fsp_integration",
    "compute_swing_fsp": ".fsp_integration",
    "detect_md_mf_window": ".fsp_integration",
    "extract_clubhead_trajectory": ".fsp_integration",
    # activation_dynamics
    "ActivationDynamics": ".activation_dynamics",
    # biomechanics_data
    "BiomechanicalData": ".biomechanics_data",
    # club_solids_partition
    "ClubSolidsPartition": ".club_solids_partition",
    "SolidSpatialInertia": ".club_solids_partition",
    "bare_club_kinematics_from_body_frame": ".club_solids_partition",
    "club_solids_partition": ".club_solids_partition",
    # drift_control_transfer
    "AttributionShares": ".drift_control_transfer",
    "ForcePathProjection": ".drift_control_transfer",
    "ImpulseDecomposition": ".drift_control_transfer",
    "JointTransferTrajectory": ".drift_control_transfer",
    "PathFrame": ".drift_control_transfer",
    "PathWeightedMeanForce": ".drift_control_transfer",
    "PhaseTransferSummary": ".drift_control_transfer",
    "PowerWorkDecomposition": ".drift_control_transfer",
    "SwingPhase": ".drift_control_transfer",
    "attribution_shares": ".drift_control_transfer",
    "build_phase_masks": ".drift_control_transfer",
    "compute_impulses": ".drift_control_transfer",
    "compute_path_frame": ".drift_control_transfer",
    "compute_path_weighted_mean_force": ".drift_control_transfer",
    "compute_power_and_work": ".drift_control_transfer",
    "project_forces_onto_path": ".drift_control_transfer",
    "summarize_phases": ".drift_control_transfer",
    # interaction_evidence
    "EvidenceManifest": ".interaction_evidence",
    "InterfaceDescriptor": ".interaction_evidence",
    "NumericalTolerance": ".interaction_evidence",
    "PredictionRecord": ".interaction_evidence",
    "SpatialWrenchTrajectory": ".interaction_evidence",
    "calibrate_convergence_tolerance": ".interaction_evidence",
    "load_evidence_manifest": ".interaction_evidence",
    "spatial_from_planar": ".interaction_evidence",
    # hill_muscle
    "HillMuscleModel": ".hill_muscle",
    "MuscleParameters": ".hill_muscle",
    "MuscleState": ".hill_muscle",
    # muscle_allocation
    "MuscleAllocationResult": ".muscle_allocation",
    "MuscleRecruitmentPath": ".muscle_allocation",
    "allocate_muscle_excitations": ".muscle_allocation",
    # kinematic_sequence
    "SegmentPeak": ".kinematic_sequence",
    "SegmentTimingResult": ".kinematic_sequence",
    # multi_muscle
    "MuscleAttachment": ".multi_muscle",
    "MuscleGroup": ".multi_muscle",
    # muscle_equilibrium
    "EquilibriumSolver": ".muscle_equilibrium",
    # muscle_force_evaluation
    "QuasistaticMuscleForceEvaluation": ".muscle_force_evaluation",
    "evaluate_quasistatic_muscle_force": ".muscle_force_evaluation",
    # swing_comparison
    "ComparisonMetric": ".swing_comparison",
    "DTWResult": ".swing_comparison",
    "SwingComparator": ".swing_comparison",
    # swing_plane_analysis
    "SwingPlaneAnalyzer": ".swing_plane_analysis",
    "SwingPlaneMetrics": ".swing_plane_analysis",
    # ztcf
    "ZTCFResult": ".ztcf",
}

_BASE_ALL: tuple[str, ...] = tuple(_UNCONDITIONAL_EXPORTS.keys())

_OPTIONAL_GROUPS: dict[str, tuple[str, ...]] = {
    ".dynamic_com": ("BiomechanicalModel", "SegmentDefinition"),
    ".humanoid_urdf_contracts": ("ContractViolation", "ValidationReport"),
    ".rust_muscle": ("f_l", "is_rust_available"),
    ".muscle_analysis": ("MuscleSynergyAnalyzer", "SynergyResult"),
    ".myoconverter_integration": ("MyoConverter",),
    ".myosuite_adapter": ("MuscleDrivenEnv",),
    ".swing_plane_visualization": ("SwingPlaneVisualization",),
    ".grf_visualization": ("plot_grf_and_com_3d",),
}

_OPTIONAL_EXPORTS_MAP: dict[str, str] = {
    sym: mod for mod, syms in _OPTIONAL_GROUPS.items() for sym in syms
}

_OPTIONAL_RESOLVED: set[str] = set()


def _resolve_optional_group(module_relpath: str) -> None:
    """Resolve an optional module group on demand, binding attributes to this module.

    If the module cannot be imported due to an ImportError, all symbols in the group
    are set to None in this module's namespace. Non-ImportError exceptions are not
    caught and propagate to the caller.
    """
    if module_relpath in _OPTIONAL_RESOLVED:
        return
    symbols = _OPTIONAL_GROUPS[module_relpath]
    mod = sys.modules[__name__]
    try:
        loaded = importlib.import_module(module_relpath, __package__)
    except ImportError:
        _OPTIONAL_RESOLVED.add(module_relpath)
        for sym in symbols:
            setattr(mod, sym, None)
        return
    _OPTIONAL_RESOLVED.add(module_relpath)
    for sym in symbols:
        setattr(mod, sym, getattr(loaded, sym))


def _resolve_all_optional_groups() -> None:
    """Resolve all optional groups on demand."""
    for mod_relpath in _OPTIONAL_GROUPS:
        _resolve_optional_group(mod_relpath)


def __getattr__(name: str) -> Any:
    """Resolve lazy facade exports and optional symbols on demand."""
    if name in _UNCONDITIONAL_EXPORTS:
        module = importlib.import_module(_UNCONDITIONAL_EXPORTS[name], __package__)
        value = getattr(module, name)
        setattr(sys.modules[__name__], name, value)
        return value
    if name in _OPTIONAL_EXPORTS_MAP:
        _resolve_optional_group(_OPTIONAL_EXPORTS_MAP[name])
        return getattr(sys.modules[__name__], name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all documented public names without triggering optional imports."""
    all_names = list(_BASE_ALL)
    for syms in _OPTIONAL_GROUPS.values():
        all_names.extend(syms)
    return all_names


class _BiomechanicsModule(types.ModuleType):
    """Custom module type supporting dynamic __all__ and redirecting child modules."""

    @property
    def __all__(self) -> list[str]:
        """Return the mutable legacy-ordered export list.

        Optional groups are resolved on demand when __all__ is accessed.
        Matches the legacy sequence: _BASE_ALL followed by optional groups
        whose sentinel is not None.
        """
        if "__all__" in self.__dict__:
            return self.__dict__["__all__"]
        _resolve_all_optional_groups()
        result: list[str] = list(_BASE_ALL)
        for syms in _OPTIONAL_GROUPS.values():
            sentinel = syms[0]
            if getattr(self, sentinel) is not None:
                result.extend(syms)
        self.__dict__["__all__"] = result
        return result

    @__all__.setter
    def __all__(self, value: list[str]) -> None:
        self.__dict__["__all__"] = value

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if (
            name in _UNCONDITIONAL_EXPORTS or name in _OPTIONAL_EXPORTS_MAP
        ) and isinstance(value, types.ModuleType):
            module_name = _UNCONDITIONAL_EXPORTS.get(
                name, _OPTIONAL_EXPORTS_MAP.get(name, "")
            )
            module = importlib.import_module(module_name, __package__)
            resolved = getattr(module, name)
            setattr(self, name, resolved)
            return resolved
        return value

    def __dir__(self) -> list[str]:
        return __dir__()


sys.modules[__name__].__class__ = _BiomechanicsModule
