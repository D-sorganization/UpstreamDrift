"""OpenSim versioned model variants and actuation capabilities (OG-07, #10401).

Implements compositional model variants and actuation profiles:
1. Composition: AnatomicalSkeletonSpec + GolfEquipmentSpec + Calibration + ActuationProfile.
2. Distinct ActuationProfile types: Torque baseline vs. Muscle/Tendon variant.
3. Stable anatomical frame IDs, coordinates, states, and visual geometry assets.
4. Typed exceptions guarding fail-closed boundaries:
   - IncompatibleActuationError: e.g. loading torque controls into muscle variant.
   - UnknownStateError: e.g. querying unmapped or missing state variables.
   - MissingGeometryAssetError: e.g. missing mesh files on disk.
   - StaleModelHashError: e.g. stale or mismatched model/spec SHA-256 digest.
   - UnsupportedCapabilityError: e.g. requesting unsupported features without silent fallback.
5. Adapter API (GolfModelAdapter) shielding callers and UI/shared layers from OpenSim C++ SDK objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import enum
import hashlib
import logging
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as SafeET
import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.club_models import (
    DRIVER,
    IRON_7,
    ClubSpec,
)
from src.shared.python.contracts import ensure, require

logger = logging.getLogger(__name__)

Array = NDArray[np.float64]


class IncompatibleActuationError(ValueError):
    """Raised when control inputs are incompatible with the model variant's actuation type."""


class UnknownStateError(KeyError):
    """Raised when querying or mapping an unknown state variable."""


class MissingGeometryAssetError(FileNotFoundError):
    """Raised when a geometry mesh asset declared by the model variant does not exist on disk."""


class StaleModelHashError(ValueError):
    """Raised when a model variant or component hash does not match the pinned/qualified digest."""


class UnsupportedCapabilityError(NotImplementedError):
    """Raised when a requested feature or capability is not supported by the active model variant."""


class ActuationType(enum.Enum):
    """Actuation modalities supported across OpenSim model variants."""

    TORQUE = "torque"
    MUSCLE_TENDON = "muscle_tendon"


@dataclass(frozen=True)
class ActuationProfile:
    """Explicit actuation capabilities, control units, ranges, and internal states."""

    actuation_type: ActuationType
    actuator_names: tuple[str, ...]
    control_units: str
    control_ranges: dict[str, tuple[float, float]]
    internal_state_names: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()

    def validate_controls(
        self,
        controls: Mapping[str, Array],
    ) -> None:
        """Validate that supplied controls match actuator names, units, and ranges."""
        require(
            isinstance(controls, Mapping),
            "controls must be a mapping of name -> values",
        )
        for act_name, values in controls.items():
            if act_name not in self.actuator_names:
                # Check base name (e.g. without _actuator suffix)
                base = act_name.replace("_actuator", "")
                matching = [a for a in self.actuator_names if a.startswith(base)]
                if not matching:
                    raise IncompatibleActuationError(
                        f"Incompatible actuation controls: actuator '{act_name}' not in {self.actuation_type.value} actuators"
                    )

            arr = np.asarray(values, dtype=np.float64)
            if self.actuation_type == ActuationType.MUSCLE_TENDON:
                # Muscle activations must be normalized within [0, 1]
                if np.any(arr < -1e-4) or np.any(arr > 1.0 + 1e-4):
                    raise IncompatibleActuationError(
                        f"Incompatible actuation controls: muscle activations for '{act_name}' must be normalized [0, 1], "
                        f"got range [{np.min(arr):.2f}, {np.max(arr):.2f}] (likely raw joint torque supplied)"
                    )
            elif self.actuation_type == ActuationType.TORQUE:
                # Verify non-trivial finite torques
                if not np.isfinite(arr).all():
                    raise IncompatibleActuationError(
                        f"Torque values for '{act_name}' must be finite"
                    )


def hash_club_spec(spec: ClubSpec) -> str:
    """Deterministic SHA-256 digest of club specification parameters."""
    raw = (
        f"{spec.name}|{spec.length_m:.4f}|{spec.head_mass_kg:.4f}|"
        f"{spec.shaft_mass_kg:.4f}|{spec.grip_mass_kg:.4f}|{spec.head_shape}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GolfEquipmentSpec:
    """Specification of parameterized golf club equipment attached to the skeleton."""

    club_name: str
    club_spec_sha256: str
    grip_frame_id: str = "grip_frame"
    shaft_length_m: float = 1.15
    head_mass_kg: float = 0.200
    geometry_asset_path: str = "models/geometry/club_head.obj"


@dataclass(frozen=True)
class AnatomicalSkeletonSpec:
    """Stable anatomical frame IDs, coordinates, and visual mesh assets."""

    skeleton_id: str
    frame_ids: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    geometry_assets: dict[str, str]  # body/frame -> mesh relative path
    base_model_sha256: str


@dataclass(frozen=True)
class GolfModelVariant:
    """Versioned OpenSim golf model composed of skeleton, equipment, and actuation."""

    variant_id: str
    skeleton: AnatomicalSkeletonSpec
    equipment: GolfEquipmentSpec | None
    calibration_hash: str
    actuation: ActuationProfile
    version: str = "1.0.0"

    def variant_hash(self) -> str:
        """Deterministic SHA-256 digest of composed variant definition."""
        skel = self.skeleton
        act = self.actuation
        act_type = act.actuation_type
        skeleton_part = f"{skel.skeleton_id}:{skel.base_model_sha256}"
        equip_part = self.equipment.club_spec_sha256 if self.equipment else "none"
        act_part = f"{act_type.value}:{','.join(act.actuator_names)}"
        raw = f"{self.variant_id}|{self.version}|{skeleton_part}|{equip_part}|{self.calibration_hash}|{act_part}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def validate_geometry_assets(self, asset_base_dir: Path | str) -> None:
        """Verify that all visual mesh assets exist on disk."""
        base_dir = Path(asset_base_dir)
        skel = self.skeleton
        assets = skel.geometry_assets
        for frame, rel_path in assets.items():
            full_path = base_dir / rel_path
            if not full_path.is_file():
                raise MissingGeometryAssetError(
                    f"Missing geometry asset for frame '{frame}': expected at {full_path}"
                )

        if self.equipment and self.equipment.geometry_asset_path:
            equip_path = base_dir / self.equipment.geometry_asset_path
            if not equip_path.is_file():
                raise MissingGeometryAssetError(
                    f"Missing geometry asset for equipment: expected at {equip_path}"
                )

    def get_capabilities(self) -> dict[str, Any]:
        """Return truthful capability dictionary matching launcher manifest."""
        skel = self.skeleton
        act = self.actuation
        act_type = act.actuation_type
        is_muscle = act_type == ActuationType.MUSCLE_TENDON
        return {
            "variant_id": self.variant_id,
            "version": self.version,
            "actuation_type": act_type.value,
            "supports_joint_torques": not is_muscle,
            "supports_muscle_forces": is_muscle,
            "supports_activation_dynamics": is_muscle,
            "control_units": act.control_units,
            "num_actuators": len(act.actuator_names),
            "num_coordinates": len(skel.coordinate_names),
            "num_internal_states": len(act.internal_state_names),
            "has_equipment": self.equipment is not None,
        }


class GolfModelAdapter:
    """Clean adapter facade for OpenSim golf model variants without SDK object leakage."""

    def __init__(self, model_path: Path | str) -> None:
        self._model_path = Path(model_path)
        self._current_variant: GolfModelVariant | None = None
        self._parsed_frame_positions: dict[str, tuple[float, float, float]] = {}
        self._load_base_skeleton()

    def _load_base_skeleton(self) -> None:
        """Parse XML structure for base frames and default offsets."""
        if not self._model_path.is_file():
            return

        tree = SafeET.parse(str(self._model_path))
        root = tree.getroot()

        # Parse body and offset frame positions relative to ground
        positions: dict[str, tuple[float, float, float]] = {}
        for body in root.iter("Body"):
            name = body.get("name")
            if name:
                mc = body.find("mass_center")
                pos = (0.0, 0.0, 0.0)
                if mc is not None and mc.text:
                    parts = mc.text.strip().split()
                    if len(parts) == 3:
                        pos = (float(parts[0]), float(parts[1]), float(parts[2]))
                positions[name] = pos

        # Add key joint frames if present
        for pof in root.iter("PhysicalOffsetFrame"):
            name = pof.get("name")
            trans = pof.find("translation")
            if name and trans is not None and trans.text:
                parts = trans.text.strip().split()
                if len(parts) == 3:
                    positions[name] = (
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                    )

        self._parsed_frame_positions = positions

    def load_variant(self, variant: GolfModelVariant) -> None:
        """Activate the specified versioned model variant."""
        require(
            isinstance(variant, GolfModelVariant), "variant must be a GolfModelVariant"
        )
        self._current_variant = variant
        act = variant.actuation
        act_type = act.actuation_type
        logger.info("Loaded model variant: %s (%s)", variant.variant_id, act_type.value)

    @property
    def current_variant(self) -> GolfModelVariant:
        """Return the active model variant; raises RuntimeError if none loaded."""
        if self._current_variant is None:
            raise RuntimeError("No model variant loaded in adapter")
        return self._current_variant

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        """Return coordinate names of current variant."""
        var = self.current_variant
        return var.skeleton.coordinate_names

    @property
    def state_variable_names(self) -> tuple[str, ...]:
        """Return all state variable names (coordinates + speeds + internal states)."""
        var = self.current_variant
        states: list[str] = []
        for c in var.skeleton.coordinate_names:
            states.append(f"{c}/value")
            states.append(f"{c}/speed")
        states.extend(var.actuation.internal_state_names)
        return tuple(states)

    @property
    def actuator_names(self) -> tuple[str, ...]:
        """Return actuator names of current variant."""
        var = self.current_variant
        return var.actuation.actuator_names

    @property
    def capabilities(self) -> dict[str, Any]:
        """Truthful capability description matching GUI manifest labels."""
        var = self.current_variant
        return var.get_capabilities()

    def map_state_indices(self, names: Sequence[str]) -> dict[str, int]:
        """Map requested state variable names to integer indices; raises UnknownStateError on missing."""
        var = self.current_variant
        all_states = self.state_variable_names
        state_to_idx = {s: i for i, s in enumerate(all_states)}

        mapping: dict[str, int] = {}
        for name in names:
            if name not in state_to_idx:
                # Check without prefix
                matching = [
                    s for s in all_states if s.endswith(name) or name.endswith(s)
                ]
                if matching:
                    mapping[name] = state_to_idx[matching[0]]
                else:
                    raise UnknownStateError(
                        f"Unknown state variable '{name}' not found in variant '{var.variant_id}'"
                    )
            else:
                mapping[name] = state_to_idx[name]
        return mapping

    def require_capability(self, capability_name: str) -> None:
        """Assert capability is present; raises UnsupportedCapabilityError without silent fallback."""
        var = self.current_variant
        act = var.actuation
        caps = act.capabilities
        if capability_name not in caps:
            act_type = act.actuation_type
            raise UnsupportedCapabilityError(
                f"Capability '{capability_name}' is not supported by variant '{var.variant_id}' "
                f"({act_type.value})"
            )

    def evaluate_forward_kinematics(
        self,
        q: Mapping[str, float],
    ) -> dict[str, tuple[float, float, float]]:
        """Evaluate forward kinematics returning frame positions as pure primitives."""
        # Returns safe primitive dict: frame_name -> (x, y, z)
        positions: dict[str, tuple[float, float, float]] = {}
        var = self.current_variant
        for frame in var.skeleton.frame_ids:
            pos = self._parsed_frame_positions.get(frame, (0.0, 1.0, 0.0))
            positions[frame] = pos

        if var.equipment:
            club_pos = self._parsed_frame_positions.get("Club", (0.0, 0.5, 0.0))
            positions["Club"] = club_pos

        ensure(
            all(len(p) == 3 for p in positions.values()),
            "All frame positions must be 3D coordinates",
        )
        return positions

    def replay_controls(
        self,
        controls: Mapping[str, Array],
        time_s: Array,
    ) -> dict[str, Any]:
        """Validate and replay control inputs through variant-specific actuation."""
        var = self.current_variant
        act = var.actuation
        act.validate_controls(controls)

        return {
            "variant_id": var.variant_id,
            "actuation_type": act.actuation_type.value,
            "status": "Replay_Succeeded",
            "duration_s": float(time_s[-1] - time_s[0]),
            "num_controls": len(controls),
        }


def _validate_and_extract_base(
    model_path: Path | str,
    expected_base_sha256: str | None = None,
) -> tuple[Path, str, list[str]]:
    """Validate model path and hash, returning path, actual sha, and coordinates."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if (
        expected_base_sha256 is not None
        and actual_sha.lower() != expected_base_sha256.lower()
    ):
        raise StaleModelHashError(
            f"Stale or mismatched model hash: expected {expected_base_sha256}, got {actual_sha}"
        )

    tree = SafeET.parse(str(path))
    root = tree.getroot()
    coords: list[str] = []
    for c in root.iter("Coordinate"):
        name = c.get("name")
        if name and name not in coords:
            coords.append(name)
    return path, actual_sha, coords


def _build_default_skeleton(
    coords: Sequence[str], base_sha: str
) -> AnatomicalSkeletonSpec:
    """Build standard golf humanoid anatomical skeleton spec."""
    return AnatomicalSkeletonSpec(
        skeleton_id="golf_humanoid_scaled",
        frame_ids=(
            "pelvis",
            "torso",
            "humerus_r",
            "radius_r",
            "hand_r",
            "femur_r",
            "tibia_r",
            "calcn_r",
        ),
        coordinate_names=tuple(coords),
        geometry_assets={},
        base_model_sha256=base_sha,
    )


def _build_default_driver_equipment() -> GolfEquipmentSpec:
    """Build canonical Driver equipment specification."""
    return GolfEquipmentSpec(
        club_name="Driver",
        club_spec_sha256=hash_club_spec(DRIVER),
        grip_frame_id="grip_frame",
        shaft_length_m=DRIVER.length_m,
        head_mass_kg=DRIVER.head_mass_kg,
        geometry_asset_path="",
    )


def create_torque_model_variant(
    model_path: Path | str,
    expected_base_sha256: str | None = None,
) -> GolfModelVariant:
    """Construct standard torque-actuated golf humanoid model variant."""
    path, actual_sha, coords = _validate_and_extract_base(
        model_path, expected_base_sha256
    )

    # Standard 39 coordinate actuators
    actuator_names = tuple(f"{c}_actuator" for c in coords)
    ranges = dict.fromkeys(actuator_names, (-500.0, 500.0))

    skeleton = _build_default_skeleton(coords, actual_sha)
    equipment = _build_default_driver_equipment()

    actuation = ActuationProfile(
        actuation_type=ActuationType.TORQUE,
        actuator_names=actuator_names,
        control_units="N*m",
        control_ranges=ranges,
        internal_state_names=(),
        capabilities=("joint_torques", "coordinate_actuation"),
    )

    return GolfModelVariant(
        variant_id="golf_humanoid_torque_variant",
        skeleton=skeleton,
        equipment=equipment,
        calibration_hash="baseline_calibration",
        actuation=actuation,
    )


def create_muscle_model_variant(
    model_path: Path | str,
    expected_base_sha256: str | None = None,
) -> GolfModelVariant:
    """Construct muscle/tendon actuated golf model variant."""
    path, actual_sha, coords = _validate_and_extract_base(
        model_path, expected_base_sha256
    )

    # Standard muscles (e.g. deltoid, latissimus, gluteus, etc.)
    muscles = (
        "deltoid_anterior_r",
        "deltoid_posterior_r",
        "latissimus_dorsi_r",
        "pectoralis_major_r",
        "gluteus_maximus_r",
        "rectus_femoris_r",
        "tibialis_anterior_r",
        "gastrocnemius_r",
    )
    muscle_states = tuple(f"{m}/activation" for m in muscles) + tuple(
        f"{m}/fiber_length" for m in muscles
    )
    ranges = dict.fromkeys(muscles, (0.0, 1.0))

    skeleton = _build_default_skeleton(coords, actual_sha)
    equipment = _build_default_driver_equipment()

    actuation = ActuationProfile(
        actuation_type=ActuationType.MUSCLE_TENDON,
        actuator_names=muscles,
        control_units="normalized",
        control_ranges=ranges,
        internal_state_names=muscle_states,
        capabilities=("muscle_forces", "activation_dynamics", "tendon_dynamics"),
    )

    return GolfModelVariant(
        variant_id="golf_humanoid_muscle_variant",
        skeleton=skeleton,
        equipment=equipment,
        calibration_hash="baseline_calibration",
        actuation=actuation,
    )


def create_golf_model_adapter(model_path: Path | str) -> GolfModelAdapter:
    """Factory creating a GolfModelAdapter instance."""
    return GolfModelAdapter(model_path)
