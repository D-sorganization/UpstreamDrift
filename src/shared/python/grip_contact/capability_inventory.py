"""Cross-model grip capability inventory loader and DbC contract (HRG-GS-M0, #13322).

Enforces:
- Schema validation against "grip-capability-inventory/v1".
- Immutable (frozen) dataclasses for inventory metadata, models, and features.
- Strict Design-by-Contract validation:
  - Tracking issues must match the format '#\\d+'.
  - Features marked available=False must carry a non-empty reason.
  - Features marked available=True must reference an evidence test file that exists.
  - Model asset paths must exist on disk.
  - Finger articulation modes must be one of 'none', 'rigid_proxy', 'articulated'.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

CURRENT_SCHEMA = "grip-capability-inventory/v1"
REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[4]
DEFAULT_INVENTORY_PATH = (
    REPO_ROOT_DEFAULT / "src" / "config" / "grip_capability_inventory.json"
)

REQUIRED_FEATURES = frozenset(
    {"visual_style", "allocated_hand_hand", "physical_pad_grip", "physical_hand_hand"}
)
VALID_FINGER_MODES = frozenset({"none", "rigid_proxy", "articulated"})
TRACKING_ISSUE_PATTERN = re.compile(r"^#\d+$")


@dataclass(frozen=True)
class GripFeatureCapability:
    """Capability status for a specific grip feature on a model asset."""

    available: bool
    reason: str
    tracking_issue: str
    evidence_test: str | None = None

    def __post_init__(self) -> None:
        """Validate feature capability contracts."""
        if not isinstance(self.available, bool):
            raise TypeError("GripFeatureCapability.available must be a bool")
        if not isinstance(self.reason, str):
            raise TypeError("GripFeatureCapability.reason must be a str")
        if not isinstance(self.tracking_issue, str) or not TRACKING_ISSUE_PATTERN.match(
            self.tracking_issue
        ):
            raise ValueError(
                f"GripFeatureCapability.tracking_issue must match '^#\\d+$', got {self.tracking_issue!r}"
            )
        if not self.available:
            if not self.reason.strip():
                raise ValueError("Unavailable feature must provide a non-empty reason")
        else:
            if self.evidence_test is None or not self.evidence_test.strip():
                raise ValueError(
                    "Available feature must reference an evidence_test path"
                )


@dataclass(frozen=True)
class ModelGripCapability:
    """Grip capability and attachment metadata for a single model asset."""

    id: str
    engine: str
    asset_path: str
    attachment_mode: str
    fingers: str
    features: Mapping[str, GripFeatureCapability]
    launcher_tile_ids: tuple[str, ...] = ()
    description: str = ""
    repo_root: Path = field(default=REPO_ROOT_DEFAULT, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate model capability contracts and asset existence."""
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("ModelGripCapability.id must be a non-empty str")
        if not isinstance(self.engine, str) or not self.engine.strip():
            raise ValueError("ModelGripCapability.engine must be a non-empty str")
        if not isinstance(self.asset_path, str) or not self.asset_path.strip():
            raise ValueError("ModelGripCapability.asset_path must be a non-empty str")
        if (
            not isinstance(self.attachment_mode, str)
            or not self.attachment_mode.strip()
        ):
            raise ValueError(
                "ModelGripCapability.attachment_mode must be a non-empty str"
            )
        if self.fingers not in VALID_FINGER_MODES:
            raise ValueError(
                f"ModelGripCapability.fingers must be one of {sorted(VALID_FINGER_MODES)}, got {self.fingers!r}"
            )
        if not isinstance(self.features, Mapping):
            raise TypeError("ModelGripCapability.features must be a Mapping")
        missing_features = REQUIRED_FEATURES - set(self.features.keys())
        if missing_features:
            raise ValueError(
                f"ModelGripCapability.features missing required feature keys: {sorted(missing_features)}"
            )

        resolved_asset = (self.repo_root / self.asset_path).resolve()
        if not resolved_asset.is_file():
            raise FileNotFoundError(
                f"ModelGripCapability asset_path does not exist on disk: {self.asset_path}"
            )

        for fname, feat in self.features.items():
            if feat.available and feat.evidence_test is not None:
                resolved_test = (self.repo_root / feat.evidence_test).resolve()
                if not resolved_test.is_file():
                    raise FileNotFoundError(
                        f"Evidence test file does not exist on disk: {feat.evidence_test} (feature {fname} on {self.id})"
                    )


@dataclass(frozen=True)
class ExcludedModel:
    """An excluded model asset with justification."""

    id: str
    reason: str
    name: str = ""

    def __post_init__(self) -> None:
        """Validate exclusion contracts."""
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("ExcludedModel.id must be a non-empty str")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("ExcludedModel.reason must be a non-empty reason string")


@dataclass(frozen=True)
class GripCapabilityInventory:
    """Root inventory of grip capabilities across physics engines and model assets."""

    schema: str
    description: str
    models: Mapping[str, ModelGripCapability]
    excluded: Mapping[str, ExcludedModel]

    def __post_init__(self) -> None:
        """Validate inventory-level schema."""
        if self.schema != CURRENT_SCHEMA:
            raise ValueError(
                f"Unsupported grip capability inventory schema: {self.schema!r} (expected {CURRENT_SCHEMA!r})"
            )

    def get_model(self, model_id: str) -> ModelGripCapability | None:
        """Look up a model by its canonical package id."""
        return self.models.get(model_id)

    def covers_model(self, model_or_tile_id: str) -> bool:
        """Check if a model package ID or launcher tile ID is covered by an active model."""
        if model_or_tile_id in self.models:
            return True
        for m in self.models.values():
            if model_or_tile_id in m.launcher_tile_ids:
                return True
        return False

    def is_excluded(self, model_id: str) -> bool:
        """Check if a model or tile id is explicitly excluded."""
        return model_id in self.excluded

    def get_excluded(self, model_id: str) -> ExcludedModel | None:
        """Look up an excluded model entry."""
        return self.excluded.get(model_id)


def load_grip_capability_inventory(
    path: Path | str | None = None,
    *,
    repo_root: Path | None = None,
) -> GripCapabilityInventory:
    """Load and validate the cross-model grip capability inventory.

    Args:
        path: Path to the inventory JSON file. Defaults to src/config/grip_capability_inventory.json.
        repo_root: Repository root path for resolving relative asset and test paths.

    Returns:
        A validated, frozen GripCapabilityInventory instance.

    Raises:
        ValueError: If JSON structure, schema version, or DbC constraints fail.
        FileNotFoundError: If the inventory file or referenced assets do not exist.
    """
    resolved_root = (repo_root or REPO_ROOT_DEFAULT).resolve()
    target_path = (Path(path) if path is not None else DEFAULT_INVENTORY_PATH).resolve()

    if not target_path.is_file():
        raise FileNotFoundError(f"Grip capability inventory not found at {target_path}")

    with target_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Inventory root must be a JSON object")
    schema = data.get("schema", "")
    if schema != CURRENT_SCHEMA:
        raise ValueError(
            f"Invalid inventory schema: {schema!r} (expected {CURRENT_SCHEMA!r})"
        )

    models_dict: dict[str, ModelGripCapability] = {}
    for entry in data.get("models", []):
        if not isinstance(entry, dict):
            raise ValueError("Model entry must be a JSON object")
        mid = entry["id"]

        raw_features = entry.get("features", {})
        if not isinstance(raw_features, dict):
            raise ValueError(f"Features on {mid} must be a dict")

        features: dict[str, GripFeatureCapability] = {}
        for fname, fdict in raw_features.items():
            if not isinstance(fdict, dict):
                raise ValueError(f"Feature {fname} on {mid} must be a dict")
            features[fname] = GripFeatureCapability(
                available=bool(fdict.get("available", False)),
                reason=str(fdict.get("reason", "")),
                tracking_issue=str(fdict.get("tracking_issue", "")),
                evidence_test=fdict.get("evidence_test"),
            )

        tile_ids = tuple(entry.get("launcher_tile_ids", []))
        models_dict[mid] = ModelGripCapability(
            id=mid,
            engine=entry["engine"],
            asset_path=entry["asset_path"],
            attachment_mode=entry["attachment_mode"],
            fingers=entry["fingers"],
            features=features,
            launcher_tile_ids=tile_ids,
            description=entry.get("description", ""),
            repo_root=resolved_root,
        )

    excluded_dict: dict[str, ExcludedModel] = {}
    for entry in data.get("excluded", []):
        if not isinstance(entry, dict):
            raise ValueError("Excluded entry must be a JSON object")
        eid = entry["id"]
        excluded_dict[eid] = ExcludedModel(
            id=eid,
            reason=entry.get("reason", ""),
            name=entry.get("name", ""),
        )

    return GripCapabilityInventory(
        schema=schema,
        description=str(data.get("description", "")),
        models=models_dict,
        excluded=excluded_dict,
    )


PHYSICS_TILE_TYPES = frozenset(
    {
        "physics_engine",
        "custom_humanoid",
        "mjcf",
        "drake",
        "pinocchio",
        "opensim",
        "matlab_app",
    }
)


def _read_yaml_models(yaml_path: Path) -> list[dict[str, Any]]:
    import yaml

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return []
    models = data.get("models", [])
    return models if isinstance(models, list) else []


def enumerate_advertised_hand_bearing_assets(
    repo_root: Path | None = None,
) -> tuple[str, ...]:
    """Enumerate all advertised hand-bearing model assets and launcher tiles.

    Reads the M0 engine model inventory (`src/config/engine_model_inventory.json`),
    the launcher manifest (`src/config/models.yaml`), and the bundled model
    library (`src/shared/python/model_generation/library/bundled/manifest.json`),
    filtering out assets explicitly listed in `excluded` of
    `grip_capability_inventory.json`.
    """
    resolved_root = (repo_root or REPO_ROOT_DEFAULT).resolve()
    inventory = load_grip_capability_inventory(repo_root=resolved_root)
    assets: set[str] = set(inventory.models.keys())

    config_dir = resolved_root / "src" / "config"
    engine_inv_path = config_dir / "engine_model_inventory.json"
    if engine_inv_path.is_file():
        raw_inv = json.loads(engine_inv_path.read_text(encoding="utf-8"))
        for pkg in raw_inv.get("packages", []):
            pid = str(pkg.get("id", ""))
            if pid and not inventory.is_excluded(pid):
                assets.add(pid)

    models_yaml_path = config_dir / "models.yaml"
    if models_yaml_path.is_file():
        for m in _read_yaml_models(models_yaml_path):
            mid = str(m.get("id", ""))
            mtype = str(m.get("type", ""))
            launcher_cfg = m.get("launcher", {})
            cat = (
                str(launcher_cfg.get("category", ""))
                if isinstance(launcher_cfg, dict)
                else ""
            )
            is_model_tile = (
                cat == "physics_engine"
                or mtype in PHYSICS_TILE_TYPES
                or mid == "character_builder"
            )
            if mid and is_model_tile and not inventory.is_excluded(mid):
                assets.add(mid)

    bundled_path = (
        resolved_root
        / "src/shared/python/model_generation/library/bundled/manifest.json"
    )
    if bundled_path.is_file():
        raw_bundled = json.loads(bundled_path.read_text(encoding="utf-8"))
        for bmodel in raw_bundled.get("models", []):
            bid = str(bmodel.get("id", ""))
            if bid and not inventory.is_excluded(bid):
                assets.add(bid)

    return tuple(sorted(assets))


def _load_valid_binding_ids(repo_root: Path) -> set[str]:
    from src.shared.python.grip_contact.binding_manifest import validate_manifest

    bindings_dir = repo_root / "src" / "config" / "grip_bindings"
    bound_ids: set[str] = set()
    if not bindings_dir.is_dir():
        return bound_ids
    for mf_path in sorted(bindings_dir.glob("*.json")):
        manifest = json.loads(mf_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        bound_ids.add(str(manifest["model_id"]))
        bound_ids.add(mf_path.stem)
    return bound_ids


def _resolve_capability_entry(
    inventory: GripCapabilityInventory,
    asset_id: str,
) -> ModelGripCapability | None:
    direct = inventory.get_model(asset_id)
    if direct is not None:
        return direct
    for model in inventory.models.values():
        if asset_id in model.launcher_tile_ids:
            return model
    return None


def verify_model_library_admission(
    inventory: GripCapabilityInventory | None = None,
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Verify 100% admission coverage of advertised hand-bearing model assets.

    Every advertised asset with hands must either have a validated
    `grip-binding/v1` manifest (`"bound"`) or be reason-coded in
    `grip_capability_inventory.json` (`"reason_coded"`).
    """
    resolved_root = (repo_root or REPO_ROOT_DEFAULT).resolve()
    inv = inventory or load_grip_capability_inventory(repo_root=resolved_root)
    bound_ids = _load_valid_binding_ids(resolved_root)
    advertised = enumerate_advertised_hand_bearing_assets(repo_root=resolved_root)

    status: dict[str, str] = {}
    for asset_id in advertised:
        cap = _resolve_capability_entry(inv, asset_id)
        if cap is None:
            raise ValueError(
                f"Unadmitted hand-bearing model asset {asset_id!r}: neither bound nor reason-coded in inventory"
            )
        has_direct_manifest = (
            asset_id in bound_ids
            or cap.id in bound_ids
            or (
                "full_body_spec_anthro" in cap.asset_path
                and "full_body_v1_spec_export" in bound_ids
            )
        )
        if has_direct_manifest:
            status[asset_id] = "bound"
            continue

        # Must be reason-coded for all unavailable features
        for fname, feat in cap.features.items():
            if not feat.available and not feat.reason.strip():
                raise ValueError(
                    f"Asset {asset_id!r} feature {fname!r} is unavailable without a reason"
                )
        status[asset_id] = "reason_coded"

    return status
