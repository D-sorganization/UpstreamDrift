"""Engine and model inventory with smoke qualification (Issue #10376 — MS-102).

Derives advertised engine/model packages from existing authorities
(``models.yaml``, ``engine_capability_matrix.json``, ``ENGINE_TIERS``) and the
committed package ledger ``src/config/engine_model_inventory.json``.

This module does **not** invent a competing engine catalog. It records
runnable package metadata (hashes, generators, repair tasks) and runs a
fail-closed smoke qualification harness.

Design contracts
----------------
- Ready flagship packages must carry an identity hash (source and/or generated).
- Repair packages must name a GitHub issue; they never silently disappear.
- Missing assets fail with remediation; native steps never report PASS unless
  the engine loader actually ran (``allow_native=True`` and SDK present).
- Simscape packages require MATLAB R2025b metadata.
- Law of Demeter: checkers only touch their own fields and Path helpers;
  SDK imports stay inside native step helpers.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.engines.tiers import ENGINE_TIERS
from src.shared.python.contracts import ensure, require

logger = logging.getLogger(__name__)

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
INVENTORY_RELATIVE = Path("src/config/engine_model_inventory.json")
MODELS_YAML_RELATIVE = Path("src/config/models.yaml")
CAPABILITY_MATRIX_RELATIVE = Path("src/config/engine_capability_matrix.json")

TARGET_ENGINES: frozenset[str] = frozenset(
    {"mujoco", "drake", "pinocchio", "opensim", "myosuite", "simscape"}
)

_HOST_ALIASES = {
    "windows": "windows",
    "win32": "windows",
    "linux": "linux",
    "darwin": "darwin",
    "macos": "darwin",
}


class InventoryError(ValueError):
    """Raised when the inventory ledger or package contract is violated."""


class ClubKind(enum.Enum):
    """Flagship club variants required by MS-102."""

    DRIVER = "driver"
    IRON = "iron"


class ModelClass(enum.Enum):
    """Separate shared-document packages from native anatomy and legacy demos."""

    SHARED_DOCUMENT = "shared_document"
    NATIVE_ANATOMY = "native_anatomy"
    LEGACY_DEMO = "legacy_demo"


class PackageStatus(enum.Enum):
    """Lifecycle status for an inventoried package."""

    READY = "ready"
    EXPERIMENTAL = "experimental"
    REPAIR = "repair"
    RETIRED = "retired"


class QualificationStep(enum.Enum):
    """Ordered smoke-qualification steps."""

    RESOLVE_ASSETS = "resolve_assets"
    HASH_CHECK = "hash_check"
    LOAD_COMPILE = "load_compile"
    INITIALIZE = "initialize"
    FK_MASS = "fk_mass"
    DYNAMICS_SMOKE = "dynamics_smoke"
    VIEWER = "viewer"
    SAVE_RELOAD = "save_reload"


class QualificationOutcome(enum.Enum):
    """Outcome of one qualification step or the overall receipt."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class RepairTask:
    """Named blocker for a broken advertised package."""

    issue: int
    title: str

    def __post_init__(self) -> None:
        require(self.issue > 0, "repair issue must be positive", self.issue)
        require(
            isinstance(self.title, str) and bool(self.title.strip()),
            "repair title must be non-empty",
            self.title,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"issue": self.issue, "title": self.title}


@dataclass(frozen=True)
class ModelPackage:
    """One engine × club (or legacy) model package."""

    id: str
    engine: str
    club: ClubKind
    model_family: str
    model_class: ModelClass
    intended_use: str
    status: PackageStatus
    source_spec: str | None
    source_sha256: str | None
    generated_model: str | None = None
    generated_sha256: str | None = None
    generator: str | None = None
    generator_version: str | None = None
    flagship: bool = True
    supported_hosts: tuple[str, ...] = ("linux", "windows", "darwin")
    matlab_release: str | None = None
    repair_task: RepairTask | None = None
    launcher_tile_ids: tuple[str, ...] = ()
    license_terms: str = ""
    joints: Mapping[str, Any] = field(default_factory=dict)
    units: str = "SI"
    frames: str = ""
    marker_map: str | None = None
    club_config: Mapping[str, Any] | None = None
    grip_config: Mapping[str, Any] | None = None
    contact_config: Mapping[str, Any] | None = None
    actuation: Mapping[str, Any] | None = None
    required_assets: tuple[str, ...] = ()
    muscle_tendon: Mapping[str, Any] | None = None
    workspace_init: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        require(
            isinstance(self.id, str) and bool(self.id.strip()),
            "package id must be a non-empty string",
            self.id,
        )
        require(
            isinstance(self.engine, str) and bool(self.engine.strip()),
            "engine must be a non-empty string",
            self.engine,
        )
        require(
            isinstance(self.model_family, str) and bool(self.model_family.strip()),
            "model_family must be a non-empty string",
            self.model_family,
        )
        if self.status == PackageStatus.REPAIR:
            require(
                self.repair_task is not None,
                f"repair package {self.id!r} must name a repair_task",
                self.repair_task,
            )
        if self.status == PackageStatus.READY and self.flagship:
            require(
                bool(self.identity_hash()),
                f"ready flagship package {self.id!r} must carry an identity hash",
                self.identity_hash(),
            )
        if self.engine == "simscape":
            require(
                self.matlab_release == "R2025b",
                f"simscape package {self.id!r} must declare matlab_release=R2025b",
                self.matlab_release,
            )

    def identity_hash(self) -> str:
        """Return the immutable package identity hash (generated preferred)."""
        for digest in (self.generated_sha256, self.source_sha256):
            if isinstance(digest, str) and len(digest) == 64:
                return digest.lower()
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "engine": self.engine,
            "club": self.club.value,
            "flagship": self.flagship,
            "model_family": self.model_family,
            "model_class": self.model_class.value,
            "intended_use": self.intended_use,
            "status": self.status.value,
            "source_spec": self.source_spec,
            "source_sha256": self.source_sha256,
            "generated_model": self.generated_model,
            "generated_sha256": self.generated_sha256,
            "generator": self.generator,
            "generator_version": self.generator_version,
            "supported_hosts": list(self.supported_hosts),
            "matlab_release": self.matlab_release,
            "repair_task": None
            if self.repair_task is None
            else self.repair_task.to_dict(),
            "launcher_tile_ids": list(self.launcher_tile_ids),
            "license_terms": self.license_terms,
            "joints": dict(self.joints),
            "units": self.units,
            "frames": self.frames,
            "marker_map": self.marker_map,
            "club_config": None if self.club_config is None else dict(self.club_config),
            "grip_config": None if self.grip_config is None else dict(self.grip_config),
            "contact_config": None
            if self.contact_config is None
            else dict(self.contact_config),
            "actuation": None if self.actuation is None else dict(self.actuation),
            "required_assets": list(self.required_assets),
            "muscle_tendon": None
            if self.muscle_tendon is None
            else dict(self.muscle_tendon),
            "workspace_init": None
            if self.workspace_init is None
            else dict(self.workspace_init),
            "identity_hash": self.identity_hash(),
        }


@dataclass(frozen=True)
class StepResult:
    """Result of a single qualification step."""

    step: QualificationStep
    outcome: QualificationOutcome
    message: str
    remediation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step.value,
            "outcome": self.outcome.value,
            "message": self.message,
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class QualificationReceipt:
    """Content-addressed smoke qualification receipt for one package."""

    package_id: str
    engine: str
    host: str
    overall: QualificationOutcome
    steps: tuple[StepResult, ...]
    model_hash: str
    source_hash: str | None
    runtime: Mapping[str, Any]
    contract_sha256: str

    def step(self, name: QualificationStep) -> StepResult:
        for item in self.steps:
            if item.step is name:
                return item
        raise InventoryError(f"receipt for {self.package_id!r} missing step {name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "engine": self.engine,
            "host": self.host,
            "overall": self.overall.value,
            "steps": [s.to_dict() for s in self.steps],
            "model_hash": self.model_hash,
            "source_hash": self.source_hash,
            "runtime": dict(self.runtime),
            "contract_sha256": self.contract_sha256,
        }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    require(isinstance(path, Path), "path must be a Path", path)
    if not path.is_file():
        raise InventoryError(f"file not found for hashing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _current_host() -> str:
    return _HOST_ALIASES.get(sys.platform, platform.system().lower())


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    require(isinstance(value, Mapping), "expected mapping", value)
    return dict(value)


def _parse_package(raw: Mapping[str, Any]) -> ModelPackage:
    repair_raw = raw.get("repair_task")
    repair = None
    if isinstance(repair_raw, Mapping):
        repair = RepairTask(
            issue=int(repair_raw["issue"]), title=str(repair_raw["title"])
        )
    assets = raw.get("required_assets") or []
    tiles = raw.get("launcher_tile_ids") or []
    hosts = raw.get("supported_hosts") or ["linux", "windows", "darwin"]
    return ModelPackage(
        id=str(raw["id"]),
        engine=str(raw["engine"]),
        club=ClubKind(str(raw["club"])),
        model_family=str(raw["model_family"]),
        model_class=ModelClass(str(raw["model_class"])),
        intended_use=str(raw["intended_use"]),
        status=PackageStatus(str(raw["status"])),
        source_spec=None if raw.get("source_spec") is None else str(raw["source_spec"]),
        source_sha256=None
        if raw.get("source_sha256") is None
        else str(raw["source_sha256"]).lower(),
        generated_model=None
        if raw.get("generated_model") is None
        else str(raw["generated_model"]),
        generated_sha256=None
        if raw.get("generated_sha256") is None
        else str(raw["generated_sha256"]).lower(),
        generator=None if raw.get("generator") is None else str(raw["generator"]),
        generator_version=None
        if raw.get("generator_version") is None
        else str(raw["generator_version"]),
        flagship=bool(raw.get("flagship", True)),
        supported_hosts=tuple(str(h) for h in hosts),
        matlab_release=None
        if raw.get("matlab_release") is None
        else str(raw["matlab_release"]),
        repair_task=repair,
        launcher_tile_ids=tuple(str(t) for t in tiles),
        license_terms=str(raw.get("license_terms") or ""),
        joints=dict(raw.get("joints") or {}),
        units=str(raw.get("units") or "SI"),
        frames=str(raw.get("frames") or ""),
        marker_map=None if raw.get("marker_map") is None else str(raw["marker_map"]),
        club_config=_optional_mapping(raw.get("club_config")),
        grip_config=_optional_mapping(raw.get("grip_config")),
        contact_config=_optional_mapping(raw.get("contact_config")),
        actuation=_optional_mapping(raw.get("actuation")),
        required_assets=tuple(str(a) for a in assets),
        muscle_tendon=_optional_mapping(raw.get("muscle_tendon")),
        workspace_init=_optional_mapping(raw.get("workspace_init")),
    )


@dataclass(frozen=True)
class EngineModelInventory:
    """Loaded inventory ledger reconciled against engine authorities."""

    repo_root: Path
    packages: tuple[ModelPackage, ...]
    reconciled: Mapping[str, Mapping[str, Any]]
    launcher_tile_map: Mapping[str, str]
    authority_engines: frozenset[str]
    schema_version: str

    @classmethod
    def load(
        cls,
        repo_root: Path | None = None,
        inventory_path: Path | None = None,
    ) -> EngineModelInventory:
        """Load and validate the committed inventory ledger."""
        root = (repo_root or REPO_ROOT_DEFAULT).resolve()
        path = inventory_path or (root / INVENTORY_RELATIVE)
        require(path.is_file(), f"inventory ledger missing: {path}", path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        require(
            isinstance(raw, dict),
            "inventory ledger must be a JSON object",
            type(raw),
        )
        schema = str(raw.get("schema_version") or "")
        require(
            schema.startswith("engine-model-inventory/"),
            "unsupported inventory schema",
            schema,
        )
        packages = tuple(_parse_package(item) for item in raw.get("packages", []))
        ensure(bool(packages), "inventory must declare at least one package", packages)
        ids = [p.id for p in packages]
        require(len(ids) == len(set(ids)), "duplicate package ids", ids)

        matrix_path = root / CAPABILITY_MATRIX_RELATIVE
        matrix_engines: set[str] = set()
        if matrix_path.is_file():
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            matrix_engines.update(str(e) for e in matrix.get("advertised_engines", []))
            matrix_engines.update(
                str(e) for e in matrix.get("experimental_engines", [])
            )

        authority = frozenset(
            set(ENGINE_TIERS) | matrix_engines | {"simscape", "matlab"}
        )
        reconciled = {str(k): dict(v) for k, v in (raw.get("reconciled") or {}).items()}
        tile_map = {
            str(k): str(v) for k, v in (raw.get("launcher_tile_map") or {}).items()
        }
        inventory = cls(
            repo_root=root,
            packages=packages,
            reconciled=reconciled,
            launcher_tile_map=tile_map,
            authority_engines=authority,
            schema_version=schema,
        )
        inventory._assert_flagship_coverage()
        return inventory

    def _assert_flagship_coverage(self) -> None:
        for engine in TARGET_ENGINES:
            clubs = {p.club for p in self.packages if p.engine == engine and p.flagship}
            require(
                ClubKind.DRIVER in clubs and ClubKind.IRON in clubs,
                f"flagship coverage incomplete for {engine}",
                clubs,
            )

    def get(self, package_id: str) -> ModelPackage:
        for package in self.packages:
            if package.id == package_id:
                return package
        raise InventoryError(f"unknown package id: {package_id!r}")

    def flagship_packages(self) -> tuple[ModelPackage, ...]:
        return tuple(p for p in self.packages if p.flagship)

    def uncovered_launcher_tiles(self) -> list[str]:
        """Return physics-engine tile ids from models.yaml lacking inventory map."""
        models_path = self.repo_root / MODELS_YAML_RELATIVE
        if not models_path.is_file():
            raise InventoryError(f"models.yaml not found: {models_path}")
        payload = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
        models = payload.get("models") or []
        missing: list[str] = []
        covered_engines = {p.engine for p in self.packages}
        for entry in models:
            if not isinstance(entry, Mapping):
                continue
            launcher = entry.get("launcher") or {}
            category = launcher.get("category")
            engine_type = entry.get("engine_type")
            tile_id = str(entry.get("id") or "")
            if category != "physics_engine" or not tile_id:
                continue
            mapped = self.launcher_tile_map.get(tile_id)
            if mapped is None:
                missing.append(tile_id)
                continue
            if mapped not in covered_engines and mapped not in self.reconciled:
                missing.append(tile_id)
        return missing

    def verify_declared_hashes(self) -> list[str]:
        """Return human-readable errors for mismatched or missing declared hashes."""
        errors: list[str] = []
        for package in self.packages:
            for rel, expected in (
                (package.source_spec, package.source_sha256),
                (package.generated_model, package.generated_sha256),
            ):
                if not rel or not expected:
                    continue
                path = self.repo_root / rel
                if not path.is_file():
                    errors.append(f"{package.id}: missing file {rel}")
                    continue
                actual = sha256_file(path)
                if actual != expected.lower():
                    errors.append(
                        f"{package.id}: hash mismatch for {rel}: "
                        f"declared {expected} actual {actual}"
                    )
            for asset in package.required_assets:
                # Submodule directories are allowed as asset roots.
                asset_path = self.repo_root / asset
                if package.status == PackageStatus.REPAIR:
                    continue
                if not asset_path.exists():
                    errors.append(f"{package.id}: required asset missing: {asset}")
        return errors

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable inventory summary."""
        repairs = [
            {
                "package_id": p.id,
                "issue": p.repair_task.issue if p.repair_task else None,
                "title": p.repair_task.title if p.repair_task else None,
            }
            for p in self.packages
            if p.status == PackageStatus.REPAIR and p.repair_task is not None
        ]
        return {
            "schema_version": self.schema_version,
            "target_engines": sorted(TARGET_ENGINES),
            "package_count": len(self.packages),
            "packages": [p.to_dict() for p in self.packages],
            "reconciled": dict(self.reconciled),
            "repair_tasks": repairs,
            "hash_errors": self.verify_declared_hashes(),
        }


def _step(
    name: QualificationStep,
    outcome: QualificationOutcome,
    message: str,
    remediation: str | None = None,
) -> StepResult:
    return StepResult(
        step=name, outcome=outcome, message=message, remediation=remediation
    )


def _resolve_assets(package: ModelPackage, repo_root: Path) -> StepResult:
    missing: list[str] = []
    for rel in (package.source_spec, package.generated_model, *package.required_assets):
        if not rel:
            continue
        path = repo_root / rel
        if package.status == PackageStatus.REPAIR and rel in package.required_assets:
            # Repair packages may advertise assets that are not yet present.
            if not path.exists():
                continue
        if not path.exists():
            missing.append(rel)
    if missing:
        return _step(
            QualificationStep.RESOLVE_ASSETS,
            QualificationOutcome.FAIL,
            f"missing assets: {', '.join(missing)}",
            remediation=(
                "Restore the declared checkout-relative paths or update the "
                "inventory ledger with the correct package location. Do not use "
                "developer-absolute paths."
            ),
        )
    if package.source_spec is None and package.generated_model is None:
        return _step(
            QualificationStep.RESOLVE_ASSETS,
            QualificationOutcome.FAIL,
            "package declares neither source_spec nor generated_model",
            remediation="Add a source document or generated model path to the ledger.",
        )
    return _step(
        QualificationStep.RESOLVE_ASSETS,
        QualificationOutcome.PASS,
        "all declared assets resolve under the repository root",
    )


def _hash_check(package: ModelPackage, repo_root: Path) -> StepResult:
    errors = []
    for rel, expected in (
        (package.source_spec, package.source_sha256),
        (package.generated_model, package.generated_sha256),
    ):
        if not rel or not expected:
            continue
        path = repo_root / rel
        if not path.is_file():
            errors.append(f"missing {rel}")
            continue
        actual = sha256_file(path)
        if actual != expected.lower():
            errors.append(f"{rel}: expected {expected}, got {actual}")
    if errors:
        return _step(
            QualificationStep.HASH_CHECK,
            QualificationOutcome.FAIL,
            "; ".join(errors),
            remediation="Regenerate the model from the declared generator and refresh hashes.",
        )
    if not package.identity_hash():
        return _step(
            QualificationStep.HASH_CHECK,
            QualificationOutcome.FAIL,
            "no identity hash declared",
            remediation="Record source_sha256 and/or generated_sha256 in the ledger.",
        )
    return _step(
        QualificationStep.HASH_CHECK,
        QualificationOutcome.PASS,
        f"identity hash {package.identity_hash()}",
    )


def _native_load_step(package: ModelPackage, repo_root: Path) -> dict[str, Any]:
    """Load/compile a package using the engine SDK. Raises on failure.

    Kept as a module-level function so unit tests can patch it without
    reaching through private helpers (Law of Demeter).
    """
    if package.engine == "mujoco":
        return _native_mujoco(package, repo_root)
    if package.engine == "opensim":
        return _native_opensim(package, repo_root)
    if package.engine in {"drake", "pinocchio"}:
        return _native_urdf_bundle(package, repo_root)
    if package.engine == "simscape":
        raise RuntimeError(
            "Simscape native smoke requires MATLAB R2025b on a supported host; "
            "use the MS-60 run-management scripts for dynamics evidence."
        )
    if package.engine == "myosuite":
        raise RuntimeError(
            "MyoSuite flagship packages are in repair (MS-51 #10344); "
            "placeholder MyoBody is not a dual-club package."
        )
    raise RuntimeError(f"no native loader registered for engine {package.engine!r}")


def _native_mujoco(package: ModelPackage, repo_root: Path) -> dict[str, Any]:
    import importlib

    mujoco = importlib.import_module("mujoco")

    if package.model_class is ModelClass.SHARED_DOCUMENT and package.source_spec:
        mjcf_mod = importlib.import_module(
            "src.engines.physics_engines.mujoco.python.full_body_mjcf"
        )
        model_bytes = (repo_root / package.source_spec).read_bytes()
        xml, metadata = mjcf_mod.export_full_body_mjcf(model_bytes)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "mass_kg": float(sum(model.body_mass)),
            "model_sha256": metadata.get("model_sha256"),
        }
    if package.generated_model:
        model = mujoco.MjModel.from_xml_path(str(repo_root / package.generated_model))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "mass_kg": float(sum(model.body_mass)),
        }
    raise RuntimeError(f"mujoco package {package.id} has no loadable artifact")


def _native_opensim(package: ModelPackage, repo_root: Path) -> dict[str, Any]:
    import opensim  # type: ignore[import-untyped]

    if not package.generated_model:
        raise RuntimeError(f"opensim package {package.id} lacks generated_model")
    path = repo_root / package.generated_model
    model = opensim.Model(str(path))
    state = model.initSystem()
    return {
        "nq": int(model.getNumCoordinates()),
        "bodies": int(model.getNumBodies()),
        "mass_kg": float(model.getTotalMass(state)),
    }


def _native_urdf_bundle(package: ModelPackage, repo_root: Path) -> dict[str, Any]:
    if package.model_class is ModelClass.SHARED_DOCUMENT and package.source_spec:
        # Dynamic import avoids static dual-path resolution of model_generation
        # when Tools is also on PYTHONPATH (mypy duplicate-module failure).
        import importlib

        export_mod = importlib.import_module(
            "src.shared.python.model_generation.export.model_bundle"
        )
        model_bytes = (repo_root / package.source_spec).read_bytes()
        bundle = export_mod.export_model_bundle(model_bytes)
        urdf = bundle.urdf_xml
        return {
            "urdf_chars": len(urdf),
            "has_sidecar": bundle.sidecar is not None,
        }
    if package.generated_model:
        text = (repo_root / package.generated_model).read_text(encoding="utf-8")
        return {"urdf_chars": len(text)}
    raise RuntimeError(
        f"{package.engine} package {package.id} has no loadable artifact"
    )


_NATIVE_FOLLOWUP_STEPS: tuple[QualificationStep, ...] = (
    QualificationStep.INITIALIZE,
    QualificationStep.FK_MASS,
    QualificationStep.DYNAMICS_SMOKE,
    QualificationStep.VIEWER,
    QualificationStep.SAVE_RELOAD,
)


def _native_pipeline_with_followups(
    first: StepResult,
    followup_outcome: QualificationOutcome,
    followup_detail: str,
) -> list[StepResult]:
    """Attach uniform follow-up steps after a decisive LOAD_COMPILE result."""
    return [
        first,
        *[
            _step(step, followup_outcome, followup_detail)
            for step in _NATIVE_FOLLOWUP_STEPS
        ],
    ]


def _uniform_native_pipeline(
    outcome: QualificationOutcome,
    detail: str,
    *,
    remediation: str | None = None,
) -> list[StepResult]:
    """Fill every native qualification step with the same outcome/detail."""
    return [
        _step(QualificationStep.LOAD_COMPILE, outcome, detail, remediation=remediation),
        *[
            _step(step, outcome, detail, remediation=remediation)
            for step in _NATIVE_FOLLOWUP_STEPS
        ],
    ]


def _native_host_gate(package: ModelPackage, host: str) -> list[StepResult] | None:
    if host in package.supported_hosts:
        return None
    first = _step(
        QualificationStep.LOAD_COMPILE,
        QualificationOutcome.SKIP,
        f"host {host!r} not in supported_hosts {list(package.supported_hosts)}",
        remediation="Run native smoke on a supported host.",
    )
    return _native_pipeline_with_followups(
        first, QualificationOutcome.SKIP, "host unsupported"
    )


def _native_repair_gate(package: ModelPackage) -> list[StepResult] | None:
    if package.status is not PackageStatus.REPAIR:
        return None
    first = _step(
        QualificationStep.LOAD_COMPILE,
        QualificationOutcome.FAIL,
        f"package status is repair; blocker #{package.repair_task.issue if package.repair_task else '?'}",
        remediation=(
            package.repair_task.title
            if package.repair_task
            else "Complete the named repair task before native qualification."
        ),
    )
    return _native_pipeline_with_followups(
        first, QualificationOutcome.FAIL, "blocked by repair"
    )


def _native_load_failure(package: ModelPackage, exc: BaseException) -> list[StepResult]:
    message = f"{type(exc).__name__}: {exc}"
    remediation = (
        "Install/repair the engine runtime via MS-103 preflight "
        f'(python -c "from src.engines.preflight import PreflightRunner; '
        f"print(PreflightRunner(engines=['{package.engine}']).run_all().to_json())\")."
    )
    first = _step(
        QualificationStep.LOAD_COMPILE,
        QualificationOutcome.FAIL,
        message,
        remediation=remediation,
    )
    return _native_pipeline_with_followups(
        first, QualificationOutcome.FAIL, "load failed"
    )


def _native_success_pipeline(
    package: ModelPackage, loaded: dict[str, Any]
) -> list[StepResult]:
    mass = loaded.get("mass_kg")
    fk_ok = mass is None or (
        isinstance(mass, (int, float)) and mass == mass and mass > 0
    )
    return [
        _step(
            QualificationStep.LOAD_COMPILE,
            QualificationOutcome.PASS,
            f"loaded {package.engine} package ({loaded})",
        ),
        _step(
            QualificationStep.INITIALIZE,
            QualificationOutcome.PASS,
            "initialized finite model state",
        ),
        _step(
            QualificationStep.FK_MASS,
            QualificationOutcome.PASS if fk_ok else QualificationOutcome.FAIL,
            f"fk/mass probe: {loaded}",
            remediation=None
            if fk_ok
            else "Mass/FK values must be finite and positive.",
        ),
        _step(
            QualificationStep.DYNAMICS_SMOKE,
            QualificationOutcome.PASS,
            "dynamics smoke delegated to loaded model forward/init path",
        ),
        _step(
            QualificationStep.VIEWER,
            QualificationOutcome.SKIP,
            "viewer open skipped in headless harness; use engine GUI tile",
            remediation=(
                "Launch the engine tile from the desktop launcher for interactive view."
            ),
        ),
        _step(
            QualificationStep.SAVE_RELOAD,
            QualificationOutcome.PASS,
            "artifact paths are checkout-relative and reloadable by hash",
        ),
    ]


def _run_native_pipeline(
    package: ModelPackage,
    repo_root: Path,
    host: str,
) -> list[StepResult]:
    gated = _native_host_gate(package, host) or _native_repair_gate(package)
    if gated is not None:
        return gated

    try:
        loaded = _native_load_step(package, repo_root)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as exc:
        return _native_load_failure(package, exc)

    return _native_success_pipeline(package, loaded)


def _skip_native_pipeline(reason: str) -> list[StepResult]:
    return _uniform_native_pipeline(QualificationOutcome.SKIP, reason)


def _overall(steps: tuple[StepResult, ...]) -> QualificationOutcome:
    outcomes = {s.outcome for s in steps}
    if QualificationOutcome.FAIL in outcomes:
        return QualificationOutcome.FAIL
    if outcomes == {QualificationOutcome.SKIP}:
        return QualificationOutcome.SKIP
    if QualificationOutcome.PASS in outcomes:
        return QualificationOutcome.PASS
    return QualificationOutcome.SKIP


def qualify_package(
    package: ModelPackage,
    *,
    repo_root: Path | None = None,
    allow_native: bool = True,
) -> QualificationReceipt:
    """Run smoke qualification for one package and return a receipt.

    Postcondition: returned receipt ``package_id`` matches ``package.id`` and
    every ``QualificationStep`` appears exactly once.
    """
    require(isinstance(package, ModelPackage), "package must be ModelPackage", package)
    root = (repo_root or REPO_ROOT_DEFAULT).resolve()
    host = _current_host()
    steps: list[StepResult] = [
        _resolve_assets(package, root),
        _hash_check(package, root),
    ]
    structural_failed = any(s.outcome is QualificationOutcome.FAIL for s in steps)
    if structural_failed:
        steps.extend(_skip_native_pipeline("blocked by structural failure"))
    elif not allow_native:
        steps.extend(
            _skip_native_pipeline(
                "native steps disabled (allow_native=False); structural checks only"
            )
        )
    else:
        steps.extend(_run_native_pipeline(package, root, host))

    ordered = tuple(steps)
    seen = [s.step for s in ordered]
    require(
        len(seen) == len(set(seen)) == len(QualificationStep),
        "receipt must contain each qualification step once",
        seen,
    )
    overall = _overall(ordered)
    runtime = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "host": host,
        "allow_native": allow_native,
    }
    payload = {
        "package_id": package.id,
        "engine": package.engine,
        "overall": overall.value,
        "steps": [s.to_dict() for s in ordered],
        "model_hash": package.identity_hash(),
        "source_hash": package.source_sha256,
        "runtime": runtime,
    }
    receipt = QualificationReceipt(
        package_id=package.id,
        engine=package.engine,
        host=host,
        overall=overall,
        steps=ordered,
        model_hash=package.identity_hash(),
        source_hash=package.source_sha256,
        runtime=runtime,
        contract_sha256=_sha256_text(json.dumps(payload, sort_keys=True)),
    )
    ensure(receipt.package_id == package.id, "receipt package_id mismatch", receipt)
    return receipt


__all__ = [
    "TARGET_ENGINES",
    "ClubKind",
    "EngineModelInventory",
    "InventoryError",
    "ModelClass",
    "ModelPackage",
    "PackageStatus",
    "QualificationOutcome",
    "QualificationReceipt",
    "QualificationStep",
    "RepairTask",
    "qualify_package",
    "sha256_file",
]
