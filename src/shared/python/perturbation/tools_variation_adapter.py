"""Fail-closed gateway to the canonical Tools variation contracts.

UpstreamDrift owns engine execution and trace adaptation, not variation-plan
sampling, persistence, dispersion, or sensitivity mathematics. This module
validates the optional public Tools boundary and returns Tools records without
wrapping or relabeling them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from os import PathLike, fspath

ROOT_MODULE = "shared.python.swing_sim.variation"
EXECUTION_METADATA_MODULE = f"{ROOT_MODULE}.execution_metadata"
EXECUTION_PROVENANCE_MODULE = f"{ROOT_MODULE}.execution_provenance"
PERSISTED_PLAN_MODULE = f"{ROOT_MODULE}.persisted_plan_io"
VARIATION_MODULES = (
    ROOT_MODULE,
    EXECUTION_METADATA_MODULE,
    EXECUTION_PROVENANCE_MODULE,
    PERSISTED_PLAN_MODULE,
)

EXPECTED_PLAN_SCHEMA_VERSION = 2
EXPECTED_EXECUTION_DOCUMENT_SCHEMA_VERSION = 3
EXPECTED_PROVENANCE_SCHEMA_VERSION = 1
EXPECTED_PLAN_BINDING_SCHEMA_VERSION = 1
EXPECTED_DATASET_JSON_SCHEMA_VERSION = 2
EXPECTED_DATASET_HDF5_SCHEMA_ID = "rate-of-closure/variation-dataset-hdf5"
EXPECTED_DATASET_HDF5_SCHEMA_VERSION = 1

Importer = Callable[[str], object]
_MISSING = object()


class ToolsVariationUnavailableError(ImportError):
    """Raised when the configured Tools variation modules are unavailable."""


class ToolsVariationCompatibilityError(ValueError):
    """Raised when available Tools modules violate the supported contract."""


@dataclass(frozen=True)
class ToolsVariationCapabilities:
    """Exact schema evidence obtained while probing the optional boundary."""

    available: bool
    module_name: str
    plan_schema_version: int | None
    execution_document_schema_version: int | None
    provenance_schema_version: int | None
    plan_binding_schema_version: int | None
    dataset_json_schema_version: int | None
    dataset_hdf5_schema_id: str | None
    dataset_hdf5_schema_version: int | None
    failure: str | None = None

    def __post_init__(self) -> None:
        if self.module_name != ROOT_MODULE:
            raise ValueError("module_name must identify the canonical Tools façade")
        identities = (
            self.plan_schema_version,
            self.execution_document_schema_version,
            self.provenance_schema_version,
            self.plan_binding_schema_version,
            self.dataset_json_schema_version,
            self.dataset_hdf5_schema_id,
            self.dataset_hdf5_schema_version,
        )
        expected = (
            EXPECTED_PLAN_SCHEMA_VERSION,
            EXPECTED_EXECUTION_DOCUMENT_SCHEMA_VERSION,
            EXPECTED_PROVENANCE_SCHEMA_VERSION,
            EXPECTED_PLAN_BINDING_SCHEMA_VERSION,
            EXPECTED_DATASET_JSON_SCHEMA_VERSION,
            EXPECTED_DATASET_HDF5_SCHEMA_ID,
            EXPECTED_DATASET_HDF5_SCHEMA_VERSION,
        )
        if self.available:
            if identities != expected or self.failure is not None:
                raise ValueError("available capabilities require exact schemas")
        elif any(identity is not None for identity in identities) or not self.failure:
            raise ValueError("unavailable capabilities require one failure description")


def _required_module(modules: Mapping[str, object], name: str) -> object:
    module = modules.get(name, _MISSING)
    if module is _MISSING:
        raise ToolsVariationCompatibilityError(
            f"Tools variation boundary is missing module {name}"
        )
    return module


def _required_export(module: object, name: str) -> object:
    value = getattr(module, name, _MISSING)
    if value is _MISSING:
        raise ToolsVariationCompatibilityError(
            f"Tools variation module is missing required export {name}"
        )
    return value


def _require_version(module: object, name: str, expected: int) -> None:
    actual = _required_export(module, name)
    if type(actual) is not int or actual != expected:
        raise ToolsVariationCompatibilityError(
            f"Tools variation boundary has incompatible {name}: "
            f"expected {expected!r}, got {actual!r}"
        )


def _require_string(module: object, name: str, expected: str) -> None:
    actual = _required_export(module, name)
    if type(actual) is not str or actual != expected:
        raise ToolsVariationCompatibilityError(
            f"Tools variation boundary has incompatible {name}: "
            f"expected {expected!r}, got {actual!r}"
        )


def _required_callable(module: object, name: str) -> Callable[..., object]:
    value = _required_export(module, name)
    if not callable(value):
        raise ToolsVariationCompatibilityError(
            f"Tools variation export {name} must be callable"
        )
    return value


def _available_capabilities() -> ToolsVariationCapabilities:
    return ToolsVariationCapabilities(
        available=True,
        module_name=ROOT_MODULE,
        plan_schema_version=EXPECTED_PLAN_SCHEMA_VERSION,
        execution_document_schema_version=(EXPECTED_EXECUTION_DOCUMENT_SCHEMA_VERSION),
        provenance_schema_version=EXPECTED_PROVENANCE_SCHEMA_VERSION,
        plan_binding_schema_version=EXPECTED_PLAN_BINDING_SCHEMA_VERSION,
        dataset_json_schema_version=EXPECTED_DATASET_JSON_SCHEMA_VERSION,
        dataset_hdf5_schema_id=EXPECTED_DATASET_HDF5_SCHEMA_ID,
        dataset_hdf5_schema_version=EXPECTED_DATASET_HDF5_SCHEMA_VERSION,
    )


def _unavailable_capabilities(failure: str) -> ToolsVariationCapabilities:
    return ToolsVariationCapabilities(
        available=False,
        module_name=ROOT_MODULE,
        plan_schema_version=None,
        execution_document_schema_version=None,
        provenance_schema_version=None,
        plan_binding_schema_version=None,
        dataset_json_schema_version=None,
        dataset_hdf5_schema_id=None,
        dataset_hdf5_schema_version=None,
        failure=failure,
    )


class ToolsVariationGateway:
    """Thin consumer of canonical Tools plan and persistence operations."""

    def __init__(
        self,
        *,
        sampler: Callable[..., object],
        plan_loader: Callable[..., object],
        plan_dumper: Callable[..., object],
        metadata_factory: Callable[..., object],
        dataset_serializer: Callable[..., object],
        dataset_deserializer: Callable[..., object],
        dataset_json_writer: Callable[..., object],
        dataset_json_reader: Callable[..., object],
        dataset_csv_writer: Callable[..., object],
        dataset_csv_reader: Callable[..., object],
        dataset_hdf5_writer: Callable[..., object],
        dataset_hdf5_reader: Callable[..., object],
    ) -> None:
        for value, name in (
            (sampler, "sampler"),
            (plan_loader, "plan_loader"),
            (plan_dumper, "plan_dumper"),
            (metadata_factory, "metadata_factory"),
            (dataset_serializer, "dataset_serializer"),
            (dataset_deserializer, "dataset_deserializer"),
            (dataset_json_writer, "dataset_json_writer"),
            (dataset_json_reader, "dataset_json_reader"),
            (dataset_csv_writer, "dataset_csv_writer"),
            (dataset_csv_reader, "dataset_csv_reader"),
            (dataset_hdf5_writer, "dataset_hdf5_writer"),
            (dataset_hdf5_reader, "dataset_hdf5_reader"),
        ):
            if not callable(value):
                raise TypeError(f"{name} must be callable")
        self._sampler = sampler
        self._plan_loader = plan_loader
        self._plan_dumper = plan_dumper
        self._metadata_factory = metadata_factory
        self._dataset_serializer = dataset_serializer
        self._dataset_deserializer = dataset_deserializer
        self._dataset_json_writer = dataset_json_writer
        self._dataset_json_reader = dataset_json_reader
        self._dataset_csv_writer = dataset_csv_writer
        self._dataset_csv_reader = dataset_csv_reader
        self._dataset_hdf5_writer = dataset_hdf5_writer
        self._dataset_hdf5_reader = dataset_hdf5_reader
        self._capabilities = _available_capabilities()

    @classmethod
    def from_modules(cls, modules: Mapping[str, object]) -> ToolsVariationGateway:
        """Validate and bind the supported public Tools modules."""
        if not isinstance(modules, Mapping):
            raise TypeError("modules must be a mapping")
        root = _required_module(modules, ROOT_MODULE)
        metadata = _required_module(modules, EXECUTION_METADATA_MODULE)
        provenance = _required_module(modules, EXECUTION_PROVENANCE_MODULE)
        persistence = _required_module(modules, PERSISTED_PLAN_MODULE)
        _require_version(root, "SCHEMA_VERSION", EXPECTED_PLAN_SCHEMA_VERSION)
        _require_version(
            root,
            "DATASET_JSON_SCHEMA_VERSION",
            EXPECTED_DATASET_JSON_SCHEMA_VERSION,
        )
        _require_string(root, "DATASET_HDF5_SCHEMA_ID", EXPECTED_DATASET_HDF5_SCHEMA_ID)
        _require_version(
            root,
            "DATASET_HDF5_SCHEMA_VERSION",
            EXPECTED_DATASET_HDF5_SCHEMA_VERSION,
        )
        _require_version(
            metadata,
            "EXECUTION_DOCUMENT_SCHEMA_VERSION",
            EXPECTED_EXECUTION_DOCUMENT_SCHEMA_VERSION,
        )
        _require_version(
            provenance,
            "PRODUCER_PROVENANCE_SCHEMA_VERSION",
            EXPECTED_PROVENANCE_SCHEMA_VERSION,
        )
        _require_version(
            persistence,
            "PLAN_BINDING_SCHEMA_VERSION",
            EXPECTED_PLAN_BINDING_SCHEMA_VERSION,
        )
        return cls(
            sampler=_required_callable(root, "sample_inputs"),
            plan_loader=_required_callable(persistence, "persisted_plan_loads"),
            plan_dumper=_required_callable(persistence, "persisted_plan_dumps"),
            metadata_factory=_required_callable(metadata, "make_execution_metadata"),
            dataset_serializer=_required_callable(root, "to_json_dict"),
            dataset_deserializer=_required_callable(root, "from_json_dict"),
            dataset_json_writer=_required_callable(root, "write_json"),
            dataset_json_reader=_required_callable(root, "read_json"),
            dataset_csv_writer=_required_callable(root, "write_csv"),
            dataset_csv_reader=_required_callable(root, "read_csv"),
            dataset_hdf5_writer=_required_callable(root, "write_hdf5"),
            dataset_hdf5_reader=_required_callable(root, "read_hdf5"),
        )

    @property
    def capabilities(self) -> ToolsVariationCapabilities:
        return self._capabilities

    def sample_inputs(self, plan: object) -> object:
        """Return the canonical deterministic Tools sample matrix unchanged."""
        if plan is None:
            raise TypeError("plan must not be None")
        return self._sampler(plan)

    def load_persisted_plan(self, text: str) -> object:
        """Parse canonical or explicitly legacy Tools plan evidence."""
        if not isinstance(text, str):
            raise TypeError("persisted plan must be text")
        return self._plan_loader(text)

    def dump_persisted_plan(
        self, plan: object, *, provenance: object | None = None
    ) -> object:
        """Serialize a plan through Tools without inventing provenance."""
        if plan is None:
            raise TypeError("plan must not be None")
        if provenance is None:
            return self._plan_dumper(plan)
        return self._plan_dumper(plan, provenance=provenance)

    def make_execution_metadata(
        self, plan: object, *, provenance: object | None = None
    ) -> object:
        """Create the canonical registry and execution identity sidecar."""
        if plan is None:
            raise TypeError("plan must not be None")
        if provenance is None:
            return self._metadata_factory(plan)
        return self._metadata_factory(plan, provenance=provenance)

    @staticmethod
    def _validated_path(path: object) -> str | PathLike[str]:
        if not isinstance(path, (str, PathLike)):
            raise TypeError("path must be text or path-like")
        normalized = fspath(path)
        if not isinstance(normalized, str):
            raise TypeError("path must be text or path-like")
        if not normalized.strip():
            raise ValueError("path must not be empty")
        return path

    @staticmethod
    def _validated_dataset(dataset: object) -> object:
        if dataset is None:
            raise TypeError("dataset must not be None")
        return dataset

    def serialize_dataset(self, dataset: object) -> object:
        """Return the canonical Tools JSON-shaped dataset document unchanged."""
        return self._dataset_serializer(self._validated_dataset(dataset))

    def deserialize_dataset(self, document: Mapping[str, object]) -> object:
        """Construct a Tools dataset from a canonical logical document."""
        if not isinstance(document, Mapping):
            raise TypeError("document must be a mapping")
        return self._dataset_deserializer(dict(document))

    def write_dataset_json(self, dataset: object, path: object) -> object:
        """Write one self-contained canonical JSON dataset through Tools."""
        return self._dataset_json_writer(
            self._validated_dataset(dataset), self._validated_path(path)
        )

    def read_dataset_json(self, path: object) -> object:
        """Read one self-contained canonical JSON dataset through Tools."""
        return self._dataset_json_reader(self._validated_path(path))

    def write_dataset_csv(self, dataset: object, path: object) -> object:
        """Write one non-authoritative review-table CSV through Tools."""
        return self._dataset_csv_writer(
            self._validated_dataset(dataset), self._validated_path(path)
        )

    def read_dataset_csv(self, path: object, plan: object) -> object:
        """Read a review-table CSV using an explicitly supplied Tools plan."""
        if plan is None:
            raise TypeError("plan must not be None")
        return self._dataset_csv_reader(self._validated_path(path), plan)

    def write_dataset_hdf5(self, dataset: object, path: object) -> object:
        """Atomically write one canonical content-checked HDF5 dataset."""
        return self._dataset_hdf5_writer(
            self._validated_dataset(dataset), self._validated_path(path)
        )

    def read_dataset_hdf5(self, path: object) -> object:
        """Read one version-qualified, content-checked HDF5 dataset."""
        return self._dataset_hdf5_reader(self._validated_path(path))


def _validate_importer(importer: Importer) -> None:
    if not callable(importer):
        raise TypeError("importer must be callable")


def load_tools_variation_gateway(
    importer: Importer = import_module,
) -> ToolsVariationGateway:
    """Load and strictly validate the configured Tools variation boundary."""
    _validate_importer(importer)
    try:
        modules = {name: importer(name) for name in VARIATION_MODULES}
    except ImportError as exc:
        raise ToolsVariationUnavailableError(
            f"Tools variation contracts are not available: {exc}"
        ) from exc
    return ToolsVariationGateway.from_modules(modules)


def probe_tools_variation(
    importer: Importer = import_module,
) -> ToolsVariationCapabilities:
    """Return capability evidence without making optional Tools fatal."""
    _validate_importer(importer)
    try:
        gateway = load_tools_variation_gateway(importer)
    except (ToolsVariationUnavailableError, ToolsVariationCompatibilityError) as exc:
        return _unavailable_capabilities(str(exc))
    return gateway.capabilities


__all__ = [
    "EXPECTED_DATASET_HDF5_SCHEMA_ID",
    "EXPECTED_DATASET_HDF5_SCHEMA_VERSION",
    "EXPECTED_DATASET_JSON_SCHEMA_VERSION",
    "EXPECTED_EXECUTION_DOCUMENT_SCHEMA_VERSION",
    "EXPECTED_PLAN_BINDING_SCHEMA_VERSION",
    "EXPECTED_PLAN_SCHEMA_VERSION",
    "EXPECTED_PROVENANCE_SCHEMA_VERSION",
    "ROOT_MODULE",
    "ToolsVariationCapabilities",
    "ToolsVariationCompatibilityError",
    "ToolsVariationGateway",
    "ToolsVariationUnavailableError",
    "load_tools_variation_gateway",
    "probe_tools_variation",
]
