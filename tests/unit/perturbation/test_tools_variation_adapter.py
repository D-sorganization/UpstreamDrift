"""Contract tests for the optional canonical Tools variation boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.shared.python import perturbation
from src.shared.python.perturbation.tools_variation_adapter import (
    EXPECTED_DATASET_HDF5_SCHEMA_ID,
    EXPECTED_DATASET_HDF5_SCHEMA_VERSION,
    EXPECTED_DATASET_JSON_SCHEMA_VERSION,
    EXPECTED_EXECUTION_DOCUMENT_SCHEMA_VERSION,
    EXPECTED_PLAN_BINDING_SCHEMA_VERSION,
    EXPECTED_PLAN_SCHEMA_VERSION,
    EXPECTED_PROVENANCE_SCHEMA_VERSION,
    ToolsVariationCompatibilityError,
    ToolsVariationGateway,
    ToolsVariationUnavailableError,
    load_tools_variation_gateway,
    probe_tools_variation,
)

pytestmark = pytest.mark.unit


def test_adapter_is_exposed_by_the_public_perturbation_package() -> None:
    assert perturbation.ToolsVariationGateway is ToolsVariationGateway
    assert perturbation.probe_tools_variation is probe_tools_variation


def _modules() -> dict[str, SimpleNamespace]:
    return {
        "shared.python.swing_sim.variation": SimpleNamespace(
            SCHEMA_VERSION=EXPECTED_PLAN_SCHEMA_VERSION,
            DATASET_JSON_SCHEMA_VERSION=EXPECTED_DATASET_JSON_SCHEMA_VERSION,
            DATASET_HDF5_SCHEMA_ID=EXPECTED_DATASET_HDF5_SCHEMA_ID,
            DATASET_HDF5_SCHEMA_VERSION=EXPECTED_DATASET_HDF5_SCHEMA_VERSION,
            sample_inputs=lambda plan: ("samples", plan),
            to_json_dict=lambda dataset: ("json-document", dataset),
            from_json_dict=lambda document: ("dataset", document),
            write_json=lambda dataset, path: ("write-json", dataset, path),
            read_json=lambda path: ("read-json", path),
            write_csv=lambda dataset, path: ("write-csv", dataset, path),
            read_csv=lambda path, plan: ("read-csv", path, plan),
            write_hdf5=lambda dataset, path: ("write-hdf5", dataset, path),
            read_hdf5=lambda path: ("read-hdf5", path),
        ),
        "shared.python.swing_sim.variation.execution_metadata": SimpleNamespace(
            EXECUTION_DOCUMENT_SCHEMA_VERSION=(
                EXPECTED_EXECUTION_DOCUMENT_SCHEMA_VERSION
            ),
            make_execution_metadata=lambda plan, provenance=None: (
                "metadata",
                plan,
                provenance,
            ),
        ),
        "shared.python.swing_sim.variation.execution_provenance": SimpleNamespace(
            PRODUCER_PROVENANCE_SCHEMA_VERSION=(EXPECTED_PROVENANCE_SCHEMA_VERSION)
        ),
        "shared.python.swing_sim.variation.persisted_plan_io": SimpleNamespace(
            PLAN_BINDING_SCHEMA_VERSION=EXPECTED_PLAN_BINDING_SCHEMA_VERSION,
            persisted_plan_loads=lambda text: ("resolution", text),
            persisted_plan_dumps=lambda plan, provenance=None: (
                "document",
                plan,
                provenance,
            ),
        ),
    }


def _importer(modules: dict[str, object]):
    def import_module(name: str) -> object:
        return modules[name]

    return import_module


def test_gateway_preserves_tools_records_and_canonical_operations() -> None:
    modules = _modules()
    gateway = ToolsVariationGateway.from_modules(modules)
    plan = object()
    provenance = object()
    dataset = object()
    document = {"schema_version": EXPECTED_DATASET_JSON_SCHEMA_VERSION}

    assert gateway.sample_inputs(plan) == ("samples", plan)
    assert gateway.load_persisted_plan("{}") == ("resolution", "{}")
    assert gateway.dump_persisted_plan(plan, provenance=provenance) == (
        "document",
        plan,
        provenance,
    )
    assert gateway.make_execution_metadata(plan, provenance=provenance) == (
        "metadata",
        plan,
        provenance,
    )
    assert gateway.serialize_dataset(dataset) == ("json-document", dataset)
    assert gateway.deserialize_dataset(document) == ("dataset", document)
    assert gateway.write_dataset_json(dataset, "study.json") == (
        "write-json",
        dataset,
        "study.json",
    )
    assert gateway.read_dataset_json("study.json") == ("read-json", "study.json")
    assert gateway.write_dataset_csv(dataset, "study.csv") == (
        "write-csv",
        dataset,
        "study.csv",
    )
    assert gateway.read_dataset_csv("study.csv", plan) == (
        "read-csv",
        "study.csv",
        plan,
    )
    assert gateway.write_dataset_hdf5(dataset, "study.h5") == (
        "write-hdf5",
        dataset,
        "study.h5",
    )
    assert gateway.read_dataset_hdf5("study.h5") == ("read-hdf5", "study.h5")
    assert gateway.capabilities.available is True
    assert (
        gateway.capabilities.dataset_json_schema_version
        == EXPECTED_DATASET_JSON_SCHEMA_VERSION
    )
    assert (
        gateway.capabilities.dataset_hdf5_schema_id == EXPECTED_DATASET_HDF5_SCHEMA_ID
    )
    assert (
        gateway.capabilities.dataset_hdf5_schema_version
        == EXPECTED_DATASET_HDF5_SCHEMA_VERSION
    )
    assert gateway.capabilities.failure is None


def test_probe_is_import_safe_when_tools_variation_is_absent() -> None:
    def missing_importer(name: str) -> object:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    capability = probe_tools_variation(importer=missing_importer)

    assert capability.available is False
    assert capability.plan_schema_version is None
    assert capability.execution_document_schema_version is None
    assert capability.provenance_schema_version is None
    assert capability.plan_binding_schema_version is None
    assert capability.dataset_json_schema_version is None
    assert capability.dataset_hdf5_schema_id is None
    assert capability.dataset_hdf5_schema_version is None
    assert "No module named" in (capability.failure or "")
    with pytest.raises(ToolsVariationUnavailableError, match="not available"):
        load_tools_variation_gateway(importer=missing_importer)


@pytest.mark.parametrize(
    ("module_name", "field", "value", "message"),
    [
        (
            "shared.python.swing_sim.variation",
            "SCHEMA_VERSION",
            3,
            "incompatible",
        ),
        (
            "shared.python.swing_sim.variation.execution_metadata",
            "make_execution_metadata",
            None,
            "callable",
        ),
        (
            "shared.python.swing_sim.variation.persisted_plan_io",
            "persisted_plan_loads",
            None,
            "callable",
        ),
        (
            "shared.python.swing_sim.variation",
            "DATASET_HDF5_SCHEMA_VERSION",
            2,
            "incompatible",
        ),
        (
            "shared.python.swing_sim.variation",
            "DATASET_HDF5_SCHEMA_ID",
            "other/dataset",
            "incompatible",
        ),
        (
            "shared.python.swing_sim.variation",
            "write_hdf5",
            None,
            "callable",
        ),
    ],
)
def test_gateway_rejects_incompatible_or_malformed_tools_modules(
    module_name: str,
    field: str,
    value: object,
    message: str,
) -> None:
    modules = _modules()
    setattr(modules[module_name], field, value)

    with pytest.raises(ToolsVariationCompatibilityError, match=message):
        ToolsVariationGateway.from_modules(modules)

    capability = probe_tools_variation(importer=_importer(modules))
    assert capability.available is False
    assert message in (capability.failure or "")


def test_gateway_validates_public_text_and_plan_boundaries() -> None:
    gateway = ToolsVariationGateway.from_modules(_modules())

    with pytest.raises(TypeError, match="plan must not be None"):
        gateway.sample_inputs(None)
    with pytest.raises(TypeError, match="persisted plan must be text"):
        gateway.load_persisted_plan(b"{}")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="plan must not be None"):
        gateway.dump_persisted_plan(None)
    with pytest.raises(TypeError, match="dataset must not be None"):
        gateway.serialize_dataset(None)
    with pytest.raises(TypeError, match="document must be a mapping"):
        gateway.deserialize_dataset([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="path must be text or path-like"):
        gateway.read_dataset_hdf5(b"study.h5")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="path must not be empty"):
        gateway.write_dataset_json(object(), "")
    with pytest.raises(TypeError, match="plan must not be None"):
        gateway.read_dataset_csv("study.csv", None)


def test_gateway_requires_complete_named_module_mapping() -> None:
    modules = _modules()
    modules.pop("shared.python.swing_sim.variation.persisted_plan_io")

    with pytest.raises(ToolsVariationCompatibilityError, match="missing module"):
        ToolsVariationGateway.from_modules(modules)
