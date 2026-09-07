"""Provider-authority contract tests for the AffineDrift companion catalog."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import jsonschema
import pytest

from src.shared.python.config.model_registry import ModelRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "docs/api/contracts/upstreamdrift-companion-v1.schema.json"
CAPABILITIES_SCHEMA_PATH = (
    REPO_ROOT / "docs/api/contracts/upstreamdrift-companion-capabilities-v1.schema.json"
)
SCREENSHOTS_SCHEMA_PATH = (
    REPO_ROOT / "docs/api/contracts/upstreamdrift-companion-screenshots-v1.schema.json"
)
BUILDER_MODULES = (
    REPO_ROOT / "scripts/companion_catalog.py",
    REPO_ROOT / "scripts/companion_publication.py",
)
pytestmark = pytest.mark.unit


def _catalog_module():
    from scripts import companion_catalog

    return companion_catalog


def _pyproject_requires_python() -> str:
    """Read ``[project].requires-python`` from the provider's pyproject.toml."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    return str(data["project"]["requires-python"])


def _requires_python_schema() -> dict:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return schema["$defs"]["compatibility"]["properties"]["requires_python"]


def test_model_registry_explicit_local_only_ignores_hybrid_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit discovery mode is stronger than workstation environment."""
    monkeypatch.setenv("UPSTREAM_DRIFT_DISCOVERY_MODE", "provider-first")

    registry = ModelRegistry(
        config_path=REPO_ROOT / "src/config/models.yaml",
        discovery_mode="local-only",
    )

    assert registry.discovery_mode == "local-only"
    assert len(registry.get_all_models()) == 56


def test_catalog_reconciles_current_registries_without_schema_count_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Migration baselines are tested here, not frozen into the public schema."""
    monkeypatch.setenv("UPSTREAM_DRIFT_DISCOVERY_MODE", "provider-first")
    monkeypatch.setenv("UPSTREAM_DRIFT_PROVIDER_ROOTS", r"Z:\not-authoritative")
    catalog = _catalog_module().build_catalog(REPO_ROOT, require_clean=False)

    assert catalog["summary"] == {
        "raw_launcher_records": 49,
        "local_model_records": 56,
        "program_records": 70,
        "feature_records": 41,
        "feature_surface_paths": 82,
        "workflow_records": 15,
        "executable_workflow_records": 14,
    }
    assert len({record["id"] for record in catalog["programs"]}) == 70
    assert len({record["id"] for record in catalog["features"]}) == 41

    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    for current_count in (49, 56, 70, 41, 82):
        assert f'"const": {current_count}' not in schema_text
        assert f'"minItems": {current_count}' not in schema_text
        assert f'"maxItems": {current_count}' not in schema_text


def test_catalog_is_environment_independent_and_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    companion_catalog = _catalog_module()
    baseline = companion_catalog.render_catalog(
        companion_catalog.build_catalog(REPO_ROOT, require_clean=False)
    )

    monkeypatch.setenv("UPSTREAM_DRIFT_DISCOVERY_MODE", "hybrid")
    monkeypatch.setenv("UPSTREAM_DRIFT_PROVIDER_ROOTS", r"Z:\another-provider")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1")
    repeated = companion_catalog.render_catalog(
        companion_catalog.build_catalog(REPO_ROOT, require_clean=False)
    )

    assert repeated == baseline
    assert baseline.endswith(b"\n")


def test_catalog_validates_strict_schema_and_resolves_references() -> None:
    catalog = _catalog_module().build_catalog(REPO_ROOT, require_clean=False)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator(schema).validate(catalog)
    program_ids = {record["id"] for record in catalog["programs"]}
    feature_ids = {record["id"] for record in catalog["features"]}
    assert all(
        tile_id in program_ids
        for feature in catalog["features"]
        for tile_id in feature["program_ids"]
    )
    assert all(
        feature_id in feature_ids
        for program in catalog["programs"]
        for feature_id in program["feature_ids"]
    )
    assert all(
        (REPO_ROOT / surface["path"]).is_file()
        for feature in catalog["features"]
        for surface in feature["surfaces"]
    )

    invalid = dict(catalog)
    invalid["undeclared"] = True
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(invalid))


def test_every_declared_object_schema_is_strict() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value.get("additionalProperties") is False
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)


def test_schema_requires_python_is_not_a_duplicated_version_literal() -> None:
    """The schema must constrain the shape of ``requires_python``, never its value.

    A ``const`` (or any embedded version literal) makes the schema a second
    source of truth for the supported Python range, which silently rejects every
    legitimate ``pyproject.toml`` change.  Guard against re-introducing it.
    """
    node = _requires_python_schema()

    assert "const" not in node, (
        "requires_python must not be a const: the value is derived from "
        "pyproject.toml [project].requires-python at export time"
    )
    assert "enum" not in node
    assert node["type"] == "string"
    assert not re.search(
        r"\d+\.\d+", node["pattern"]
    ), f"schema pattern {node['pattern']!r} embeds a Python version literal"


def test_schema_accepts_the_live_pyproject_requires_python() -> None:
    """Whatever ``pyproject.toml`` declares must validate against the schema."""
    pattern = _requires_python_schema()["pattern"]

    assert re.match(pattern, _pyproject_requires_python()) is not None


@pytest.mark.parametrize(
    "specifier",
    ["3.11", ">3.11", ">=3", "", ">=3.11,<3.13,!=3.12", ">=3.11, <3.13"],
)
def test_schema_rejects_malformed_requires_python(specifier: str) -> None:
    """The shape constraint still fails closed on non-canonical specifiers."""
    pattern = _requires_python_schema()["pattern"]

    assert re.match(pattern, specifier) is None


def test_catalog_requires_python_is_derived_from_pyproject() -> None:
    """The exported manifest must copy ``requires-python`` verbatim."""
    catalog = _catalog_module().build_catalog(REPO_ROOT, require_clean=False)

    assert catalog["compatibility"]["requires_python"] == _pyproject_requires_python()


def test_catalog_supported_minors_match_the_pyproject_range() -> None:
    """The advertised minors must be exactly those the declared range admits."""
    catalog = _catalog_module().build_catalog(REPO_ROOT, require_clean=False)
    specifier = _pyproject_requires_python()
    match = re.fullmatch(r">=3\.(\d+),<3\.(\d+)", specifier)

    assert match is not None, (
        f"requires-python {specifier!r} is not a bounded 3.x range, so the "
        "supported minors cannot be derived; widen this test deliberately"
    )
    floor, ceiling = int(match.group(1)), int(match.group(2))
    expected = [f"3.{minor}" for minor in range(floor, ceiling)]

    assert catalog["compatibility"]["supported_python_minors"] == expected


def test_catalog_compatibility_contract_is_exact() -> None:
    catalog = _catalog_module().build_catalog(REPO_ROOT, require_clean=False)

    assert catalog["compatibility"] == {
        "requires_python": _pyproject_requires_python(),
        "supported_python_minors": ["3.11", "3.12"],
        "verification_command": {
            "executable": "python",
            "arguments": ["scripts/ci/verify_installation.py"],
        },
    }
    assert {engine["id"]: engine["support_tier"] for engine in catalog["engines"]} == {
        "drake": "extended",
        "mujoco": "supported",
        "myosuite": "experimental",
        "opensim": "experimental",
        "pinocchio": "extended",
    }


def test_authoritative_export_refuses_dirty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    companion_catalog = _catalog_module()
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, shell=False)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
        shell=False,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Contract Test"],
        check=True,
        shell=False,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True, shell=False
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "fixture"],
        check=True,
        shell=False,
    )
    (tmp_path / "dirty.txt").write_text("not authoritative\n", encoding="utf-8")

    with pytest.raises(
        companion_catalog.CatalogAuthorityError, match="requires a clean tree"
    ):
        companion_catalog.build_catalog(tmp_path)


def test_authoritative_export_refuses_mismatched_ci_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    companion_catalog = _catalog_module()
    monkeypatch.setenv("GITHUB_SHA", "0" * 40)

    with pytest.raises(companion_catalog.CatalogAuthorityError, match="does not match"):
        companion_catalog.build_catalog(REPO_ROOT, require_clean=False)


def test_catalog_pins_exact_provider_and_input_provenance() -> None:
    catalog = _catalog_module().build_catalog(REPO_ROOT, require_clean=False)
    source = catalog["source"]
    tools = next(
        provider for provider in catalog["providers"] if provider["id"] == "tools"
    )

    assert len(source["commit"]) == 40
    assert tools["pin_kind"] == "gitlink"
    assert tools["pinned_commit"] == "3d93bb2c89813e17551814d3be7e895f791e29af"
    assert tools["vendor_path"] == "vendor/ud-tools"
    assert {item["path"] for item in source["inputs"]} == {
        "docs/api/contracts/upstreamdrift-companion-compatibility-v1.json",
        "docs/api/contracts/upstreamdrift-companion-v1.schema.json",
        "pyproject.toml",
        "scripts/config/companion_workflows.v1.json",
        "src/config/feature_parity.json",
        "src/config/launcher_manifest.json",
        "src/config/models.yaml",
        "tests/fixtures/launch_monitor/trackman.csv",
    }
    assert all(len(item["sha256"]) == 64 for item in source["inputs"])


def test_catalog_keeps_status_dimensions_independent() -> None:
    catalog = _catalog_module().build_catalog(REPO_ROOT, require_clean=False)
    program = catalog["programs"][0]
    feature = catalog["features"][0]

    assert set(program) >= {
        "maturity",
        "availability",
        "support_tier",
        "scientific_qualification",
        "legacy_statuses",
    }
    assert set(feature) >= {"parity", "scientific_qualification"}
    assert {engine["support_tier"] for engine in catalog["engines"]} == {
        "supported",
        "extended",
        "experimental",
    }


def test_write_catalog_emits_detached_sha256(tmp_path: Path) -> None:
    companion_catalog = _catalog_module()
    output = tmp_path / "upstreamdrift-companion.v1.json"
    digest_path = companion_catalog.write_catalog(
        REPO_ROOT,
        output=output,
        require_clean=False,
    )

    payload = output.read_bytes()
    assert digest_path == output.with_suffix(".json.sha256")
    assert digest_path.read_text(encoding="ascii") == (
        f"{hashlib.sha256(payload).hexdigest()}  {output.name}\n"
    )


@pytest.mark.parametrize(
    "value",
    [Path("../outside.json"), Path("/absolute.json"), Path(r"C:\outside.json")],
)
def test_repo_relative_input_contract_rejects_external_paths(value: Path) -> None:
    companion_catalog = _catalog_module()
    with pytest.raises(ValueError, match="repo-relative"):
        companion_catalog.validate_repo_relative(value)


# --- #9416: import-free builder and consumer payloads -----------------------


def _assert_strict_objects(schema: object) -> None:
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False
        for child in schema.values():
            _assert_strict_objects(child)
    elif isinstance(schema, list):
        for child in schema:
            _assert_strict_objects(child)


@pytest.mark.parametrize("module_path", BUILDER_MODULES, ids=lambda p: p.name)
def test_builder_modules_import_nothing_from_src(module_path: Path) -> None:
    """The publication job installs only jsonschema+pyyaml; src.* is off-limits."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offending.extend(
                alias.name for alias in node.names if alias.name.split(".")[0] == "src"
            )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root == "src":
                offending.append(f"from {'.' * node.level}{node.module} import ...")
    assert offending == [], offending


def test_builder_imports_without_numpy(tmp_path: Path) -> None:
    """A poisoned numpy on PYTHONPATH proves the builder never needs it."""
    poison = tmp_path / "poison" / "numpy"
    poison.mkdir(parents=True)
    (poison / "__init__.py").write_text(
        'raise ImportError("numpy must not be imported by the companion builder")\n',
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path / "poison"), str(REPO_ROOT)])
    env.pop("PYTHONSAFEPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.companion_catalog, scripts.companion_publication; "
            "assert 'numpy' not in sys.modules; print('import-free ok')",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        encoding="utf-8",
        shell=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "import-free ok" in completed.stdout


def test_local_reader_matches_runtime_registry_facts() -> None:
    """The standalone YAML reader reproduces every fact the catalog exports."""
    companion_catalog = _catalog_module()
    payload = (REPO_ROOT / "src/config/models.yaml").read_bytes()
    local = {
        record.id: record for record in companion_catalog.read_local_models(payload)
    }
    registry = ModelRegistry(
        config_path=REPO_ROOT / "src/config/models.yaml",
        strict=True,
        discovery_mode="local-only",
    )
    runtime = {model.id: model for model in registry.get_all_models()}

    assert list(local) == list(runtime)
    for model_id, model in runtime.items():
        record = local[model_id]
        assert (record.name, record.description, record.type, record.path) == (
            model.name,
            model.description,
            model.type,
            model.path,
        )
        assert record.engine_type == model.engine_type
        assert record.provider == model.provider
        assert record.hidden == model.hidden
        assert record.capabilities == model.capabilities
        if model.launcher is None:
            assert record.launcher is None
        else:
            assert record.launcher.category == model.launcher.category
            assert record.launcher.status == model.launcher.status


def test_local_reader_tolerates_unknown_keys_and_synthesises_virtual_paths() -> None:
    companion_catalog = _catalog_module()
    document = (
        b"models:\n"
        b"  - id: ' pinn '\n"
        b"    name: 'PINN'\n"
        b"    description: ''\n"
        b"    type: physics_informed\n"
        b"    maturity: beta\n"
        b"    surfaces: [pyqt]\n"
        b"    help: {doc: x}\n"
        b"    feature_id: f1\n"
        b"    web: {mode: route}\n"
        b"    capabilities: ['Forward_Dynamics', 'forward-dynamics', 'PINN']\n"
        b"    launcher: {category: simulation, logo: a.png, status: beta}\n"
    )

    [record] = companion_catalog.read_local_models(document)

    assert record.id == "pinn"
    assert record.description == "PINN"
    assert record.path == "virtual/physics_informed/pinn"
    assert record.capabilities == ("dynamics", "pinn")
    assert record.launcher is not None and record.launcher.category == "simulation"


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("models:\n  - id: a\n    name: A\n    type: mjcf\n", "missing required"),
        (
            "models:\n  - id: a\n    name: A\n    description: d\n    type: mjcf\n"
            "    path: ''\n",
            "non-empty string",
        ),
        (
            "models:\n  - id: a\n    name: A\n    description: d\n    type: mjcf\n"
            "    path: p\n    launcher: {category: bogus, logo: l, status: s}\n",
            "unknown launcher category",
        ),
        ("models: {}\n", "must be a list"),
        ("- not a mapping\n", "root must be a mapping"),
    ],
)
def test_local_reader_fails_closed_like_the_strict_registry(
    document: str, message: str
) -> None:
    companion_catalog = _catalog_module()
    with pytest.raises(companion_catalog.CatalogAuthorityError, match=message):
        companion_catalog.read_local_models(document.encode("utf-8"))


def test_capabilities_payload_resolves_every_program_and_validates() -> None:
    companion_catalog = _catalog_module()
    payloads = companion_catalog.build_payload_set(REPO_ROOT, require_clean=False)
    capabilities = payloads.capabilities
    schema = json.loads(CAPABILITIES_SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator(schema).validate(capabilities)
    program_ids = {program["id"] for program in payloads.catalog["programs"]}
    assert {program["id"] for program in capabilities["programs"]} == program_ids
    assert all(
        program_id in program_ids
        for record in capabilities["capabilities"]
        for program_id in record["program_ids"]
    )
    by_program = {p["id"]: p["capability_ids"] for p in capabilities["programs"]}
    for record in capabilities["capabilities"]:
        assert all(record["id"] in by_program[pid] for pid in record["program_ids"])
        assert record["program_ids"] == sorted(set(record["program_ids"]))
    assert [c["id"] for c in capabilities["capabilities"]] == sorted(
        c["id"] for c in capabilities["capabilities"]
    )
    assert {engine["id"] for engine in capabilities["engines"]} == {
        engine["id"] for engine in payloads.catalog["engines"]
    }
    assert capabilities["source"]["commit"] == payloads.catalog["source"]["commit"]
    assert capabilities["manifest_id"] == "upstreamdrift-companion-capabilities"
    # models.yaml and launcher_manifest.json both contribute; the union is
    # what AffineDrift renders, so no program may be capability-free today.
    assert all(by_program[p] for p in program_ids), "program without capabilities"


def test_screenshots_payload_is_pending_metadata_only() -> None:
    companion_catalog = _catalog_module()
    payloads = companion_catalog.build_payload_set(REPO_ROOT, require_clean=False)
    screenshots = payloads.screenshots
    schema = json.loads(SCREENSHOTS_SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator(schema).validate(screenshots)
    visible = [p["id"] for p in payloads.catalog["programs"] if not p["hidden"]]
    assert [r["program_id"] for r in screenshots["records"]] == visible
    assert len(visible) < len(payloads.catalog["programs"])
    for record in screenshots["records"]:
        assert record["status"] == "pending"
        assert record["id"] == f"{record['program_id']}-primary"
        assert record["reason"] == companion_catalog.SCREENSHOT_PENDING_REASON
        assert record["artifact_class"] == "illustrative"
        assert record["visible_limitations"] == []
        for field in (
            "path",
            "sha256",
            "width",
            "height",
            "viewport",
            "theme",
            "capture_workflow_id",
            "capture_step_id",
            "alt_text",
            "caption",
        ):
            assert record[field] is None, field
    # The top-level manifest inventory stays empty until #9191 lands.
    assert payloads.catalog["screenshots"] == []


def test_screenshot_schema_encodes_pending_and_captured_conditions() -> None:
    schema = json.loads(SCREENSHOTS_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    base = {
        "$schema": schema["$id"],
        "schema_version": "1.0.0",
        "manifest_id": "upstreamdrift-companion-screenshots",
        "source": {
            "repository": "https://github.com/D-sorganization/UpstreamDrift",
            "commit": "1" * 40,
        },
    }
    pending = {
        "id": "p-primary",
        "program_id": "p",
        "status": "pending",
        "path": None,
        "sha256": None,
        "width": None,
        "height": None,
        "viewport": None,
        "theme": None,
        "capture_workflow_id": None,
        "capture_step_id": None,
        "alt_text": None,
        "caption": None,
        "visible_limitations": [],
        "artifact_class": "illustrative",
        "reason": "No governed capture exists at this commit (#9191).",
    }
    captured = dict(
        pending,
        status="captured",
        path="docs/screenshots/p.png",
        sha256="a" * 64,
        width=1280,
        height=800,
        viewport={"width": 1280, "height": 800},
        theme="dark",
        capture_workflow_id="wf",
        capture_step_id="step",
        alt_text="The launcher",
        caption="Launcher tile",
        reason=None,
    )

    validator.validate(dict(base, records=[pending, captured]))
    for broken in (
        dict(pending, reason=""),
        dict(pending, path="x.png"),
        dict(captured, sha256=None),
        dict(captured, alt_text=None),
        dict(captured, theme=None),
    ):
        assert list(validator.iter_errors(dict(base, records=[broken])))


def test_capabilities_and_screenshots_schemas_are_strict() -> None:
    for path in (CAPABILITIES_SCHEMA_PATH, SCHEMA_PATH, SCREENSHOTS_SCHEMA_PATH):
        schema = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        _assert_strict_objects(schema)
        assert schema["properties"]["schema_version"] == {"const": "1.0.0"}


def test_capabilities_payload_refuses_dangling_program_ids() -> None:
    companion_catalog = _catalog_module()
    catalog = companion_catalog.build_catalog(REPO_ROOT, require_clean=False)

    with pytest.raises(companion_catalog.CatalogAuthorityError, match="unknown"):
        companion_catalog._capabilities_payload(catalog, {"not-a-program": ["x"]})
