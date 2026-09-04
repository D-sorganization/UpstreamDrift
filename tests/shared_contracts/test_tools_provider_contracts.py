from __future__ import annotations

import contextlib
import importlib
import json
import sys
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest


def _normalized_path(value: str) -> str:
    return value.replace("\\", "/").lower()


@contextlib.contextmanager
def _fresh_provider_import(name: str) -> Iterator[None]:
    """Import ``name`` through the promoted Tools paths, then restore the cache.

    These contracts must observe where a *fresh* import resolves. By the time a
    test runs, the local copies are already cached: conftest and plugin imports
    execute before this package's ``pytest_configure`` promotes the Tools paths,
    and ``shared.python.import_aliases`` canonicalises every provider spelling
    to whichever copy loaded first. Asserting on the cached module therefore
    tested import history, not resolution order - the suite failed even when
    the vendored tree was correctly provisioned and first on ``sys.path``.

    The cache surgery is scoped to this context manager and fully restored, so
    the surrounding session (which legitimately tests the local copies in the
    ``tests (3.x)`` lanes) keeps its module identities untouched.
    """
    prefixes = [
        name,
        f"shared.python.{name}",
        f"src.shared.python.{name}",
        f"src.{name}",
        "shared",
    ]
    # ``src.shared`` is the downstream namespace. Clearing it wholesale also
    # evicts UpstreamDrift-owned packages when Tools is first on PYTHONPATH.
    if name == "sidekick":
        # The alias finder canonicalises sidekick against the deprecated
        # upstream_drift_tools spellings too; a cached local copy of ANY of
        # them re-binds sidekick to the local tree.
        prefixes += [
            "upstream_drift_tools",
            "shared.python.upstream_drift_tools",
            "src.shared.python.upstream_drift_tools",
        ]
    prefixes = tuple(prefixes)

    def _affected(module_name: str) -> bool:
        return any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for prefix in prefixes
        )

    saved = {key: mod for key, mod in sys.modules.items() if _affected(key)}
    for key in saved:
        del sys.modules[key]
    try:
        yield
    finally:
        for key in [key for key in sys.modules if _affected(key)]:
            del sys.modules[key]
        sys.modules.update(saved)


def _assert_from_tools(path: Path) -> None:
    normalized = _normalized_path(str(path))
    assert any(
        marker in normalized
        for marker in (
            "/_tools_dep/",
            "/vendor/ud-tools/",
            "/repositories/tools/",
            "/repositories/tools-worktrees/",
            "/tools/",
        )
    ), f"Expected Tools-backed provider path, got: {path}"


@pytest.mark.unit
def test_fresh_provider_import_preserves_downstream_modules() -> None:
    """Refreshing one Tools provider must not evict UpstreamDrift packages."""
    module_name = "src.shared.python.perturbation.tools_variation_adapter"
    module = importlib.import_module(module_name)

    with _fresh_provider_import("swing_sim"):
        assert sys.modules[module_name] is module


def test_signal_toolkit_imports_resolve_from_tools_provider() -> None:
    with _fresh_provider_import("signal_toolkit"):
        module = importlib.import_module("signal_toolkit")
        _assert_from_tools(Path(module.__file__).resolve())

        signal = module.SignalGenerator.sinusoid(
            np.linspace(0.0, 1.0, 16), amplitude=1.0, frequency=2.0
        )
        assert len(signal.values) == 16


def test_humanoid_character_builder_imports_resolve_from_tools_provider() -> None:
    with _fresh_provider_import("humanoid_character_builder"):
        module = importlib.import_module("humanoid_character_builder")
        _assert_from_tools(Path(module.__file__).resolve())

        params = module.BodyParameters(height_m=1.75, mass_kg=72.0)
        assert params.height_m == 1.75
        assert params.mass_kg == 72.0


def test_model_generation_imports_resolve_from_tools_provider() -> None:
    with _fresh_provider_import("model_generation"):
        module = importlib.import_module("model_generation")
        _assert_from_tools(Path(module.__file__).resolve())

        # Execute actual contract behavior instead of just checking callability
        urdf = module.quick_urdf(height_m=1.8, mass_kg=80.0, robot_name="test_robot")
        assert "test_robot" in urdf
        assert "<link" in urdf
        assert module.DEFAULT_HEIGHT_M > 0


def test_sidekick_imports_resolve_from_tools_provider() -> None:
    with _fresh_provider_import("sidekick"):
        module = importlib.import_module("sidekick")
        _assert_from_tools(Path(module.__file__).resolve())

        state_manager = importlib.import_module("sidekick.utils.state_manager")
        _assert_from_tools(Path(state_manager.__file__).resolve())

        # Execute actual contract behavior
        manager = state_manager.StateManager()
        manager.save_state("test_key", {"test_key": "test_value"})
        assert manager.load_state("test_key") == {"test_key": "test_value"}


@pytest.mark.integration
def test_rotating_base_provider_retains_complete_qualified_authority() -> None:
    """The pinned Tools provider must retain every scientific boundary."""
    with _fresh_provider_import("swing_sim"):
        module = importlib.import_module("shared.python.swing_sim.rotating_base")
        module_path = Path(module.__file__).resolve()
        _assert_from_tools(module_path)

        assert module.EXPECTED_UPSTREAM_SOURCE_REVISION == (
            "967c40f54cc03f8cae89cde09268d62771d220fe"
        )
        assert module.EXPECTED_STUDY_SHA256 == (
            "e6a55e6cf91e51f21fe3eb8bcb07b990a7798f18abcaf5ca73f5214cb6c5f9ec"
        )
        assert module.EXPECTED_RUN_CATALOG_SHA256 == (
            "66493b833955c6492a00eae4a600df795df60a6f473f9a11c403084b58e51678"
        )
        assert module.MODEL_TIER == ("planar_rotating_base_two_hand_compliant_club")

        study = module.load_embedded_qualified_study().study
        assert study.attempted_case_count == 18
        assert study.valid_case_count == 13
        assert [case.case_index for case in study.cases if not case.valid] == [
            6,
            7,
            8,
            15,
            16,
        ]
        assert study.human_coaching_supported is False

        catalog_path = (
            module_path.parent / "resources" / "rotating_base_registered_runs_v1.json"
        )
        catalog_text = catalog_path.read_text(encoding="utf-8").rstrip("\n")
        assert sha256(catalog_text.encode("utf-8")).hexdigest() == (
            module.EXPECTED_RUN_CATALOG_SHA256
        )
        catalog = json.loads(catalog_text)
        assert catalog["attempted_run_count"] == 18
        assert catalog["source_revision"] == module.EXPECTED_UPSTREAM_SOURCE_REVISION
        assert catalog["study_sha256"] == module.EXPECTED_STUDY_SHA256
        assert [run["request"]["case_index"] for run in catalog["runs"]] == list(
            range(18)
        )
        assert [
            run["case"]["case_index"]
            for run in catalog["runs"]
            if not run["case"]["valid"]
        ] == [6, 7, 8, 15, 16]
        assert all(
            run["boundaries"]
            == {
                "coaching_recommendation": "unsupported",
                "coordinate_semantics": "nonanatomical_model_coordinate",
                "human_validation": "unavailable",
            }
            for run in catalog["runs"]
        )


@pytest.mark.integration
def test_variation_gateway_executes_immutable_tools_contracts(
    tmp_path: Path,
) -> None:
    """Exercise persistence and analysis through the real pinned provider."""
    with _fresh_provider_import("swing_sim"):
        variation = importlib.import_module("shared.python.swing_sim.variation")
        _assert_from_tools(Path(variation.__file__).resolve())
        gateway_module = importlib.import_module(
            "src.shared.python.perturbation.tools_variation_adapter"
        )
        gateway = gateway_module.load_tools_variation_gateway()

        input_name = f"{variation.CATEGORY_LAUNCH}.ball_speed_mph"
        plan = variation.VariationPlan(
            mode="launch",
            noise=(variation.NoiseSpec(input_name, scale=1.0),),
            n_runs=4,
            seed=17,
        )
        samples = gateway.sample_inputs(plan)
        np.testing.assert_array_equal(samples, gateway.sample_inputs(plan))

        outputs = np.column_stack((samples[:, 0], -samples[:, 0]))
        outputs[1] = np.nan
        dataset = variation.VariationDataset(
            plan=plan,
            input_names=(input_name,),
            inputs=samples,
            output_names=("carry_m", "lateral_m"),
            outputs=outputs,
            success=np.array((True, False, True, True)),
        )
        logical_round_trip = gateway.deserialize_dataset(
            gateway.serialize_dataset(dataset)
        )
        np.testing.assert_array_equal(logical_round_trip.inputs, dataset.inputs)
        np.testing.assert_allclose(
            logical_round_trip.outputs,
            dataset.outputs,
            equal_nan=True,
        )

        for suffix, writer, reader in (
            ("json", gateway.write_dataset_json, gateway.read_dataset_json),
            (
                "csv",
                gateway.write_dataset_csv,
                lambda path: gateway.read_dataset_csv(path, plan),
            ),
            ("h5", gateway.write_dataset_hdf5, gateway.read_dataset_hdf5),
        ):
            path = tmp_path / f"variation.{suffix}"
            writer(dataset, path)
            restored = reader(path)
            np.testing.assert_array_equal(restored.inputs, dataset.inputs)
            np.testing.assert_allclose(
                restored.outputs,
                dataset.outputs,
                equal_nan=True,
            )
            np.testing.assert_array_equal(restored.success, dataset.success)

        stats = {item.name: item for item in gateway.summarize_dataset(dataset)}
        assert stats["carry_m"].n == 3
        rank = gateway.compute_spearman_attribution(dataset)
        assert rank[0, 0] == pytest.approx(1.0)
        assert rank[0, 1] == pytest.approx(-1.0)
        oat = gateway.build_oat_sensitivity(
            (input_name,),
            dataset.output_names,
            np.array(((2.0, 1.0),)),
        )
        np.testing.assert_array_equal(oat.normalized, np.ones((1, 2)))

        positions = np.zeros((4, 3, 1, 3), dtype=float)
        offsets = np.array((-1.0, 0.0, 1.0, 2.0))
        positions[:, :, 0, 0] = offsets[:, np.newaxis] * np.array((0.1, 0.2, 1.0))
        ensemble = variation.EnsemblePositionTraces(
            variation=dataset,
            sample_times_s=np.array((0.0, 0.5, 1.0)),
            coordinate_frame="swing.world",
            point_ids=("swing.clubhead",),
            positions_m=positions,
            sample_valid=np.ones((4, 3), dtype=bool),
            impact_sample_indices=np.array((2, -1, 2, -1)),
        )
        dispersion = gateway.compute_geometry_dispersion(ensemble)
        np.testing.assert_array_equal(dispersion.count[:, 0], np.full(3, 4))
        quiet = gateway.find_quiet_zones(
            dispersion,
            variation.LowVariabilityCriteria(
                max_rms_radius_m=0.25,
                min_samples=2,
                point_ids=("swing.clubhead",),
            ),
        )
        assert [(interval.start_index, interval.end_index) for interval in quiet] == [
            (0, 1)
        ]


@pytest.mark.integration
def test_rate_of_closure_provider_exposes_governed_analysis_policy() -> None:
    """The immutable provider must expose the reviewed R14.3 policy surface."""
    with _fresh_provider_import("rate_of_closure"):
        package = importlib.import_module("rate_of_closure")
        _assert_from_tools(Path(package.__file__).resolve())
        policy = importlib.import_module("rate_of_closure.variation.analysis_policy")
        _assert_from_tools(Path(policy.__file__).resolve())

        assert policy.ANALYSIS_EXECUTIONS == (
            "all_together",
            "individual",
            "both",
        )
        assert policy.planned_analysis_runs(20, 3, "all_together") == 20
        assert policy.planned_analysis_runs(20, 3, "individual") == 60
        assert policy.planned_analysis_runs(20, 3, "both") == 80

        manifest_path = (
            Path(package.__file__).resolve().parent / "visual_baselines.v1.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        variation = next(
            entry
            for entry in manifest["baselines"]
            if entry["surface"] == "pyqt" and entry["tab_id"] == "variation"
        )
        assert variation["sha256"] == (
            "650267b346dab8651b6163d83046ed46cbe604c83c71908cca2b1168e84a78cd"
        )
        assert variation["tolerance"] == {
            "changed_channel_threshold": 1,
            "max_mean_channel_delta_microunits": 200,
            "max_changed_pixel_fraction_microunits": 250,
        }
