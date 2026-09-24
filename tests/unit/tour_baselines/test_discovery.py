"""Tests for baseline discovery, portable loading and safe model presets (TB-10 #10595).

TDD suite verifying:
1. SafeModelPreset creation from BaselinePackage with full state, geometry, inertia, and controls.
2. Fail-closed compatibility checking refusing topology mismatches and missing dependencies.
3. Safe session cloning preserving user workspace without silent overwrites.
4. BaselineDiscoveryService scanning across configurable search paths (no hardcoded machine paths).
5. Multi-field filtering (model, club, horizon, qualification status).
6. Nominated baseline distinguished from newest / lowest-loss candidate.
7. Unverified/unqualified baselines can NEVER be auto-selected as default presets.
8. Portable export and clean-machine import with SHA-256 verification and dependency diagnostics.
9. Deterministic re-indexing preserving baseline identities.
10. Headless CLI / API list, inspect, and export commands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.shared.python.tour_baselines.baseline_package import (
    BaselineIdentity,
    BaselinePackage,
    DynamicFeasibilityStatus,
    KinematicAccuracyStatus,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverConvergenceStatus,
    StatusBundle,
    export_baseline_package,
)
from src.shared.python.tour_baselines.discovery import (
    BaselineDetail,
    BaselineDiscoveryService,
    BaselineFilter,
    BaselineNotFoundError,
    BaselineSummary,
    IncompatiblePresetError,
    MissingDependencyError,
    PRESET_MANIFEST_KEY,
    SafeModelPreset,
    export_to_ledger_rows,
    main,
)
from src.shared.python.tour_baselines.fit_metrics import (
    MarkerMetricSummary,
    PhaseMetricSummary,
    PhysicalFitMetrics,
)
from src.shared.python.tour_baselines.models import (
    BackendType,
    FitMode,
    ModelTopology,
)
from tests.unit.tour_baselines.test_qualification import _make_test_package

# ---------------------------------------------------------------------------
# 1. SafeModelPreset & Compatibility Contracts
# ---------------------------------------------------------------------------


def test_preset_creation_from_package() -> None:
    """SafeModelPreset accurately extracts state, geometry, inertia, and controls."""
    package = _make_test_package(
        model_id="driven_double_pendulum",
        topology=ModelTopology.PLANAR_DRIVEN_PENDULUM,
        capture="driver",
        horizon="G1",
    )
    preset = SafeModelPreset.from_package(
        package,
        is_nominated=True,
        dependencies={"casadi": ">=3.6.0"},
    )

    assert preset.model_id == "driven_double_pendulum"
    assert preset.topology == ModelTopology.PLANAR_DRIVEN_PENDULUM
    assert preset.club == "driver"
    assert preset.horizon == "G1"
    assert preset.is_nominated_baseline is True
    assert preset.is_qualified is True
    assert preset.q0.shape == (2,)
    assert preset.v0.shape == (2,)
    assert "mass" in preset.inertia
    assert "arm_length" in preset.geometry
    assert preset.dependencies == {"casadi": ">=3.6.0"}
    assert len(preset.package_digest) == 64


def test_preset_compatibility_check_passes_on_matching() -> None:
    """Compatibility check succeeds when topology and available dependencies match."""
    package = _make_test_package()
    preset = SafeModelPreset.from_package(
        package,
        dependencies={"casadi": ">=3.6.0"},
    )
    # Should not raise
    preset.verify_compatibility(
        target_model_id="driven_double_pendulum",
        target_topology=ModelTopology.PLANAR_DRIVEN_PENDULUM,
        available_dependencies={"casadi": "3.6.7"},
    )


def test_preset_compatibility_check_fails_on_topology_mismatch() -> None:
    """Compatibility check fails closed when model topology differs."""
    package = _make_test_package(topology=ModelTopology.PLANAR_DRIVEN_PENDULUM)
    preset = SafeModelPreset.from_package(package)

    with pytest.raises(IncompatiblePresetError, match="Topology mismatch"):
        preset.verify_compatibility(target_topology=ModelTopology.FULL_BODY_MULTIBODY)


def test_preset_compatibility_check_fails_on_missing_dependency() -> None:
    """Compatibility check fails closed when a required simulator dependency is absent."""
    package = _make_test_package()
    preset = SafeModelPreset.from_package(
        package,
        dependencies={"pinocchio": ">=2.6.0", "drake": ">=1.22.0"},
    )

    with pytest.raises(
        MissingDependencyError, match="Missing required dependency: drake"
    ):
        preset.verify_compatibility(
            target_topology=ModelTopology.PLANAR_DRIVEN_PENDULUM,
            available_dependencies={"pinocchio": "2.6.5"},
        )


def test_preset_safe_clone_into_session(tmp_path: Path) -> None:
    """Cloning preset creates an independent session copy without modifying original."""
    package = _make_test_package()
    preset = SafeModelPreset.from_package(package, is_nominated=True)

    session_dir = tmp_path / "user_session"
    cloned = preset.clone_into_session(session_dir, new_preset_id="user_custom_preset")

    assert cloned.preset_id == "user_custom_preset"
    assert (
        cloned.is_nominated_baseline is False
    )  # Cloned user copy is not the official nominated baseline
    assert cloned.baseline_id == preset.baseline_id
    assert (session_dir / "user_custom_preset.json").is_file()


# ---------------------------------------------------------------------------
# 2. Discovery Service & Filtering
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_baseline_dir(tmp_path: Path) -> Path:
    """Populate a temporary directory with qualified, unqualified, driver and iron packages."""
    root = tmp_path / "baselines"
    root.mkdir()

    # 1. Qualified nominated driver baseline
    p1 = _make_test_package(
        model_id="driven_double_pendulum",
        capture="driver",
        horizon="G1",
    )
    export_baseline_package(p1, root / "ddp_driver_qualified.npz")

    # 2. Qualified iron baseline
    p2 = _make_test_package(
        model_id="driven_double_pendulum",
        capture="iron",
        horizon="G1",
    )
    export_baseline_package(p2, root / "ddp_iron_qualified.npz")

    # 3. Unqualified / unverified candidate package
    p3 = _make_test_package(
        model_id="driven_double_pendulum",
        capture="driver",
        horizon="G1",
    )
    st3 = StatusBundle(
        solver_convergence=SolverConvergenceStatus.CONVERGED,
        kinematic_accuracy=KinematicAccuracyStatus.WITHIN_TOLERANCE,
        dynamic_feasibility=DynamicFeasibilityStatus.PHYSICALLY_FEASIBLE,
        scientific_qualification=ScientificQualificationStatus.UNVERIFIED,
        product_promotion=ProductPromotionStatus.EXPLORATORY,
        has_native_replay=True,
    )
    object.__setattr__(p3, "statuses", st3)
    export_baseline_package(p3, root / "ddp_driver_unverified.npz")

    return root


def test_discovery_service_scans_clean_directory(populated_baseline_dir: Path) -> None:
    """Discovery service finds and indexes packages without machine-specific hardcoding."""
    service = BaselineDiscoveryService(search_roots=[populated_baseline_dir])
    summaries = service.discover()
    assert len(summaries) == 3
    ids = {s.baseline_id for s in summaries}
    assert len(ids) == 3


def test_discovery_service_filters_by_club_and_qualification(
    populated_baseline_dir: Path,
) -> None:
    """Discovery filtering by club and qualification status behaves correctly."""
    service = BaselineDiscoveryService(search_roots=[populated_baseline_dir])

    # Filter: driver only
    driver_only = service.discover(BaselineFilter(club="driver"))
    assert len(driver_only) == 2

    # Filter: iron only
    iron_only = service.discover(BaselineFilter(club="iron"))
    assert len(iron_only) == 1
    assert iron_only[0].capture == "iron"

    # Filter: only qualified
    qualified_only = service.discover(BaselineFilter(only_qualified=True))
    assert len(qualified_only) == 2
    assert all(s.is_qualified for s in qualified_only)


def test_unverified_package_never_auto_selected_as_default(tmp_path: Path) -> None:
    """get_default_preset fails closed when only unverified packages exist."""
    root = tmp_path / "only_unverified"
    root.mkdir()

    p = _make_test_package()
    st = StatusBundle(
        solver_convergence=SolverConvergenceStatus.CONVERGED,
        kinematic_accuracy=KinematicAccuracyStatus.WITHIN_TOLERANCE,
        dynamic_feasibility=DynamicFeasibilityStatus.PHYSICALLY_FEASIBLE,
        scientific_qualification=ScientificQualificationStatus.UNVERIFIED,
        product_promotion=ProductPromotionStatus.EXPLORATORY,
        has_native_replay=True,
    )
    object.__setattr__(p, "statuses", st)
    export_baseline_package(p, root / "unverified_pkg.npz")

    service = BaselineDiscoveryService(search_roots=[root])
    with pytest.raises(
        IncompatiblePresetError, match="No qualified baseline preset available"
    ):
        service.get_default_preset(
            "driven_double_pendulum", club="driver", horizon="G1"
        )


def test_nominated_baseline_distinguished_from_lowest_loss(
    populated_baseline_dir: Path,
) -> None:
    """Service distinguishes nominated baseline from exploratory or unverified packages."""
    service = BaselineDiscoveryService(search_roots=[populated_baseline_dir])
    service.set_nominated_baseline("driven_double_pendulum", "ddp_driver_qualified")

    nominated = service.discover(BaselineFilter(only_nominated=True))
    assert len(nominated) == 1
    assert nominated[0].is_nominated is True


# ---------------------------------------------------------------------------
# 3. Portable Archive Import/Export
# ---------------------------------------------------------------------------


def test_portable_export_and_clean_machine_import(tmp_path: Path) -> None:
    """Exporting to archive and importing in fresh location verifies integrity."""
    package = _make_test_package()
    preset = SafeModelPreset.from_package(package, is_nominated=True)

    archive_path = tmp_path / "portable_preset.npz"
    service = BaselineDiscoveryService(search_roots=[tmp_path])
    exported = service.export_preset_package(preset, archive_path)
    assert exported.is_file()

    # Import in clean machine directory
    import_dir = tmp_path / "clean_import"
    imported = service.import_preset_package(archive_path, target_dir=import_dir)
    assert imported.model_id == preset.model_id
    assert imported.package_digest == preset.package_digest
    assert np.allclose(imported.q0, preset.q0)
    assert np.allclose(imported.v0, preset.v0)


def test_tampered_archive_import_fails_closed(tmp_path: Path) -> None:
    """Corrupted portable archive raises integrity error on import."""
    package = _make_test_package()
    preset = SafeModelPreset.from_package(package)

    archive_path = tmp_path / "tampered.npz"
    service = BaselineDiscoveryService(search_roots=[tmp_path])
    service.export_preset_package(preset, archive_path)

    # Tamper with file
    content = bytearray(archive_path.read_bytes())
    content[100] ^= 0xFF
    archive_path.write_bytes(content)

    with pytest.raises((ValueError, OSError)):
        service.import_preset_package(archive_path, target_dir=tmp_path / "corrupt")


def test_tampered_manifest_with_valid_arrays_fails_closed(tmp_path: Path) -> None:
    """TB-10 bot review #10795: Tampering with manifest array values while keeping array checksums intact fails closed."""
    package = _make_test_package()
    preset = SafeModelPreset.from_package(package)

    archive_path = tmp_path / "tampered_manifest.npz"
    service = BaselineDiscoveryService(search_roots=[tmp_path])
    service.export_preset_package(preset, archive_path)

    # Read archive, tamper with manifest q0 values while leaving data['q0'] and checksum intact
    with np.load(archive_path, allow_pickle=False) as data:
        manifest_str = str(data[PRESET_MANIFEST_KEY])
        manifest = json.loads(manifest_str)
        manifest["q0"] = [999.0, 999.0]
        tampered_manifest_json = json.dumps(manifest)
        q0 = np.array(data["q0"])
        v0 = np.array(data["v0"])

    np.savez(
        archive_path,
        **{
            PRESET_MANIFEST_KEY: np.array(tampered_manifest_json),
            "q0": q0,
            "v0": v0,
        },
    )

    with pytest.raises(
        ValueError,
        match="Manifest embedded array 'q0' does not match verified array member",
    ):
        service.import_preset_package(
            archive_path, target_dir=tmp_path / "tampered_dir"
        )


def test_rebuilding_index_preserves_identities(populated_baseline_dir: Path) -> None:
    """Rebuilding index yields deterministic identities and manifests."""
    service = BaselineDiscoveryService(search_roots=[populated_baseline_dir])
    run1 = service.discover()
    run2 = service.rebuild_index()

    assert len(run1) == len(run2)
    assert [s.baseline_id for s in run1] == [s.baseline_id for s in run2]
    assert [s.package_digest for s in run1] == [s.package_digest for s in run2]


def test_ledger_row_conversion(populated_baseline_dir: Path) -> None:
    """Exporting to ledger rows conforms to the shared result index schema."""
    service = BaselineDiscoveryService(search_roots=[populated_baseline_dir])
    summaries = service.discover()
    rows = export_to_ledger_rows(summaries)
    assert len(rows) == len(summaries)
    for r in rows:
        assert "receipt_path" in r
        assert "sha256" in r
        assert "engine" in r
        assert "metrics" in r
        assert "acceptance" in r


# ---------------------------------------------------------------------------
# 4. Headless CLI Execution
# ---------------------------------------------------------------------------


def test_cli_list_and_inspect(
    populated_baseline_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI list and inspect commands output valid JSON with expected metadata."""
    # List command
    exit_code = main(
        ["list", "--root", str(populated_baseline_dir), "--format", "json"]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    listed = json.loads(captured.out)
    assert isinstance(listed, list)
    assert len(listed) == 3

    # Inspect command
    b_id = listed[0]["baseline_id"]
    exit_code2 = main(
        ["inspect", b_id, "--root", str(populated_baseline_dir), "--format", "json"]
    )
    assert exit_code2 == 0
    captured2 = capsys.readouterr()
    detail = json.loads(captured2.out)
    assert detail["baseline_id"] == b_id
    assert "metrics" in detail
