"""Tests for src.engines.model_inventory (Issue #10376 — MS-102).

TDD coverage for the engine/model inventory and smoke qualification harness.
Native SDK steps are exercised when the engine is installed; otherwise the
harness records an honest fail/skip with remediation (never a fake pass).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.engines.model_inventory import (
    ClubKind,
    InventoryError,
    ModelClass,
    ModelPackage,
    PackageStatus,
    QualificationOutcome,
    QualificationReceipt,
    QualificationStep,
    TARGET_ENGINES,
    EngineModelInventory,
    qualify_package,
    sha256_file,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
INVENTORY_PATH = REPO_ROOT / "src" / "config" / "engine_model_inventory.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def inventory() -> EngineModelInventory:
    return EngineModelInventory.load(repo_root=REPO_ROOT)


# ---------------------------------------------------------------------------
# Enum / contract shape
# ---------------------------------------------------------------------------


class TestPackageContracts:
    def test_target_engines_are_the_six_flagship_backends(self) -> None:
        expected = frozenset(
            {"mujoco", "drake", "pinocchio", "opensim", "myosuite", "simscape"}
        )
        assert expected == TARGET_ENGINES

    def test_club_kinds_cover_driver_and_iron(self) -> None:
        assert {c.value for c in ClubKind} == {"driver", "iron"}

    def test_model_classes_separate_native_and_shared_document(self) -> None:
        assert ModelClass.SHARED_DOCUMENT.value == "shared_document"
        assert ModelClass.NATIVE_ANATOMY.value == "native_anatomy"
        assert ModelClass.LEGACY_DEMO.value == "legacy_demo"

    def test_package_rejects_empty_id(self) -> None:
        with pytest.raises((InventoryError, ValueError)):
            ModelPackage(
                id="",
                engine="mujoco",
                club=ClubKind.DRIVER,
                model_family="full_body_anthro",
                model_class=ModelClass.SHARED_DOCUMENT,
                intended_use="matched_swing_smoke",
                status=PackageStatus.READY,
                source_spec="docs/x.json",
                source_sha256="a" * 64,
            )


# ---------------------------------------------------------------------------
# Inventory load + authority reconciliation
# ---------------------------------------------------------------------------


class TestInventoryLoad:
    def test_inventory_file_exists(self) -> None:
        assert INVENTORY_PATH.is_file()

    def test_load_returns_packages(self, inventory: EngineModelInventory) -> None:
        assert len(inventory.packages) >= 12  # 6 engines × driver/iron

    def test_every_target_engine_has_driver_and_iron(
        self, inventory: EngineModelInventory
    ) -> None:
        for engine in TARGET_ENGINES:
            clubs = {
                p.club for p in inventory.packages if p.engine == engine and p.flagship
            }
            assert ClubKind.DRIVER in clubs, f"{engine} missing driver flagship"
            assert ClubKind.IRON in clubs, f"{engine} missing iron flagship"

    def test_no_competing_engine_catalog(self, inventory: EngineModelInventory) -> None:
        """Inventory engines must come from existing authorities."""
        asserted = set(inventory.authority_engines)
        for package in inventory.packages:
            assert package.engine in asserted or package.engine in TARGET_ENGINES

    def test_jaxsim_is_reconciled_not_silently_dropped(
        self, inventory: EngineModelInventory
    ) -> None:
        assert "jaxsim" in inventory.reconciled
        assert inventory.reconciled["jaxsim"]["status"] in {
            "experimental",
            "retired",
            "specialized",
        }

    def test_models_yaml_physics_tiles_are_covered(
        self, inventory: EngineModelInventory
    ) -> None:
        missing = inventory.uncovered_launcher_tiles()
        assert missing == [], f"Advertised physics tiles lack inventory: {missing}"

    def test_ready_flagship_packages_have_model_hashes(
        self, inventory: EngineModelInventory
    ) -> None:
        for package in inventory.flagship_packages():
            if package.status != PackageStatus.READY:
                continue
            assert package.identity_hash(), f"{package.id} missing identity hash"
            assert len(package.identity_hash()) == 64

    def test_repair_packages_name_an_issue(
        self, inventory: EngineModelInventory
    ) -> None:
        for package in inventory.packages:
            if package.status != PackageStatus.REPAIR:
                continue
            assert package.repair_task is not None, package.id
            assert package.repair_task.issue > 0, package.id
            assert package.repair_task.title.strip(), package.id

    def test_simscape_packages_require_r2025b(
        self, inventory: EngineModelInventory
    ) -> None:
        for package in inventory.packages:
            if package.engine != "simscape":
                continue
            assert package.matlab_release == "R2025b", package.id

    def test_source_hashes_match_tree(self, inventory: EngineModelInventory) -> None:
        errors = inventory.verify_declared_hashes()
        assert errors == [], errors


# ---------------------------------------------------------------------------
# Hash helper
# ---------------------------------------------------------------------------


class TestSha256:
    def test_hashes_file_contents(self, tmp_path: Path) -> None:
        path = tmp_path / "a.bin"
        path.write_bytes(b"abc")
        digest = sha256_file(path)
        assert digest == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InventoryError, match="not found"):
            sha256_file(tmp_path / "missing.bin")


# ---------------------------------------------------------------------------
# Qualification harness
# ---------------------------------------------------------------------------


class TestQualificationHarness:
    def test_missing_assets_fail_with_remediation(
        self, inventory: EngineModelInventory, tmp_path: Path
    ) -> None:
        package = next(p for p in inventory.packages if p.engine == "opensim")
        broken = ModelPackage(
            id="opensim/driver-broken",
            engine=package.engine,
            club=package.club,
            model_family=package.model_family,
            model_class=package.model_class,
            intended_use=package.intended_use,
            status=PackageStatus.READY,
            source_spec="docs/development/full_body_models/does_not_exist.json",
            source_sha256="0" * 64,
            generated_model=None,
            generated_sha256=None,
            generator=package.generator,
            flagship=True,
            supported_hosts=package.supported_hosts,
            matlab_release=package.matlab_release,
            repair_task=None,
            launcher_tile_ids=package.launcher_tile_ids,
            license_terms=package.license_terms,
            joints=package.joints,
            units=package.units,
            frames=package.frames,
            marker_map=package.marker_map,
            club_config=package.club_config,
            grip_config=package.grip_config,
            contact_config=package.contact_config,
            actuation=package.actuation,
            required_assets=("missing/asset.bin",),
            muscle_tendon=package.muscle_tendon,
            workspace_init=package.workspace_init,
        )
        receipt = qualify_package(broken, repo_root=REPO_ROOT, allow_native=False)
        assert receipt.overall == QualificationOutcome.FAIL
        asset_step = receipt.step(QualificationStep.RESOLVE_ASSETS)
        assert asset_step.outcome == QualificationOutcome.FAIL
        assert asset_step.remediation
        assert "missing" in asset_step.message.lower() or "not found" in (
            asset_step.message.lower()
        )

    def test_structural_pass_for_opensim_driver(
        self, inventory: EngineModelInventory
    ) -> None:
        package = inventory.get("opensim/driver")
        receipt = qualify_package(package, repo_root=REPO_ROOT, allow_native=False)
        assert receipt.package_id == "opensim/driver"
        assert receipt.step(QualificationStep.RESOLVE_ASSETS).outcome == (
            QualificationOutcome.PASS
        )
        assert receipt.step(QualificationStep.HASH_CHECK).outcome == (
            QualificationOutcome.PASS
        )
        # Native steps are not claimed when allow_native=False.
        native = receipt.step(QualificationStep.LOAD_COMPILE)
        assert native.outcome in {
            QualificationOutcome.SKIP,
            QualificationOutcome.FAIL,
        }
        assert native.outcome != QualificationOutcome.PASS

    def test_receipt_is_json_serialisable(
        self, inventory: EngineModelInventory
    ) -> None:
        package = inventory.get("opensim/iron")
        receipt = qualify_package(package, repo_root=REPO_ROOT, allow_native=False)
        payload = receipt.to_dict()
        round_trip = json.loads(json.dumps(payload))
        assert round_trip["package_id"] == "opensim/iron"
        assert "steps" in round_trip
        assert "contract_sha256" in round_trip

    def test_repair_package_does_not_claim_ready_native(
        self, inventory: EngineModelInventory
    ) -> None:
        repairs = [p for p in inventory.packages if p.status == PackageStatus.REPAIR]
        assert repairs, "expected at least one repair package (MyoSuite)"
        receipt = qualify_package(repairs[0], repo_root=REPO_ROOT, allow_native=True)
        assert receipt.overall != QualificationOutcome.PASS

    def test_harness_records_native_pass_when_loader_succeeds(
        self, inventory: EngineModelInventory
    ) -> None:
        package = inventory.get("mujoco/driver")
        fake_loader = MagicMock(return_value={"nq": 44, "mass_kg": 75.0})
        with patch(
            "src.engines.model_inventory._native_load_step",
            fake_loader,
        ):
            receipt = qualify_package(package, repo_root=REPO_ROOT, allow_native=True)
        assert receipt.step(QualificationStep.LOAD_COMPILE).outcome == (
            QualificationOutcome.PASS
        )


class TestInventorySummary:
    def test_summary_json_lists_repair_blockers(
        self, inventory: EngineModelInventory
    ) -> None:
        summary = inventory.summary()
        assert "packages" in summary
        assert "repair_tasks" in summary
        assert isinstance(summary["repair_tasks"], list)
        assert any(t["issue"] == 10344 for t in summary["repair_tasks"])
