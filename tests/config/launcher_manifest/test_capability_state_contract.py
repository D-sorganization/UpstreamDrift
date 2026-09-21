"""TDD Acceptance Suite for Capability State Contract (ORG-02, Issue #10511).

Contracts tested:
1. Disentangled orthogonal fields: maturity, availability, qualification.
2. Installed engine with no qualified receipt must never serialize release-ready.
3. Distinct actionable states: missing provider, stale pin, CLI-only target,
   planned shell, and native-only browser target.
4. Validation: missing remediation command/action on unavailable surface fails DbC.
5. Consistent display names: Simscape/Matlab Models and Simulator/Golf Simulation Suite
   resolve identically across native and shared views.
6. Lazy probing and cache keys tied to runtime pin/identity.
7. Backward-compatible legacy status preservation for old clients.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch
import pytest

from src.config.capability_state import (
    CANONICAL_TILE_DISPLAY_NAMES,
    CapabilityAvailability,
    CapabilityQualification,
    RuntimeProbeCache,
    RuntimeProbeKey,
    SurfaceAvailability,
    adapt_engine_matrix_qualification,
    resolve_canonical_display_name,
)
from src.config.launcher_manifest_loader import (
    LauncherManifest,
    LauncherTile,
)
from src.shared.python.config.model_registry import ModelRegistry

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_YAML = REPO_ROOT / "src" / "config" / "models.yaml"
MANIFEST_JSON = REPO_ROOT / "src" / "config" / "launcher_manifest.json"

pytestmark = pytest.mark.unit


# =============================================================================
# RED Acceptance Case 1: Installed Engine Without Qualified Receipt
# =============================================================================


class TestEngineQualificationReleaseGate:
    """Installed engine with no qualified receipt must never serialize release-ready."""

    def test_installed_engine_without_qualified_receipt_never_serializes_release_ready(
        self,
    ) -> None:
        """An engine that is installed/runtime-available but lacks a qualified receipt

        must have is_qualified=False, qualification.status != 'advertised_and_qualified',
        legacy status != 'ready' / 'engine_ready', and maturity != 'stable'.
        """
        # Audit engine qualification with no receipt via adapter
        qualification = adapt_engine_matrix_qualification(
            engine_name="mujoco",
            is_engine=True,
            receipt=None,
        )

        assert not qualification.is_qualified
        assert qualification.status in ("qualification_failed", "unqualified")
        assert "missing_engine_receipt" in qualification.failure_reasons

        # A tile created with this unqualified state must not serialize release-ready
        tile = LauncherTile(
            id="mujoco",
            name="MuJoCo Humanoid",
            description="MuJoCo physics simulation",
            category="physics_engine",
            type="mujoco",
            path="src/engines/physics_engines/mujoco/python/humanoid_launcher.py",
            logo="mujoco_humanoid.svg",
            status="experimental",  # Never "ready" or "engine_ready" without receipt
            engine_type="mujoco",
            maturity="experimental",
            qualification=qualification,
        )

        serialized = tile.to_dict()
        assert serialized["status"] != "ready"
        assert serialized["status"] != "engine_ready"
        assert serialized["status"] != "release_ready"
        assert serialized["qualification"]["is_qualified"] is False
        assert serialized["qualification"]["status"] != "advertised_and_qualified"
        assert serialized["maturity"] != "stable"

    def test_non_engine_tools_are_exempt_from_scientific_qualification(self) -> None:
        """Non-engine tools need availability without pretending to have scientific qualification."""
        qualification = adapt_engine_matrix_qualification(
            engine_name=None,
            is_engine=False,
            receipt=None,
        )

        assert qualification.status == "exempt"
        assert not qualification.is_qualified
        assert qualification.failure_reasons == ()


# =============================================================================
# RED Acceptance Case 2: Distinct Actionable Unavailable States & Remediation
# =============================================================================


class TestSurfaceAvailabilityStates:
    """Missing provider, stale pin, CLI-only target, planned shell, and

    native-only browser target serialize distinct actionable states.
    """

    def test_missing_remediation_on_unavailable_surface_fails_validation(self) -> None:
        """Missing remediation command or empty reason on unavailable surface must fail validation."""
        with pytest.raises(ValueError, match="remediation"):
            SurfaceAvailability(
                available=False,
                reason="Provider not found",
                remediation="",  # Invalid: must provide actionable remediation
            )

        with pytest.raises(ValueError, match="reason"):
            SurfaceAvailability(
                available=False,
                reason="",  # Invalid: reason must be non-empty
                remediation="Run pip install",
            )

    def test_distinct_actionable_states_for_unavailable_surfaces(self) -> None:
        """Verify the 5 distinct actionable failure states."""
        # 1. Missing provider
        missing_provider = SurfaceAvailability(
            available=False,
            reason="Provider source root missing: external/my_provider",
            remediation="Clone provider repository or configure valid source_root in models.yaml",
        )

        # 2. Stale pin
        stale_pin = SurfaceAvailability(
            available=False,
            reason="Tools vendor authority pin is stale (expected sha_abc, found sha_xyz)",
            remediation="Run 'python -m scripts.sync_vendor_tools' to synchronize the vendor submodule",
        )

        # 3. CLI-only target
        cli_only_desktop = SurfaceAvailability(
            available=False,
            reason="CLI-only utility without a graphical desktop interface",
            remediation="Run from terminal: python src/tools/my_cli.py --help",
        )

        # 4. Planned shell
        planned_shell = SurfaceAvailability(
            available=False,
            reason="Web shell affordance is planned but not yet implemented",
            remediation="Use desktop Qt launcher or track release milestone in issue tracker",
        )

        # 5. Native-only browser target
        native_only_browser = SurfaceAvailability(
            available=False,
            reason="Native-window launch target requires desktop launcher environment; unavailable in browser",
            remediation="Launch from PyQt desktop launcher or run: python src/launchers/my_tool.py",
        )

        states = [
            missing_provider,
            stale_pin,
            cli_only_desktop,
            planned_shell,
            native_only_browser,
        ]

        # All 5 states must have non-empty reasons and remediation commands
        for state in states:
            assert not state.available
            assert state.reason and len(state.reason.strip()) > 10
            assert state.remediation and len(state.remediation.strip()) > 10

        # All reasons and remediation commands must be distinct
        reasons = [s.reason for s in states]
        remediations = [s.remediation for s in states]
        assert len(set(reasons)) == 5
        assert len(set(remediations)) == 5

    def test_capability_availability_surface_query(self) -> None:
        """CapabilityAvailability exposes per-surface query methods."""
        avail = CapabilityAvailability(
            surfaces={
                "desktop": SurfaceAvailability(available=True),
                "web": SurfaceAvailability(
                    available=False,
                    reason="No web affordance",
                    remediation="Use desktop launcher",
                ),
                "api": SurfaceAvailability(available=True),
                "cli": SurfaceAvailability(available=True),
            }
        )
        assert avail.desktop.available
        assert not avail.web.available
        assert avail.web.reason == "No web affordance"
        assert avail.web.remediation == "Use desktop launcher"
        assert avail.api.available
        assert avail.cli.available


# =============================================================================
# RED Acceptance Case 3: Display Name Consistency Across Shells
# =============================================================================


class TestDisplayNameAuthority:
    """Conflicting YAML/JSON display names resolve identically across native/shared views."""

    def test_conflicting_display_names_resolve_identically(self) -> None:
        """Simscape/Matlab Models and Simulator/Golf Simulation Suite must resolve

        identically across native registry models and manifest tiles from one authority.
        """
        registry = ModelRegistry(config_path=MODELS_YAML)
        manifest = LauncherManifest.load(MANIFEST_JSON)

        # 1. Matlab Models / Simscape authority resolution
        native_matlab = registry.get_model("matlab_suite")
        manifest_matlab = manifest.get_tile("matlab_suite")

        assert native_matlab is not None
        assert manifest_matlab is not None
        assert native_matlab.name == manifest_matlab.name
        assert manifest_matlab.name == "Matlab Models"

        # 2. Simulator / Golf Simulation Suite authority resolution
        native_sim = registry.get_model("golf_simulation_suite")
        manifest_sim = manifest.get_tile("golf_simulation_suite")

        assert native_sim is not None
        assert manifest_sim is not None
        assert native_sim.name == manifest_sim.name
        assert manifest_sim.name == "Golf Simulation Suite"

    def test_canonical_display_name_resolver(self) -> None:
        """Canonical display name resolver returns authoritative name."""
        assert resolve_canonical_display_name("matlab_suite") == "Matlab Models"
        assert (
            resolve_canonical_display_name("golf_simulation_suite")
            == "Golf Simulation Suite"
        )
        assert (
            resolve_canonical_display_name("unknown_tile", default="Default Name")
            == "Default Name"
        )


# =============================================================================
# RED Acceptance Case 4: Lazy Probing & Cache Key Pin Identity
# =============================================================================


class TestLazyProbingAndCaching:
    """Uncacheable or eager provider import fails probe contract;

    probe results cache against runtime pins.
    """

    def test_uncacheable_target_fails_probe_contract(self) -> None:
        """Probe key creation without a runtime pin/identity must fail validation."""
        with pytest.raises(ValueError, match="pin_or_identity"):
            RuntimeProbeKey(
                target_id="my_engine",
                pin_or_identity="",  # Invalid: cannot cache without a pin/identity
                env_fingerprint=sys.version,
            )

        with pytest.raises(ValueError, match="target_id"):
            RuntimeProbeKey(
                target_id="",
                pin_or_identity="v1.0.0",
                env_fingerprint=sys.version,
            )

    def test_runtime_probe_cache_memoization(self) -> None:
        """Runtime probe cache correctly stores and invalidates entries."""
        cache = RuntimeProbeCache()
        key1 = RuntimeProbeKey(
            target_id="drake",
            pin_or_identity="pydrake==1.25.0",
            env_fingerprint="py311",
        )
        key2 = RuntimeProbeKey(
            target_id="drake",
            pin_or_identity="pydrake==1.26.0",
            env_fingerprint="py311",
        )

        cache.set(key1, True)
        assert cache.get(key1) is True
        assert cache.get(key2) is None  # Cache missed due to changed pin

    def test_schema_only_manifest_loading_performs_no_heavy_engine_imports(
        self,
    ) -> None:
        """Catalog loading must never eagerly import heavy engine modules (e.g. mujoco, pydrake, pinocchio, opensim)."""
        import subprocess

        # Run clean process to verify no heavy engine is imported during LauncherManifest.load()
        cmd = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from src.config.launcher_manifest_loader import LauncherManifest; "
                "manifest = LauncherManifest.load(); "
                "assert 'mujoco' not in sys.modules, 'mujoco was eagerly imported'; "
                "assert 'pydrake' not in sys.modules, 'pydrake was eagerly imported'; "
                "assert 'pinocchio' not in sys.modules, 'pinocchio was eagerly imported'; "
                "assert 'opensim' not in sys.modules, 'opensim was eagerly imported'; "
            ),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"Subprocess failed:\n{proc.stderr}\n{proc.stdout}"


# =============================================================================
# GREEN Acceptance Case 5: Legacy Status & Orthogonal Serialization
# =============================================================================


class TestLegacyStatusCompatibility:
    """Old clients receive compatible status fields; dynamic providers and

    hidden aliases remain covered.
    """

    def test_tile_serializes_both_legacy_status_and_orthogonal_fields(self) -> None:
        """LauncherTile.to_dict() must include legacy 'status' AND orthogonal fields."""
        manifest = LauncherManifest.load(MANIFEST_JSON)
        tile = manifest.get_tile("model_explorer")
        assert tile is not None

        data = tile.to_dict()
        # Legacy field preserved
        assert "status" in data
        assert isinstance(data["status"], str)

        # Orthogonal fields present
        assert "maturity" in data
        assert "availability" in data
        assert "qualification" in data
        assert isinstance(data["availability"], dict)
        assert "desktop" in data["availability"]
        assert "web" in data["availability"]

    def test_hidden_aliases_have_valid_orthogonal_state(self) -> None:
        """Hidden legacy alias tiles preserve valid orthogonal state."""
        manifest = LauncherManifest.load(MANIFEST_JSON)
        alias_tile = manifest.get_tile("starting_pose_matcher")
        assert alias_tile is not None
        assert alias_tile.hidden

        data = alias_tile.to_dict()
        assert "status" in data
        assert "maturity" in data
        assert "availability" in data
        assert "qualification" in data
