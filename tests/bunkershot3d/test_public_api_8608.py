"""The BunkerShot3D public API surface (issue #8608, W1).

Wave 1 shipped ``sand``, ``study``, ``provenance`` and a new ``geometry`` API,
none of which ``bunkershot3d/__init__.py`` re-exported, so consumers reached
into submodules. The fix is *not* to flatten every name: ``study.rng`` and
``provenance.rng`` both export ``SeedRecord``, and flattening ~200 names would
make the collision silent. The package therefore re-exports the subpackages
plus a curated set of entry-point names, and ``__all__`` is authoritative.
"""

from __future__ import annotations

import importlib

import pytest

import bunkershot3d

pytestmark = pytest.mark.unit

#: Subpackages a consumer is expected to reach for by name.
EXPECTED_SUBPACKAGES = (
    "backends",
    "calibration",
    "config",
    "domain",
    "exceptions",
    "geometry",
    "io",
    "kinematics",
    "postproc",
    "provenance",
    "sand",
    "study",
    "units",
)

#: Names that existed before #8608 and must keep working.
LEGACY_NAMES = (
    "AngleOfReposeExperiment",
    "BackendNotImplementedError",
    "BunkerShotResultReader",
    "BunkerShotResultWriter",
    "CalibrationOptimizer",
    "ChronoDriver",
    "ClubheadGenerator",
    "CoSimulator",
    "CoupledDoublePendulum",
    "DrainedShearCellExperiment",
    "LiggghtsDriver",
    "MPMDriver",
    "SwingTrajectory",
    "WrenchTrace",
    "__version__",
    "generate_reference_trajectory",
)

#: The value objects and entry points #8608 promotes to the top level.
NEW_NAMES = (
    "BunkerShot3DError",
    "BunkerShotConfig",
    "ContactMaterial",
    "DomainBox",
    "GrainPopulation",
    "RunManifest",
    "SandState",
    "SolverSettings",
    "SwingCondition",
    "TrajectorySource",
    "Validity",
    "WedgeGeometry",
)


class TestAllIsAuthoritative:
    def test_all_is_sorted_and_unique(self) -> None:
        names = list(bunkershot3d.__all__)
        assert names == sorted(names), "__all__ must be sorted"
        assert len(names) == len(set(names)), "__all__ has duplicates"

    def test_every_exported_name_resolves(self) -> None:
        missing = [n for n in bunkershot3d.__all__ if not hasattr(bunkershot3d, n)]
        assert not missing, missing

    def test_star_import_matches_all(self) -> None:
        namespace: dict[str, object] = {}
        exec("from bunkershot3d import *", namespace)  # noqa: S102
        exported = set(namespace) - {"__builtins__"}
        assert exported == set(bunkershot3d.__all__)

    def test_no_public_attribute_is_omitted_from_all(self) -> None:
        """``__all__`` is the whole public surface, not a subset of it."""
        public = {
            name
            for name in vars(bunkershot3d)
            if not name.startswith("_") or name == "__version__"
        }
        undeclared = sorted(public - set(bunkershot3d.__all__))
        assert not undeclared, (
            f"these public attributes are not in __all__: {undeclared}. "
            "__all__ must be authoritative."
        )


class TestSubpackagesAreReachable:
    @pytest.mark.parametrize("name", EXPECTED_SUBPACKAGES)
    def test_subpackage_is_re_exported(self, name: str) -> None:
        assert name in bunkershot3d.__all__
        assert getattr(bunkershot3d, name) is importlib.import_module(
            f"bunkershot3d.{name}"
        )


class TestCuratedNames:
    @pytest.mark.parametrize("name", LEGACY_NAMES)
    def test_pre_existing_name_still_exported(self, name: str) -> None:
        assert name in bunkershot3d.__all__

    @pytest.mark.parametrize("name", NEW_NAMES)
    def test_new_name_exported(self, name: str) -> None:
        assert name in bunkershot3d.__all__

    def test_curated_names_are_the_same_objects_as_in_their_subpackages(self) -> None:
        assert bunkershot3d.SandState is bunkershot3d.sand.SandState
        assert bunkershot3d.WedgeGeometry is bunkershot3d.geometry.WedgeGeometry
        assert bunkershot3d.RunManifest is bunkershot3d.provenance.RunManifest
        assert bunkershot3d.DomainBox is bunkershot3d.domain.DomainBox

    def test_the_colliding_name_is_not_flattened(self) -> None:
        """``SeedRecord`` exists in both ``study`` and ``provenance``; promoting
        either to the top level would silently pick a winner."""
        assert "SeedRecord" not in bunkershot3d.__all__
        assert bunkershot3d.study.SeedRecord is not bunkershot3d.provenance.SeedRecord
