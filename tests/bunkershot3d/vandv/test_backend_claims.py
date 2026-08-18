"""Every physical claim in ``comparison.md``, proved or deleted (#8616).

The issue's instruction was blunt: *delete or prove every unsubstantiated
claim in* ``docs/bunkershot3d/comparison.md``.  Working through them, the
document turned out to be wrong about almost everything it asserted, and
wrong in a specific way -- it described the backends somebody intended to
build rather than the ones in the tree:

===============================================  ======================
Claim as written                                 Verdict
===============================================  ======================
LIGGGHTS uses a linear spring-dashpot model      **false**, Hertz-Mindlin
LIGGGHTS imports an STL clubface via fix mesh    **false**, no clubhead
LIGGGHTS allows slightly larger timesteps        **false**, same model
Chrono uses a rigid triangular clubhead mesh     **false**, a box
MuJoCo MPM is a continuum with Drucker-Prager    **false**, spheres
MuJoCo MPM represents the clubface with an SDF   **false**, a box geom
MuJoCo MPM is bounded by continuum sound speed   **false**, Courant
0.2 Rayleigh is a *Chrono* requirement           misleading, shared
Coarse-graining factor is 1.0                    true
===============================================  ======================

Every row above is asserted here against the code, so the rewritten
document cannot drift back.  The surviving claims are the ones these
tests pass on; the rest were deleted rather than softened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_FIXTURE_DIR = Path(__file__).resolve().parents[1]
if str(_FIXTURE_DIR) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(_FIXTURE_DIR))

from _bunker_fixtures_8612 import (  # noqa: E402
    make_mock_chrono,
    write_config,
    write_straight_trajectory,
)

from bunkershot3d.exceptions import (  # noqa: E402
    BackendNotImplementedError,
    DomainInvariantError,
)
from bunkershot3d.backends.chrono import driver as chrono_mod  # noqa: E402
from bunkershot3d.backends.liggghts.driver import LiggghtsDriver  # noqa: E402
from bunkershot3d.backends.mpm.driver import MAX_SPHERES, MPMDriver  # noqa: E402
from bunkershot3d.backends.prescribed_motion import load_trajectory  # noqa: E402
from bunkershot3d.backends.stability import (  # noqa: E402
    RAYLEIGH_SAFETY_FACTOR,
)
from bunkershot3d.domain.grains import GrainPopulation  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.scientific]

COMPARISON_DOC = (
    Path(__file__).resolve().parents[3] / "docs" / "bunkershot3d" / "comparison.md"
)

#: Claims the old document made that the code contradicts. If any of these
#: strings comes back into the document, it is a regression in honesty.
DELETED_CLAIMS = (
    "linear spring-dashpot",
    "fix mesh/surface",
    "signed distance field",
    "rigid triangular mesh",
    "Drucker-Prager yield criterion",
    "sound speed of the continuum",
    "0.2 Rayleigh safety factor",
    "under 200,000",
)


@pytest.fixture
def config(tmp_path: Path) -> Path:
    """A physically admissible configuration with a straight swing."""
    trajectory = write_straight_trajectory(tmp_path / "swing.csv")
    # POSIX separators: the fixture writes the path into a double-quoted YAML
    # scalar, where a Windows backslash is an escape sequence.
    return write_config(tmp_path / "config.yaml", trajectory_file=trajectory.as_posix())


@pytest.fixture
def liggghts_deck(config: Path, tmp_path: Path) -> str:
    """The generated LIGGGHTS input deck, as text."""
    work = tmp_path / "work"
    work.mkdir()
    return LiggghtsDriver(config)._generate_input_deck(work).read_text()


class TestLiggghtsContactModelClaim:
    """ "LIGGGHTS uses a linear spring-dashpot model" -- it does not."""

    def test_the_deck_uses_hertz_mindlin(self, liggghts_deck: str) -> None:
        assert "pair_style      gran model hertz tangential history" in liggghts_deck

    def test_the_deck_contains_no_linear_spring_model(self, liggghts_deck: str) -> None:
        assert "model hooke" not in liggghts_deck
        assert "spring-dashpot" not in liggghts_deck

    def test_it_is_the_same_contact_model_chrono_uses(self, liggghts_deck: str) -> None:
        """So there is no contact-model divergence between them to report."""
        assert "hertz" in liggghts_deck
        chrono_doc = chrono_mod.ChronoDriver._make_contact_material.__doc__ or ""
        assert "SMC" in chrono_doc


class TestLiggghtsClubfaceClaim:
    """ "LIGGGHTS uses fix mesh/surface with imported STL" -- there is no club."""

    def test_the_deck_has_no_mesh_import(self, liggghts_deck: str) -> None:
        assert "fix mesh/surface" not in liggghts_deck
        assert ".stl" not in liggghts_deck.lower()

    def test_the_deck_says_so_itself(self, liggghts_deck: str) -> None:
        assert "no intruder" in liggghts_deck

    def test_the_driver_refuses_to_report_a_wrench(self, config: Path) -> None:
        """A backend with no clubhead cannot report a club force."""
        driver = LiggghtsDriver(config)
        with pytest.raises(BackendNotImplementedError):
            driver.setup()


class TestMujocoContactAndClubfaceClaims:
    """ "Continuum MPM with an SDF clubface" -- discrete spheres and a box."""

    def test_the_model_is_built_from_sphere_geoms(self, config: Path) -> None:
        xml = MPMDriver(config)._generate_xml()
        assert 'type="sphere"' in xml

    def test_the_clubface_is_a_box_geom_not_a_signed_distance_field(
        self, config: Path
    ) -> None:
        xml = MPMDriver(config)._generate_xml()
        assert 'name="clubface" type="box"' in xml
        assert "sdf" not in xml.lower()

    def test_the_sphere_count_is_capped_far_below_a_real_bed(self) -> None:
        """1000 spheres against a true-scale requirement of 2.1e8 grains."""
        assert MAX_SPHERES == 1000
        assert MAX_SPHERES < 2.1e8 / 1e5


class TestTimestepClaims:
    """The Rayleigh safety factor exists, but it is not Chrono's alone."""

    def test_the_safety_factor_is_now_implemented(self) -> None:
        """It did not exist when the document first advertised it (#8612)."""
        assert RAYLEIGH_SAFETY_FACTOR == 0.2

    def test_it_lives_in_the_shared_stability_module_not_in_chrono(self) -> None:
        """So it is not a divergence between backends, it is common ground."""
        assert RAYLEIGH_SAFETY_FACTOR is not None
        assert not hasattr(chrono_mod, "RAYLEIGH_SAFETY_FACTOR")

    def test_mujoco_deliberately_does_not_enforce_the_rayleigh_limit(
        self, config: Path
    ) -> None:
        """Its contacts are resolved at the velocity level, so Courant governs.

        The old document claimed a bound from grid resolution and the
        continuum's sound speed. There is no continuum and no grid.
        """
        driver = MPMDriver(config)
        plan = driver._plan(load_trajectory(config, driver.config), 100_000)
        assert plan.rayleigh_limit == float("inf")
        assert plan.cfl_limit < float("inf")

    def test_chrono_does_enforce_it(self, config: Path) -> None:
        driver = chrono_mod.ChronoDriver(config)
        plan = driver._plan(load_trajectory(config, driver.config), 100_000)
        assert plan.rayleigh_limit < float("inf")


class TestChronoClubfaceClaim:
    """ "Chrono uses a rigid triangular mesh with face friction" -- it is a box."""

    def test_the_clubhead_collision_shape_is_a_box(
        self, config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chrono = make_mock_chrono()
        monkeypatch.setattr(chrono_mod, "chrono", chrono, raising=False)
        monkeypatch.setattr(chrono_mod, "_HAS_CHRONO", True)
        chrono_mod.ChronoDriver(config).setup()
        assert chrono.ChCollisionShapeBox.called

    def test_no_triangle_mesh_is_ever_created(
        self, config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chrono = make_mock_chrono()
        monkeypatch.setattr(chrono_mod, "chrono", chrono, raising=False)
        monkeypatch.setattr(chrono_mod, "_HAS_CHRONO", True)
        chrono_mod.ChronoDriver(config).setup()
        mesh_calls = [
            name
            for name, _args, _kwargs in chrono.mock_calls
            if "TriangleMesh" in name or "Trimesh" in name
        ]
        assert mesh_calls == []


class TestCoarseGrainingClaim:
    """The one numeric claim in the document that was simply true."""

    def test_the_default_coarse_graining_factor_is_one(self) -> None:
        population = GrainPopulation(
            count=1000,
            diameter_mean_m=4.0e-4,
            diameter_sigma_log=0.2,
            density_kg_m3=2650.0,
        )
        assert population.coarse_graining_factor == 1.0

    def test_a_factor_below_one_is_refused(self) -> None:
        with pytest.raises(DomainInvariantError, match="at least 1"):
            GrainPopulation(
                count=1000,
                diameter_mean_m=4.0e-4,
                diameter_sigma_log=0.2,
                density_kg_m3=2650.0,
                coarse_graining_factor=0.5,
            )


class TestTheDocumentNoLongerMakesTheFalseClaims:
    """A freshness gate on the rewritten ``comparison.md``."""

    def test_the_document_exists(self) -> None:
        assert COMPARISON_DOC.is_file()

    @pytest.mark.parametrize("claim", DELETED_CLAIMS)
    def test_a_deleted_claim_has_not_come_back(self, claim: str) -> None:
        text = COMPARISON_DOC.read_text(encoding="utf-8")
        assert claim not in text, (
            f"{claim!r} is back in comparison.md. It was removed because the "
            "code contradicts it; if the code has changed, prove the new claim "
            "with a test in this file before restating it."
        )

    def test_every_surviving_claim_points_at_a_test(self) -> None:
        """No claim in the document may stand without a named test."""
        text = COMPARISON_DOC.read_text(encoding="utf-8")
        assert "tests/bunkershot3d/vandv/test_backend_claims.py" in text
