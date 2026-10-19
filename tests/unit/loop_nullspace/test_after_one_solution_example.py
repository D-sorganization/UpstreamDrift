"""NSE-UD-4a (#13464): the after-one-solution example and its mapping document.

The example script ``docs/research/loop_nullspace/after_one_solution_example.py``
walks the post-fit workflow on the generic planar fixture, the sympy-backed
solution-manifold circle and a three-muscle segment fibre.  These tests run it
and check every number it reports against an independent statement of the
property, so the document claims nothing the tests do not verify.

Tolerances (derived):

* ``INVARIANT_TOL = 1e-8``: the normalised annihilation residual of a selector in
  ``range(A^T)`` is roundoff of an SVD null space, a few ``1e-15``; ``1e-8`` is the
  tolerance the NSE-UD-1 tests use.
* Sampled values are compared with LP bounds and with the absolute bound to
  ``1e-9``, the LP feasibility tolerance of the envelope solver.
* The circle ``F(u) = |u|^2 - 1`` has residual ``h^2`` at the tangent predictor
  and ``F^2 / (4 |u|^2) <= F^2 / 4`` after one Newton update, so the corrected
  residual is far below ``CHART_TOL = 1e-10``.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("highspy")
pytest.importorskip("clarabel")

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "docs" / "research" / "loop_nullspace"
SCRIPT = RESEARCH / "after_one_solution_example.py"
DOC = RESEARCH / "after_one_solution.md"
README = RESEARCH / "README.md"
ADR = ROOT / "docs" / "adr" / "0054-closed-loop-null-space-exploration.md"

INVARIANT_TOL = 1e-8
LP_TOL = 1e-9
CHART_TOL = 1e-10


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("after_one_solution_example", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def result() -> dict:
    return _load_script().run_example()


# ------------------------------------------------------------- the script
def test_script_runs_and_returns_every_section(result: dict) -> None:
    assert set(result) == {
        "fibre",
        "identifiability",
        "envelope",
        "attribution",
        "chart",
        "muscle",
    }


def test_fibre_dimension_accounting(result: dict) -> None:
    fibre = result["fibre"]
    assert fibre["nullity"] == fibre["dim_x"] - fibre["rank_A"]
    assert fibre["nullity"] >= 1


def test_identifiability_labels_and_absolute_bound(result: dict) -> None:
    ident = result["identifiability"]
    assert ident["invariant_status"] == "numerically_invariant"
    assert ident["invariant_residual"] <= INVARIANT_TOL
    assert ident["family_status"] == "family_dependent"
    assert INVARIANT_TOL < ident["family_residual"] <= 1.0
    # |s.(x - x0)| <= ||N^T s|| R with R an enclosing radius.
    assert ident["max_sampled_deviation"] <= ident["deviation_bound"] + LP_TOL
    assert ident["width_bound"] == pytest.approx(2.0 * ident["deviation_bound"])
    assert ident["invariant_record"] == "invariant"
    assert ident["family_record"] == "family_dependent"


def test_envelope_contains_every_sampled_value(result: dict) -> None:
    env = result["envelope"]
    assert env["status"] == "bounded"
    assert env["lower"] <= env["sampled_min"] + LP_TOL
    assert env["sampled_max"] <= env["upper"] + LP_TOL
    assert env["lower"] < env["upper"]


def test_attribution_total_is_invariant_but_a_source_is_not(result: dict) -> None:
    att = result["attribution"]
    assert att["total_status"] == "numerically_invariant"
    assert "family_dependent" in att["source_statuses"]
    assert att["partition_sum_error"] <= LP_TOL


def test_chart_step_stays_on_the_manifold_and_rank_drop_raises(result: dict) -> None:
    chart = result["chart"]
    assert chart["rank"] == 1
    assert chart["residual_after_step"] <= CHART_TOL
    assert chart["moved_distance"] > 0.0
    assert chart["crossing_raises"] is True


def test_muscle_fibre_hull_is_smaller_than_the_kernel(result: dict) -> None:
    muscle = result["muscle"]
    assert muscle["nullity"] == 2
    assert muscle["hull_dimension"] == 1
    assert muscle["samples_vary"] is True
    assert muscle["level_subfibre_dimension"] == 0


def test_script_is_deterministic() -> None:
    module = _load_script()
    assert module.run_example() == module.run_example()


# ---------------------------------------------------------- the document
_SECTION_COUNT = 11
_NAME = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)`")


def _package_names() -> set[str]:
    import importlib
    import pkgutil

    import src.shared.python.loop_nullspace as pkg

    names: set[str] = set()
    for info in pkgutil.iter_modules(pkg.__path__):
        names.add(info.name)
        try:
            module = importlib.import_module(f"{pkg.__name__}.{info.name}")
        except ImportError:  # optional engine dependency absent
            continue
        names.update(n for n in dir(module) if not n.startswith("_"))
        for member in vars(module).values():
            if isinstance(member, type):  # methods and properties of classes
                names.update(n for n in dir(member) if not n.startswith("_"))
    return names


def test_document_has_eleven_numbered_sections() -> None:
    text = DOC.read_text(encoding="utf-8")
    rows = re.findall(r"^\| (\d+) \|", text, flags=re.MULTILINE)
    assert [int(r) for r in rows] == list(range(1, _SECTION_COUNT + 1))


def test_document_states_the_binding_caveats() -> None:
    text = DOC.read_text(encoding="utf-8")
    for needle in (
        "declared polygonal",
        "certified outer bounds",
        "inner bounds only",
        "no exact second-order-cone",
        "numerical invariance",
        "exact invariance",
        "enclosing radius",
        "local only",
    ):
        assert needle in text.lower().replace("local-only", "local only"), needle


def test_every_named_api_exists_in_the_package() -> None:
    text = DOC.read_text(encoding="utf-8")
    known = _package_names()
    declared = set(re.findall(r"<!-- api: ([A-Za-z_][A-Za-z0-9_.]*) -->", text))
    assert len(declared) >= 25
    missing = sorted(n for n in declared if n.split(".")[-1] not in known)
    assert not missing, missing


def test_document_mentions_only_existing_example_paths() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "after_one_solution_example.py" in text
    assert SCRIPT.is_file()
    assert "test_after_one_solution_example.py" in text


def test_readme_and_adr_link_the_document() -> None:
    assert "after_one_solution.md" in README.read_text(encoding="utf-8")
    assert "after_one_solution.md" in ADR.read_text(encoding="utf-8")
