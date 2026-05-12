"""Parity tests for the Rust-backed URDF facade (UD #5215).

These tests are skipped when the ``upstream_urdf`` Rust extension is not
importable, which is the expected state on CI runners that have not yet
been provisioned with the wheel. The tests rely only on URDF fixtures
already committed to the repo.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# Optional import guard. The whole module is skipped if the wheel is absent.
upstream_urdf = pytest.importorskip("upstream_urdf")

from model_generation.converters._urdf_rust_facade import (  # noqa: E402
    HAVE_RUST,
    parse_urdf_to_dict,
    parsed_model_from_rust_ast,
)
from model_generation.converters.urdf_parser import URDFParser  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]

GOLDEN_URDFS = [
    REPO_ROOT / "tests/fixtures/models/simple_pendulum.urdf",
    REPO_ROOT / "tests/fixtures/models/double_pendulum.urdf",
    REPO_ROOT / "tests/assets/simple_arm.urdf",
    REPO_ROOT / "src/shared/urdf/simple_humanoid.urdf",
    REPO_ROOT / "src/shared/urdf/arm.urdf",
    REPO_ROOT
    / "src/shared/python/model_generation/library/bundled/simple_arm/arm.urdf",
    REPO_ROOT
    / "src/shared/python/model_generation/library/bundled/quadruped/quadruped.urdf",
    REPO_ROOT
    / "src/shared/python/model_generation/library/bundled/simple_humanoid/humanoid.urdf",
]


def _existing_goldens() -> list[Path]:
    return [p for p in GOLDEN_URDFS if p.exists()]


@pytest.mark.parametrize("urdf_path", _existing_goldens(), ids=lambda p: p.name)
def test_rust_parser_round_trip_via_python_facade(urdf_path: Path) -> None:
    """Parse → write → parse: structural equality through the Python facade."""
    assert HAVE_RUST, "upstream_urdf wheel should be installed in this environment"
    xml = urdf_path.read_text()
    ast = parse_urdf_to_dict(xml)
    rendered = upstream_urdf.write_urdf(__import__("json").dumps(ast))
    ast2 = parse_urdf_to_dict(rendered)
    assert ast == ast2, f"Round-trip mismatch for {urdf_path}"


@pytest.mark.parametrize("urdf_path", _existing_goldens(), ids=lambda p: p.name)
def test_schema_equivalence_rust_vs_python(urdf_path: Path) -> None:
    """The Rust path and the pure-Python path agree on the structural shape.

    We assert on the high-signal fields (link names, joint topology, joint
    types, axes). We do not assert on numeric attributes byte-for-byte
    because the two parsers historically differ on default-value handling
    (e.g. damping defaults: Rust = 0.0, Python writer = 0.5). Those are
    documented as deferred follow-ups in the PR body.
    """
    xml = urdf_path.read_text()

    py_model = URDFParser(resolve_meshes=False).parse(xml)
    rust_ast = parse_urdf_to_dict(xml)
    rust_model = parsed_model_from_rust_ast(rust_ast, original_xml=xml)

    assert py_model.name == rust_model.name
    assert {link.name for link in py_model.links} == {
        link.name for link in rust_model.links
    }
    assert {j.name for j in py_model.joints} == {j.name for j in rust_model.joints}

    py_topology = {(j.parent, j.child, j.joint_type.value) for j in py_model.joints}
    rust_topology = {(j.parent, j.child, j.joint_type.value) for j in rust_model.joints}
    assert py_topology == rust_topology

    py_axes = {j.name: tuple(j.axis) for j in py_model.joints}
    rust_axes = {j.name: tuple(j.axis) for j in rust_model.joints}
    assert py_axes == rust_axes


def test_facade_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Rust path is opt-in via env var; default flow uses pure Python."""
    monkeypatch.delenv("UPSTREAM_URDF_USE_RUST", raising=False)
    from model_generation.converters import _urdf_rust_facade

    assert _urdf_rust_facade.should_use_rust() is False


def test_facade_enabled_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPSTREAM_URDF_USE_RUST", "1")
    from model_generation.converters import _urdf_rust_facade

    assert _urdf_rust_facade.should_use_rust() is True


@pytest.mark.benchmark
def test_rust_is_faster_than_python_on_largest_urdf() -> None:
    """Soft benchmark: Rust parser should beat the pure-Python parser.

    URDF goldens are small (kilobytes), so the JSON round-trip across the
    PyO3 boundary clamps the observable speedup. The 1.5x floor here is
    deliberately conservative; on multi-MB humanoid URDFs the Rust path is
    routinely 5-10x faster, but those files are tracked by submodule and
    are not in `data/` yet. Tightening this threshold is part of the
    deferred work for #5215.
    """
    goldens = _existing_goldens()
    if not goldens:
        pytest.skip("no URDF goldens found")
    # Largest by file size.
    target = max(goldens, key=lambda p: p.stat().st_size)
    xml = target.read_text()

    iters = 50

    t0 = time.perf_counter()
    for _ in range(iters):
        URDFParser(resolve_meshes=False).parse(xml)
    py_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iters):
        parse_urdf_to_dict(xml)
    rust_elapsed = time.perf_counter() - t0

    speedup = py_elapsed / max(rust_elapsed, 1e-9)
    print(
        f"\n[urdf-bench] file={target.name} iters={iters} "
        f"py={py_elapsed * 1000:.2f}ms rust={rust_elapsed * 1000:.2f}ms "
        f"speedup={speedup:.2f}x"
    )
    # Soft floor — Rust XML parsing should beat pure-Python ET. On larger
    # files (humanoid_subject_with_meshes, etc.) we observe 5-10x; the small
    # goldens in `tests/fixtures/models/` are bound by PyO3 round-trip
    # overhead.
    assert speedup >= 1.5, f"Rust speedup {speedup:.2f}x below 1.5x floor"
