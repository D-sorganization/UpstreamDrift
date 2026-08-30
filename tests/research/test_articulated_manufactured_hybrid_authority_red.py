"""RED contracts for hybrid manufactured-solution authority (#9236)."""

from __future__ import annotations

import copy
from collections.abc import Callable
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts.research.proximal_distal_energy import (
    run_articulated_manufactured_solution as runner,
)
from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    require_robotics_pinocchio,
)

pytestmark = [pytest.mark.scientific]

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
COMMITTED = DATA / "articulated_manufactured_solution.json"
AUTHORITY_LOCK = runner.AUTHORITY_LOCK
AUTHORITY_PROFILE = "articulated-manufactured-authority-py311-v1"
ROLLING_PROFILE = "articulated-manufactured-rolling-native-v1"


def _serializer() -> Callable[[dict[str, Any]], bytes]:
    serializer = getattr(runner, "canonical_record_bytes", None)
    assert callable(serializer), "one canonical finite-value serializer is required"
    return serializer


def _native_stack_available() -> bool:
    try:
        import mujoco  # noqa: F401 - availability probe
        import pinocchio as pin

        require_robotics_pinocchio(pin)
    except (ImportError, RuntimeError):
        return False
    return True


requires_native_stack = pytest.mark.skipif(
    not _native_stack_available(),
    reason="the authority and rolling CI lanes install both native engines",
)


def _write_in_process(output: Path, profile: str) -> subprocess.CompletedProcess[str]:
    command = (
        "from pathlib import Path; import sys; "
        "from scripts.research.proximal_distal_energy."
        "run_articulated_manufactured_solution import write_record; "
        "write_record(Path(sys.argv[1]), profile=sys.argv[2])"
    )
    return subprocess.run(
        [sys.executable, "-c", command, str(output), profile],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


def _require_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stderr or result.stdout


def _load_committed() -> dict[str, Any]:
    return json.loads(COMMITTED.read_text(encoding="utf-8"))


def _requirement_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line and not line[0].isspace() and not line.startswith("#"):
            if current:
                blocks.append("\n".join(current))
            current = [line]
        elif current and line.strip():
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_canonical_serializer_rejects_nested_nonfinite_values(nonfinite: float) -> None:
    """NaN and infinities must fail before evidence bytes exist."""

    with pytest.raises(ValueError, match="finite"):
        _serializer()({"outer": {"values": [1.0, nonfinite]}})


def test_canonical_serializer_is_order_independent_and_newline_terminated() -> None:
    """One serializer must canonicalize mapping order and byte framing."""

    left = {"z": {"b": 2.0, "a": 1.0}, "a": [3.0, 4.0]}
    right = {"a": [3.0, 4.0], "z": {"a": 1.0, "b": 2.0}}
    left_bytes = _serializer()(left)

    assert left_bytes == _serializer()(right)
    assert left_bytes.endswith(b"\n")
    assert json.loads(left_bytes) == left


def test_record_writer_delegates_to_the_canonical_serializer() -> None:
    """The writer cannot maintain a second JSON serialization policy."""

    source = Path(runner.__file__).read_text(encoding="utf-8")
    writer_source = source[
        source.index("def write_record(") : source.index("def main(")
    ]
    assert "canonical_record_bytes" in writer_source
    assert "json.dumps" not in writer_source


def test_authority_lock_is_exact_and_hash_complete() -> None:
    """Every authority distribution must be exact and wheel-hash locked."""

    text = AUTHORITY_LOCK.read_text(encoding="utf-8")
    blocks = _requirement_blocks(text)
    names = {block.split("==", maxsplit=1)[0].lower() for block in blocks}

    assert {"mujoco", "numpy", "pin", "scipy"} <= names
    assert blocks
    for block in blocks:
        assert "==" in block.splitlines()[0]
        assert "--hash=sha256:" in block


def test_committed_record_pins_authority_profile_runtime_lock_and_sources() -> None:
    """Committed bytes must identify the exact authoritative environment."""

    record = _load_committed()
    profile = record["execution_profile"]
    lock = profile["dependency_lock"]
    runtime = profile["runtime_versions"]

    assert profile["id"] == AUTHORITY_PROFILE
    assert profile["publication_authority"] == "authoritative"
    assert profile["python_minor"] == "3.11"
    assert lock["path"] == AUTHORITY_LOCK.relative_to(ROOT).as_posix()
    assert lock["sha256"] == hashlib.sha256(AUTHORITY_LOCK.read_bytes()).hexdigest()
    assert {"python", "numpy", "mujoco", "pinocchio"} == set(runtime)
    assert record["source_sha256"]
    for relative_path, expected in record["source_sha256"].items():
        assert (
            hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected
        )


def test_semantic_comparison_ignores_provenance_but_fails_closed_on_gates() -> None:
    """Rolling compatibility may vary by runtime but never by registered gates."""

    authority = _load_committed()
    rolling = copy.deepcopy(authority)
    rolling["execution_profile"] = {
        "id": ROLLING_PROFILE,
        "publication_authority": "non_authoritative_compatibility_only",
        "publication_eligible": False,
        "runtime_versions": {"python": "different-runtime"},
    }
    compare = getattr(runner, "compare_semantic_evidence", None)
    assert callable(compare), "rolling semantic comparison must be explicit"
    assert compare(authority, rolling)["all_registered_gates_pass"] is True

    rolling["free_body"]["all_gates_pass"] = False
    with pytest.raises(ValueError, match="gate|semantic"):
        compare(authority, rolling)


@requires_native_stack
def test_same_environment_two_process_authority_bytes_match_committed(
    tmp_path: Path,
) -> None:
    """Two fresh authority processes must equal each other and committed bytes."""

    first = tmp_path / "authority-first.json"
    second = tmp_path / "authority-second.json"
    _require_success(_write_in_process(first, "authority"))
    _require_success(_write_in_process(second, "authority"))

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == COMMITTED.read_bytes()


@requires_native_stack
def test_rolling_native_output_has_actual_provenance_and_no_authority(
    tmp_path: Path,
) -> None:
    """Rolling native output is temporary compatibility evidence only."""

    committed_before = COMMITTED.read_bytes()
    output = tmp_path / "rolling-native.json"
    _require_success(_write_in_process(output, "rolling"))
    rolling = json.loads(output.read_text(encoding="utf-8"))
    profile = rolling["execution_profile"]

    assert profile["id"] == ROLLING_PROFILE
    assert profile["publication_authority"] == "non_authoritative_compatibility_only"
    assert profile["publication_eligible"] is False
    assert profile["runtime_versions"]["python"] == sys.version.split()[0]
    for distribution in ("numpy", "mujoco", "pin"):
        assert profile["runtime_versions"][distribution] == importlib.metadata.version(
            distribution
        )
    assert runner.compare_semantic_evidence(_load_committed(), rolling)[
        "all_registered_gates_pass"
    ]
    assert COMMITTED.read_bytes() == committed_before


def test_corrupted_engine_killswitch_remains_in_native_contract() -> None:
    """The hybrid split cannot remove the registered corrupted-engine falsifier."""

    source = (
        ROOT / "tests/research/test_articulated_manufactured_solution.py"
    ).read_text(encoding="utf-8")
    assert (
        "test_manufactured_solution_killswitch_detects_corrupt_native_inverse" in source
    )
    assert "closed_form_check_passed is False" in source
