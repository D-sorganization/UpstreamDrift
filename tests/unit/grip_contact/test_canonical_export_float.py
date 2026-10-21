"""Deterministic float canonicalization for grip-contact codegen exports (#13334)."""

from __future__ import annotations

import json

import pytest

from src.shared.python.grip_contact.matlab_kernel import (
    build_kernel_params_payload,
    export_matlab_kernel_bundle,
)
from src.shared.python.grip_contact.net_wrench_parity import (
    canonical_export_float,
    canonicalize_export_tree,
)
from src.shared.python.grip_contact.simscape_grip_contact import export_scenarios_json

pytestmark = pytest.mark.unit


def test_canonical_export_float_json_round_trip() -> None:
    """Codegen floats use the same decimal text as json.dumps (cross-platform)."""
    raw = 0.3826656624324893
    canon = canonical_export_float(raw)
    assert canon == json.loads(json.dumps(round(raw, 11)))
    assert json.dumps(canon) == json.dumps(round(raw, 11))


def test_canonical_export_float_unifies_python_matrix_damping() -> None:
    """Python 3.11/3.12 dissipation codegen samples collapse to one export float."""
    py311_sample = 0.38266566243249167
    py312_sample = 0.3826656624324893
    assert canonical_export_float(py311_sample) == canonical_export_float(py312_sample)


def test_canonicalize_export_tree_recurses() -> None:
    tree = canonicalize_export_tree(
        {"law": {"damping_s_m": 0.3826656624324893}, "rows": [[1.0, 2.0000000000000004]]}
    )
    assert tree["law"]["damping_s_m"] == canonical_export_float(0.3826656624324893)
    assert tree["rows"][0][1] == canonical_export_float(2.0000000000000004)


def test_committed_matlab_kernel_and_scenarios_exports_are_fresh() -> None:
    """Committed generated artifacts match the Python source of truth."""
    assert export_matlab_kernel_bundle(check=True) == 0
    assert export_scenarios_json(check=True) == 0


def test_kernel_params_damping_matches_canonical_law() -> None:
    from src.shared.python.grip_contact.net_wrench_parity import canonicalize_export_tree

    export_params = canonicalize_export_tree(build_kernel_params_payload())
    damping = export_params["law"]["damping_s_m"]
    assert damping == canonical_export_float(damping)
    assert isinstance(damping, float)
