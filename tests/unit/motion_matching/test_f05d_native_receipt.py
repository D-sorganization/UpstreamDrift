"""Fail-closed evidence parsing for the native manifold candidate."""

from __future__ import annotations

import importlib
from html import escape
import json
from pathlib import Path

from defusedxml import ElementTree

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _receipt_module() -> object:
    pytest.importorskip("crocoddyl")
    pytest.importorskip("mujoco")
    return importlib.import_module("scripts.f05d_native_manifold_receipt")


@pytest.mark.parametrize("failure_tag", ("failure", "error", "skipped"))
def test_receipt_rejects_incomplete_native_suite(
    tmp_path: Path, failure_tag: str
) -> None:
    module = _receipt_module()
    junit = tmp_path / "incomplete.xml"
    junit.write_text(
        f'<testsuite><testcase name="native"><{failure_tag}/></testcase></testsuite>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="failed or skipped"):
        module._cases(junit)
    junit.write_text(
        '<testsuite><testcase name="native"/></testsuite>', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="inventory is incomplete"):
        module._cases(junit)


def test_receipt_rejects_false_accepted_step_count() -> None:
    module = _receipt_module()
    values = {
        "common_bootstrap_s": "0.1",
        "execution_order": json.dumps(["box", "scipy"]),
        "source_model_sha256": "a" * 64,
        "loaded_model_sha256": "f" * 64,
        "initial_state_sha256": "b" * 64,
        "policy_sha256": "c" * 64,
        "time_grid_sha256": "d" * 64,
        "state_schema_sha256": "1" * 64,
        "input_channel_schema_sha256": "2" * 64,
    }
    for method in ("box", "scipy"):
        values.update({f"{method}_{suffix}": "0.1" for suffix in module._NUMERIC})
        values[f"{method}_accepted_steps"] = "4"
        values[f"{method}_statuses"] = json.dumps(["fallback_timeout"] * 4)
        values[f"{method}_solver_id"] = method
        values[f"{method}_applied_sha256"] = "e" * 64
        values[f"{method}_qpos_final"] = json.dumps([0.0] * 9)
    properties = "".join(
        f'<property name="{escape(name, quote=True)}" '
        f'value="{escape(value, quote=True)}"/>'
        for name, value in values.items()
    )
    testcase = ElementTree.fromstring(
        f"<testcase><properties>{properties}</properties></testcase>"
    )
    with pytest.raises(ValueError, match="accepted-step count"):
        module._trial(testcase, "0.4")
