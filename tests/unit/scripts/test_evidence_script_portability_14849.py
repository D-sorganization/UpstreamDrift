"""Portability and redaction tests for evidence scripts and docs (#14849)."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from docs.development.full_body_models.evidence.fb3_drake import verify_drake_full_body
from docs.development.full_body_models.evidence.fb3_pinocchio import (
    verify_pinocchio_full_body,
)
from docs.development.full_body_models.evidence.fb4_calibration import (
    verify_full_body_calibration,
)
from docs.development.full_body_models.evidence.fb6_parity import (
    verify_cross_engine_parity,
)
from docs.development.full_body_models.evidence.setup_parity import verify_setup_parity

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT = REPO_ROOT / "docs/development/full_body_models/evidence"

EVIDENCE_SCRIPTS = (
    "fb3_drake/verify_drake_full_body.py",
    "fb3_pinocchio/verify_pinocchio_full_body.py",
    "fb4_calibration/verify_full_body_calibration.py",
    "fb6_parity/verify_cross_engine_parity.py",
    "setup_parity/verify_setup_parity.py",
    "grip_kinetics/render_grip_clip.py",
    "grip_kinetics/run_full_swing_grip_kinetics.py",
    "grip_kinetics/run_grip_kinetics.py",
    "grip_kinetics/run_grip_parity.py",
)

GRIP_KINETICS_CONSTANTS = {
    "grip_kinetics/render_grip_clip.py": "OUT_DIR",
    "grip_kinetics/run_full_swing_grip_kinetics.py": "DEFAULT_OUT",
    "grip_kinetics/run_grip_kinetics.py": "DEFAULT_OUT",
    "grip_kinetics/run_grip_parity.py": "PLOTS",
}

SHELL_CHECK_DOCS = (
    "docs/development/matched_swing_program/MS31_PINOCCHIO_CROCODDYL_TURNOVER.md",
    "docs/development/matched_swing_program/AGENT_CONTINUATION_PROMPT.md",
    "docs/development/feedback_controls/F09E-NATIVE-MARKER-FK.md",
    "docs/development/full_body_models/NEXT_AGENT_PROMPT.md",
    "docs/development/full_body_models/VISUALS_HANDOFF.md",
    "docs/development/impact_acoustics_program.md",
)

PLACEHOLDER_TOKENS = ("<owner-home>", "<host>")
HOME_LITERAL_RE = re.compile(
    r"(?:[A-Za-z]:/Users/|/mnt/c/Users/|/home/)(?!runner\b|ubuntu\b|owner\b)[A-Za-z0-9._-]+"
)
FENCE_RE = re.compile(r"```(?:bash|sh|powershell)\n(.*?)```", re.DOTALL)


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                ids.add(id(node.body[0].value))
    return ids


@pytest.mark.parametrize("rel_path", EVIDENCE_SCRIPTS)
def test_evidence_scripts_contain_no_placeholder_literals(rel_path: str) -> None:
    """Runnable Python scripts must not embed redaction placeholder literals."""
    source = (EVIDENCE_ROOT / rel_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    doc_ids = _docstring_node_ids(tree)
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in doc_ids:
                continue
            if any(token in node.value for token in PLACEHOLDER_TOKENS):
                offenders.append((node.lineno, node.value))
        elif isinstance(node, ast.Constant) and isinstance(node.value, bytes):
            decoded = node.value.decode("utf-8", errors="replace")
            if any(token in decoded for token in PLACEHOLDER_TOKENS):
                offenders.append((node.lineno, decoded))
    assert not offenders, f"{rel_path} embeds placeholder literals: {offenders}"


@pytest.mark.parametrize(("rel_path", "const_name"), GRIP_KINETICS_CONSTANTS.items())
def test_grip_kinetics_default_paths_use_path_home(
    rel_path: str, const_name: str
) -> None:
    """Grip kinetics default output paths must derive from Path.home()."""
    source = (EVIDENCE_ROOT / rel_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned_exprs: list[ast.expr] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == const_name:
                    assigned_exprs.append(node.value)
    assert len(assigned_exprs) == 1, f"{rel_path} missing top-level {const_name}"
    calls = [
        ast.unparse(n)
        for n in ast.walk(assigned_exprs[0])
        if isinstance(n, ast.Call) and "Path.home" in ast.unparse(n)
    ]
    assert calls, (
        f"{rel_path} {const_name} should use Path.home(), got {ast.unparse(assigned_exprs[0])}"
    )


def test_remote_runners_require_ud_remote_host_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote verification runners must fail fast when UD_REMOTE_HOST is unset."""
    monkeypatch.delenv("UD_REMOTE_HOST", raising=False)
    monkeypatch.delenv("UD_REMOTE_HOME", raising=False)

    with pytest.raises(RuntimeError, match="UD_REMOTE_HOST"):
        verify_drake_full_body.run_remote_host()

    with pytest.raises(RuntimeError, match="UD_REMOTE_HOST"):
        verify_pinocchio_full_body.run_remote_host()

    with pytest.raises(RuntimeError, match="UD_REMOTE_HOST"):
        verify_full_body_calibration.run_remote_engine("mujoco", 20, 3)

    with pytest.raises(RuntimeError, match="UD_REMOTE_HOST"):
        verify_cross_engine_parity.run_remote_engine_replay("mujoco", EVIDENCE_ROOT, 10)

    with pytest.raises(RuntimeError, match="UD_REMOTE_HOST"):
        verify_setup_parity.run_remote("drake")


def test_remote_runners_use_configured_env_without_hardcoded_hosts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Remote verification runners must dispatch to configured host and remote home."""
    monkeypatch.setenv("UD_REMOTE_HOST", "example-host")
    monkeypatch.setenv("UD_WSL_DISTRO", "Example-Distro")
    monkeypatch.setenv("UD_REMOTE_HOME", "/srv/remote-runner")

    recorded_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> MagicMock:
        recorded_cmds.append(list(cmd))
        res = MagicMock()
        res.returncode = 0
        res.stderr = b""
        if "verify_setup_parity.py" in str(_kwargs.get("input", b"")):
            res.stdout = b'{"Pelvis": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}'
        else:
            res.stdout = b'{"status": "PASSED", "gates": {}}'
        return res

    monkeypatch.setattr(verify_drake_full_body.subprocess, "run", fake_run)
    receipt = verify_drake_full_body.run_remote_host()
    assert receipt["status"] == "PASSED"
    assert recorded_cmds
    assert all(cmd[:4] == ["ssh", "example-host", "wsl", "-d"] for cmd in recorded_cmds)
    assert all(cmd[4] == "Example-Distro" for cmd in recorded_cmds)

    recorded_cmds.clear()
    monkeypatch.setattr(verify_pinocchio_full_body.subprocess, "run", fake_run)
    pin_receipt = verify_pinocchio_full_body.run_remote_host()
    assert pin_receipt["status"] == "PASSED"
    assert all(
        cmd[:5] == ["ssh", "example-host", "wsl", "-d", "Example-Distro"]
        for cmd in recorded_cmds
    )

    recorded_cmds.clear()
    monkeypatch.setattr(verify_setup_parity.subprocess, "run", fake_run)
    monkeypatch.setattr(
        verify_setup_parity,
        "SYNC",
        (
            "docs/development/full_body_models/evidence/setup_parity/verify_setup_parity.py",
        ),
    )
    setup_poses = verify_setup_parity.run_remote("drake")
    assert "Pelvis" in setup_poses
    assert all(
        cmd[:5] == ["ssh", "example-host", "wsl", "-d", "Example-Distro"]
        for cmd in recorded_cmds
    )
    assert tmp_path.exists()


@pytest.mark.parametrize("rel_path", SHELL_CHECK_DOCS)
def test_docs_have_no_unredacted_home_paths_or_shell_placeholders(
    rel_path: str,
) -> None:
    """Modified docs must not leak user home paths or break fenced shell commands."""
    text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    leaks = HOME_LITERAL_RE.findall(text)
    assert not leaks, f"{rel_path} still contains home path literals: {leaks}"
    for block in FENCE_RE.findall(text):
        found = [tok for tok in PLACEHOLDER_TOKENS if tok in block]
        assert not found, f"{rel_path} fenced shell block embeds placeholder {found}"
