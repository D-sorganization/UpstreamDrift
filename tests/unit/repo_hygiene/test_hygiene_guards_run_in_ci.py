"""Guard the guards: CI jobs that run the hygiene suite must materialise vendor/ud-tools.

The repo-hygiene guards in this package (``test_no_shadow_of_tools_shared`` and
``test_vendor_submodule_clean``) are only meaningful when
``vendor/ud-tools`` is present. UpstreamDrift deliberately does **not** pass
``submodules: recursive`` to ``actions/checkout`` in the main CI workflows —
three of the four submodules are large model repositories
(``opensim-models``, ``myo_sim``, ``human-gazebo``) and cloning them on every
job would be wasteful. Instead ``.github/actions/fetch-pinned-tools``
materialises just ``vendor/ud-tools`` at the pinned gitlink revision.

That indirection is fine, but it is invisible: if someone drops the
``fetch-pinned-tools`` step from a job that runs the unit suite, the hygiene
guards silently degrade to a skip and nobody notices. This test makes that
regression a hard CI failure.

A job is considered to run the hygiene suite when it invokes pytest such that
``tests/unit/repo_hygiene`` is collected (no path arguments, so pyproject
``testpaths`` applies, or an explicit path covering the package) **and** its
``-m`` marker expression selects this package's markers.
"""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"
_HYGIENE_PACKAGE = "tests/unit/repo_hygiene"
_VENDOR_ACTION = "fetch-pinned-tools"

# Markers carried by every test in this package. A workflow ``-m`` expression
# is "selecting" when it evaluates true for a test with exactly these markers.
_HYGIENE_MARKERS = frozenset({"unit", "headless_safe"})

_LINE_CONTINUATION = re.compile(r"\\\s*\n\s*")
_PIP_COMMAND = re.compile(r"pip\s+(?:install|uninstall)")
_MARKER_OPTION = re.compile(r"-m\s+(\"[^\"]*\"|'[^']*'|\S+)")
_SHELL_SEPARATOR = re.compile(r"(?:\|\||&&|[;|])")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PYTHON_EXECUTABLE = re.compile(r"^python[0-9.]*$")

# Commands that wrap another command; the real command name follows them.
_COMMAND_WRAPPERS = frozenset(
    {
        "xvfb-run",
        "run_with_heartbeat",
        "time",
        "env",
        "nice",
        "sudo",
        "timeout",
        "stdbuf",
    }
)
_TEMPLATE = re.compile(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}")


def _iter_workflows() -> list[Path]:
    return sorted(_WORKFLOW_DIR.glob("*.yml")) + sorted(_WORKFLOW_DIR.glob("*.yaml"))


def _matrix_values(job: dict[str, Any], key: str) -> list[str]:
    """Return every statically declared value of ``matrix.<key>`` for *job*."""
    matrix = ((job.get("strategy") or {}).get("matrix")) or {}
    if not isinstance(matrix, dict):
        return []
    values: list[str] = []
    direct = matrix.get(key)
    if isinstance(direct, list):
        values.extend(str(v) for v in direct)
    include = matrix.get("include")
    if isinstance(include, list):
        for entry in include:
            if isinstance(entry, dict) and key in entry:
                values.append(str(entry[key]))
    return values


def _expand_templates(text: str, job: dict[str, Any]) -> list[str]:
    """Expand ``${{ matrix.x }}`` into every statically known value.

    Returns a list of concrete candidate strings. An unresolvable template
    yields the original text so the caller stays conservative.
    """
    match = _TEMPLATE.search(text)
    if match is None:
        return [text]
    values = _matrix_values(job, match.group(1))
    if not values:
        return [text]
    expanded: list[str] = []
    for value in values:
        expanded.extend(
            _expand_templates(text[: match.start()] + value + text[match.end() :], job)
        )
    return expanded


def _evaluate_marker_node(node: ast.expr) -> bool:
    """Evaluate a parsed marker expression against :data:`_HYGIENE_MARKERS`.

    Only the node types pytest's own ``-m`` grammar produces are accepted:
    names, ``and``/``or``, and ``not``. Anything else raises, which the caller
    treats as "selecting" so the guard errs toward failing loudly.
    """
    if isinstance(node, ast.Name):
        return node.id in _HYGIENE_MARKERS
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _evaluate_marker_node(node.operand)
    if isinstance(node, ast.BoolOp):
        results = [_evaluate_marker_node(value) for value in node.values]
        return all(results) if isinstance(node.op, ast.And) else any(results)
    raise ValueError(f"unsupported marker expression node: {type(node).__name__}")


def marker_expression_selects_hygiene(expression: str) -> bool:
    """Return whether a pytest ``-m`` expression selects this package's tests.

    Each bare identifier is treated as "is this one of the markers every test
    in this package carries", then combined with pytest's ``and``/``or``/``not``
    semantics.
    """
    if not expression.strip():
        return True
    try:
        parsed = ast.parse(expression, mode="eval")
        return _evaluate_marker_node(parsed.body)
    except (SyntaxError, ValueError, TypeError):
        # An expression we cannot understand is treated as selecting, so the
        # guard fails loudly rather than silently waving the job through.
        return True


def _tokenize(segment: str) -> list[str]:
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def invokes_pytest(segment: str) -> bool:
    """Return whether *segment* actually executes pytest as a command.

    Only pytest in *command position* counts. The word appearing inside a
    prose argument (for example a Jules ``--session`` prompt that says
    "generate pytest unit tests") or inside a grep pattern is not an
    invocation, and flagging those produced false positives.
    """
    tokens = _tokenize(segment)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            token in _COMMAND_WRAPPERS
            or token.startswith("-")
            or _ENV_ASSIGNMENT.match(token)
        ):
            index += 1
            continue
        if token == "pytest" or token.endswith("/pytest"):
            return True
        if _PYTHON_EXECUTABLE.match(token):
            return tokens[index + 1 : index + 3] == ["-m", "pytest"]
        if index > 0 and tokens[index - 1] in _COMMAND_WRAPPERS:
            # A wrapper's non-flag operand: keep scanning for the real command.
            index += 1
            continue
        return False
    return False


def _pytest_commands(run: str) -> list[str]:
    joined = _LINE_CONTINUATION.sub(" ", run)
    commands: list[str] = []
    for raw_line in joined.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if _PIP_COMMAND.search(line):
            continue
        if "--collect-only" in line:
            # Collection does not execute test bodies, so the guards cannot
            # skip or fail there.
            continue
        for segment in _SHELL_SEPARATOR.split(line):
            if invokes_pytest(segment):
                commands.append(line)
                break
    return commands


def _collects_hygiene_package(command: str) -> bool:
    """Return whether *command*'s path arguments cover the hygiene package."""
    path_args = [
        token.strip("'\"")
        for token in command.split()
        if token.strip("'\"").startswith(("tests/", "tests/", "src/"))
        or token.strip("'\"") == "tests"
    ]
    if not path_args:
        # No explicit paths: pyproject `testpaths` applies, which includes
        # `tests`, so the hygiene package is collected.
        return True
    return any(
        _HYGIENE_PACKAGE.startswith(arg.rstrip("/")) or arg.startswith(_HYGIENE_PACKAGE)
        for arg in path_args
    )


def _runs_hygiene_suite(command: str, job: dict[str, Any]) -> bool:
    if not _collects_hygiene_package(command):
        return False
    marker_match = _MARKER_OPTION.search(command)
    if marker_match is None:
        return True
    raw_expression = marker_match.group(1).strip("'\"")
    return any(
        marker_expression_selects_hygiene(candidate)
        for candidate in _expand_templates(raw_expression, job)
    )


def _job_materialises_vendor(job: dict[str, Any]) -> bool:
    for step in job.get("steps") or []:
        if not isinstance(step, dict):
            continue
        uses = str(step.get("uses", ""))
        if _VENDOR_ACTION in uses:
            return True
        if uses.startswith("actions/checkout"):
            submodules = str((step.get("with") or {}).get("submodules", "")).lower()
            if submodules in {"true", "recursive"}:
                return True
    return False


def _jobs_running_hygiene_without_vendor() -> list[str]:
    offenders: list[str] = []
    for workflow in _iter_workflows():
        try:
            document = yaml.safe_load(workflow.read_text(encoding="utf-8")) or {}
        except (
            yaml.YAMLError
        ) as exc:  # pragma: no cover - lint-workflow-files gates this
            raise AssertionError(f"Could not parse {workflow}: {exc}") from exc
        for job_name, job in (document.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                for command in _pytest_commands(str(step.get("run") or "")):
                    if not _runs_hygiene_suite(command, job):
                        continue
                    if _job_materialises_vendor(job):
                        continue
                    offenders.append(
                        f"{workflow.name}: job {job_name!r}, "
                        f"step {str(step.get('name', '<unnamed>'))!r}"
                    )
    return sorted(set(offenders))


# ── the guard ────────────────────────────────────────────────────────────────


def test_every_ci_job_running_hygiene_tests_materialises_vendor() -> None:
    """The hygiene guards must never run without ``vendor/ud-tools`` present.

    Without the vendored Tools tree both guards used to ``pytest.skip`` and
    passed vacuously on every CI run. They now fail closed in CI, so a job
    missing the vendor step is a hard error rather than a silent pass — this
    test reports it against the workflow definition instead of waiting for a
    confusing runtime failure.
    """
    offenders = _jobs_running_hygiene_without_vendor()

    assert not offenders, (
        "These CI jobs run tests/unit/repo_hygiene but never materialise "
        "vendor/ud-tools. Add `- uses: ./.github/actions/fetch-pinned-tools` "
        "after the checkout step (preferred — it fetches only the pinned Tools "
        "revision), or set `submodules: recursive` on the checkout if the "
        "large model submodules are genuinely needed.\n\n"
        "Offending jobs:\n  " + "\n  ".join(offenders)
    )


def test_unit_test_gate_materialises_vendor_before_running_pytest() -> None:
    """Ordering matters: the vendor step must precede the pytest step.

    ``unit-test-gate`` is the job that actually executes the hygiene guards on
    every PR (it selects ``-m unit`` over the default testpaths).
    """
    document = yaml.safe_load(
        (_WORKFLOW_DIR / "ci-standard.yml").read_text(encoding="utf-8")
    )
    job = document["jobs"]["unit-test-gate"]
    steps = job["steps"]

    vendor_index = next(
        (i for i, s in enumerate(steps) if _VENDOR_ACTION in str(s.get("uses", ""))),
        None,
    )
    pytest_index = next(
        (i for i, s in enumerate(steps) if _pytest_commands(str(s.get("run") or ""))),
        None,
    )

    assert vendor_index is not None, (
        "ci-standard.yml unit-test-gate must materialise vendor/ud-tools via "
        "./.github/actions/fetch-pinned-tools."
    )
    assert pytest_index is not None, (
        "ci-standard.yml unit-test-gate no longer runs pytest — this guard "
        "needs to be repointed at whichever job runs the unit suite."
    )
    assert vendor_index < pytest_index, (
        "ci-standard.yml unit-test-gate materialises vendor/ud-tools at step "
        f"{vendor_index} but runs pytest at step {pytest_index}. The vendor "
        "tree must exist before the hygiene guards execute."
    )


# ── cover the detection logic itself ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("expression", "selected"),
    [
        ("", True),
        ("unit", True),
        ("unit and not slow", True),
        ("not slow", True),
        ("headless_safe", True),
        ("requires_drake", False),
        ("requires_mujoco and not slow", False),
        ("live_simulation and requires_pinocchio", False),
        ("not unit", False),
    ],
)
def test_marker_expression_selection(expression: str, selected: bool) -> None:
    """The marker evaluator must mirror pytest's ``-m`` semantics."""
    assert marker_expression_selects_hygiene(expression) is selected


def test_explicit_unrelated_path_is_not_treated_as_running_hygiene() -> None:
    """A job scoped to another package must not be flagged."""
    assert not _collects_hygiene_package("pytest tests/unit/robotics -v")


def test_default_testpaths_invocation_is_treated_as_running_hygiene() -> None:
    """No path arguments means pyproject testpaths, which include this package."""
    assert _collects_hygiene_package("python -m pytest -m unit --timeout=60")


def test_parent_path_argument_is_treated_as_running_hygiene() -> None:
    """A parent directory argument still collects the hygiene package."""
    assert _collects_hygiene_package("pytest tests/unit")


@pytest.mark.parametrize(
    ("segment", "invokes"),
    [
        ("pytest -v", True),
        ("python -m pytest -v", True),
        ("python3 -m pytest tests/unit", True),
        ("xvfb-run --auto-servernum python -m pytest -m unit", True),
        ('run_with_heartbeat "core pytest lane" xvfb-run python -m pytest', True),
        # Regression: the word "pytest" inside a prose argument is not a run.
        (
            'jules remote new --session "Generate comprehensive pytest unit tests"',
            False,
        ),
        ('grep -q "FAILED\\|AssertionError\\|pytest"', False),
        ("python -m ruff check", False),
        ("echo pytest", False),
    ],
)
def test_pytest_invocation_detection(segment: str, invokes: bool) -> None:
    """Only pytest in command position counts as an invocation."""
    assert invokes_pytest(segment) is invokes
