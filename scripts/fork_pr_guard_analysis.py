"""Condition and head-checkout analysis for ``fork_pr_runner_guard`` (RM#1996).

Pure helpers over workflow expressions and steps; no I/O. Split out of the
guard so each module stays small. Vendored next to the guard, so both must live
in the same directory.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

#: Text that checks out or fetches a pull request's head.
HEAD_REF_PATTERNS = (
    re.compile(r"github\.event\.pull_request\.head\.(sha|ref)"),
    re.compile(r"github\.head_ref"),
    re.compile(r"github\.event\.workflow_run\.head_(sha|branch)"),
    # The fork repository itself: checkout without a ref takes its default branch.
    re.compile(
        r"github\.event\.pull_request\.head\.repo\."
        r"(full_name|clone_url|ssh_url|git_url|html_url)"
    ),
    re.compile(
        r"github\.event\.workflow_run\.head_repository\."
        r"(full_name|clone_url|ssh_url|git_url|html_url)"
    ),
    re.compile(r"refs/pull/"),
    re.compile(r"pull/(\$\{\{[^}]*\}\}|[^\s/]+)/(head|merge)"),
    re.compile(r"gh\s+pr\s+checkout"),
)


#: A step restricted to this event never runs on a privileged trigger.
PULL_REQUEST_ONLY = "github.event_name == 'pull_request'"
#: Job-level conditions that make a privileged-event job same-repo only. Each is
#: accepted in either operand order and only when ANDed into the job ``if:``.
SAME_REPO_CONDITIONS = (
    "github.event.pull_request.head.repo.full_name == github.repository",
    "github.event.workflow_run.head_repository.full_name == github.repository",
)

#: Text that reads as a shell command able to fetch or check out code: any git
#: invocation (``git -C d ...``, ``git clone``, ``git pull``), ``gh pr checkout``,
#: ``gh repo clone``, and ``curl``/``wget`` downloads.
HEAD_SINK_COMMAND = re.compile(
    r"\bgit\b|\bgh\s+(?:pr\s+checkout|repo\s+clone)\b|\b(?:curl|wget)\b"
)

#: ``ctx['key']`` / ``ctx["key"]`` property access in a GitHub expression.
_BRACKET_PROPERTY = re.compile(r"\[\s*['\"]([A-Za-z_][\w-]*)['\"]\s*\]")


def dotted(text: str) -> str:
    """Rewrite bracket property access as dots, so ``head['sha']`` reads ``head.sha``.

    Postcondition: dotted-only text is returned unchanged.
    """
    return _BRACKET_PROPERTY.sub(r".\1", text)


def normalize(expression: str) -> str:
    """Collapse whitespace and strip ``${{ }}`` and redundant outer parens."""
    text = " ".join(expression.split())
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2].strip()
    while text.startswith("(") and _matching_paren(text) == len(text) - 1:
        text = text[1:-1].strip()
    return text


def _matching_paren(text: str) -> int:
    """Return the index of the paren closing ``text[0]``, or -1."""
    depth = 0
    in_quote = False
    for index, char in enumerate(text):
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char == "(":
            depth += 1
        elif not in_quote and char == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


def split_top_level(expression: str, operator: str) -> list[str]:
    """Split ``expression`` on ``operator`` outside quotes and parentheses."""
    parts: list[str] = []
    depth = 0
    in_quote = False
    start = 0
    index = 0
    while index < len(expression):
        char = expression[index]
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char == "(":
            depth += 1
        elif not in_quote and char == ")":
            depth -= 1
        elif not in_quote and depth == 0 and expression.startswith(operator, index):
            parts.append(expression[start:index].strip())
            index += len(operator)
            start = index
            continue
        index += 1
    parts.append(expression[start:].strip())
    return parts


def requires_conjunct(condition: Any, conjunct: str) -> bool:
    """True when ``condition`` is ``conjunct`` or ANDs it at the top level.

    A top-level ``||`` anywhere would let another branch bypass the conjunct,
    so it disqualifies the condition.
    """
    if not isinstance(condition, str):
        return False
    text = normalize(condition)
    wanted = normalize(conjunct)
    if text == wanted:
        return True
    if len(split_top_level(text, "||")) != 1:
        return False
    return any(normalize(part) == wanted for part in split_top_level(text, "&&"))


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def head_ref_aliases(*scopes: Any) -> set[str]:
    """Return env names, across ``scopes``, whose value reads the PR head."""
    names: set[str] = set()
    for scope in scopes:
        env = scope.get("env") if isinstance(scope, dict) else None
        for name, value in env.items() if isinstance(env, dict) else ():
            text = dotted("\n".join(_strings(value)))
            if any(p.search(text) for p in HEAD_REF_PATTERNS):
                names.add(str(name))
    return names


def _alias_reference(name: str) -> re.Pattern[str]:
    """Match an expression or shell reference to env var ``name``."""
    n = re.escape(name)
    return re.compile(
        rf"env\.{n}(?!\w)|env\[\s*['\"]{n}['\"]\s*\]"
        rf"|\$\{{?{n}(?!\w)|\$env:{n}(?!\w)"
    )


def _swap_operands(equality: str) -> str:
    """Return ``a == b`` as ``b == a``.

    Precondition: ``equality`` contains exactly one ``==``.
    """
    assert equality.count("==") == 1, equality
    left, right = (side.strip() for side in equality.split("=="))
    return f"{right} == {left}"


def same_repo_conditioned(condition: Any) -> bool:
    """True when ``condition`` ANDs a same-repo head check at the top level.

    Preconditions: none; non-string input is simply not a condition.
    Postconditions: True only for :data:`SAME_REPO_CONDITIONS` in either operand
    order, alone or as a top-level ``&&`` conjunct. A top-level ``||`` (or any
    form not understood) returns False, so the job is not exempted.
    """
    if not isinstance(condition, str):
        return False
    text = dotted(condition)
    return any(
        requires_conjunct(text, form)
        for wanted in SAME_REPO_CONDITIONS
        for form in (wanted, _swap_operands(wanted))
    )


def _checkout_action_reads_head(
    inputs: Any, patterns: Sequence[re.Pattern[str]]
) -> bool:
    """True when an ``actions/checkout`` ``with:`` names the head in ref/repository.

    Precondition: ``inputs`` is the step's ``with:`` value (any type).
    Postcondition: other inputs (``path``, ``token``...) never count.
    """
    if not isinstance(inputs, dict):
        return False
    text = dotted("\n".join(_strings([inputs.get("ref"), inputs.get("repository")])))
    return any(pattern.search(text) for pattern in patterns)


def _run_step_checks_out_head(
    run: str, step_text: str, patterns: Sequence[re.Pattern[str]]
) -> bool:
    """True when a ``run:`` step reads the head AND invokes a fetching command.

    Precondition: ``step_text`` is the step's dotted text (``run`` and ``env``).
    Postcondition: the head and the command are correlated per step, never per
    line, so ``REF=...`` on one line and ``git checkout "$REF"`` on another
    still count. A head ref handed to a script that runs no git/gh/curl/wget in
    that step is data.
    """
    if HEAD_SINK_COMMAND.search(run) is None:
        return False
    return any(pattern.search(step_text) for pattern in patterns)


def _step_checks_out_head(
    step: dict[str, Any], patterns: Sequence[re.Pattern[str]]
) -> bool:
    """Return whether one step checks out the PR head.

    Postcondition: only ``actions/checkout`` inputs and ``run:`` steps matching
    :data:`HEAD_SINK_COMMAND` count; a head ref handed to any other step is data.
    """
    uses, run = step.get("uses"), step.get("run")
    if isinstance(uses, str) and uses.startswith("actions/checkout"):
        return _checkout_action_reads_head(step.get("with"), patterns)
    if isinstance(run, str):
        fields = {k: v for k, v in step.items() if k != "if"}
        text = dotted("\n".join(_strings(fields)))
        return _run_step_checks_out_head(dotted(run), text, patterns)
    return False


def head_checkout_steps(job: dict[str, Any], aliases: set[str]) -> list[str]:
    """Return the names of steps that check out PR head on any event.

    ``aliases`` are workflow- or job-level env names that hold a head ref; a
    step that references one reads the head as surely as a literal does.
    """
    patterns = [*HEAD_REF_PATTERNS, *(_alias_reference(a) for a in aliases)]
    found: list[str] = []
    steps = job.get("steps")
    for index, step in enumerate(steps if isinstance(steps, list) else []):
        if not isinstance(step, dict):
            continue
        if requires_conjunct(step.get("if"), PULL_REQUEST_ONLY):
            continue
        if _step_checks_out_head(step, patterns):
            found.append(str(step.get("name") or step.get("uses") or f"#{index}"))
    return found


def passes_head_ref(job: dict[str, Any], aliases: set[str]) -> bool:
    """Return whether a reusable-workflow call hands the PR head to its callee.

    A ``uses:`` job has no steps of its own; the callee may check out whatever
    ref its inputs name, so a head ref in ``with:`` is a head checkout.
    """
    if not isinstance(job.get("uses"), str):
        return False
    patterns = [*HEAD_REF_PATTERNS, *(_alias_reference(a) for a in aliases)]
    text = dotted("\n".join(_strings(job.get("with", {}))))
    return any(pattern.search(text) for pattern in patterns)
