#!/usr/bin/env python3
"""Change-fragment schema: parse, validate, render (RM-5 / #1894).

Split out of ``changes_fragment.py``, which stays the public facade and CLI.
Standard library only; copied fleet-wide next to ``development_log.py``.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


def sibling(name: str) -> ModuleType:
    """Import a sibling fleet module by package, else by file path.

    The fleet copies these modules side by side into repositories that do not
    expose ``shared_scripts`` as an importable package.
    """
    try:
        return importlib.import_module(f"shared_scripts.{name}")
    except ImportError:
        cached = sys.modules.get(f"_fleet_{name}")
        if cached is not None:
            return cached
        path = Path(__file__).with_name(f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"_fleet_{name}", path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise
        module = importlib.util.module_from_spec(spec)
        # Register before exec: dataclasses resolve their module in sys.modules.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            # Never leave a half-initialised module cached for the next caller.
            sys.modules.pop(spec.name, None)
            raise
        return module


development_log = sibling("development_log")
handoff_validator = sibling("handoff_validator")

CHANGES_DIR = "changes"
FILE_NAME = re.compile(r"^(?P<issue>\d+)(?:-[a-z0-9][a-z0-9-]*)?\.md$")
FRONT_MATTER = re.compile(r"\A---[ \t]*\n(?P<meta>.*?)\n?---[ \t]*(?:\n|\Z)", re.DOTALL)
META_LINE = re.compile(r"^(?P<key>[a-z_]+):[ \t]*(?P<value>.*?)[ \t]*$")

REQUIRED_KEYS = ("issue", "summary")
OPTIONAL_KEYS = ("dl_state", "next_step", "title", "owner", "branch", "paths")
ALLOWED_KEYS = frozenset(REQUIRED_KEYS + OPTIONAL_KEYS)


def _title_case_checker() -> ModuleType | None:
    """Return ``document_title_case``, or ``None`` when this copy lacks it.

    Fail-open: repositories that vendor the fragment modules without the
    checker (or its ``defusedxml`` dependency) skip the title check.
    """
    try:
        return sibling("document_title_case")
    except (ImportError, FileNotFoundError):
        return None


class FragmentError(ValueError):
    """Raised when a fragment, or a request built from one, is invalid."""


@dataclass(frozen=True)
class Fragment:
    """One parsed, validated change fragment.

    Invariants: ``issue >= 1``; ``summary`` is one non-empty line without an
    unescaped ``|``; ``dl_state`` is ``None`` or a development-log state; a
    live ``dl_state`` always carries a ``next_step``.
    """

    issue: int
    summary: str
    dl_state: str | None = None
    next_step: str | None = None
    title: str | None = None
    owner: str | None = None
    branch: str | None = None
    paths: str | None = None
    handoff: str = ""


def is_fragment_path(rel_path: str) -> bool:
    """True when ``rel_path`` names a fragment: ``changes/<name>.md``, not README."""
    parts = rel_path.replace("\\", "/").strip().split("/")
    return (
        len(parts) == 2
        and parts[0] == CHANGES_DIR
        and parts[1].endswith(".md")
        and parts[1].lower() != "readme.md"
    )


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def _unquote(value: str) -> str:
    """Return a scalar's string value: JSON-style double or YAML single quotes."""
    if len(value) >= 2 and value[0] == value[-1] == '"':
        decoded = json.loads(value)
        if not isinstance(decoded, str):  # pragma: no cover - json of "..." is str
            raise ValueError("not a string")
        return decoded
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def _parse_meta(meta: str) -> tuple[dict[str, str], list[str]]:
    """Parse ``key: value`` front-matter lines into a dict plus findings."""
    values: dict[str, str] = {}
    errors: list[str] = []
    for raw in meta.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        match = META_LINE.match(raw)
        if match is None:
            errors.append(f"unparsable front-matter line: {raw.strip()!r}")
            continue
        key = match.group("key")
        if key not in ALLOWED_KEYS:
            errors.append(f"unknown key {key!r}; allowed: {sorted(ALLOWED_KEYS)}")
            continue
        if key in values:
            errors.append(f"duplicate key {key!r}")
            continue
        try:
            values[key] = _unquote(match.group("value")).strip()
        except ValueError:
            errors.append(f"{key}: malformed quoted value")
    return values, errors


def effective_title(title: str | None, summary: str) -> str:
    """The development-log heading text: ``title`` else ``summary``, title-cased.

    Falls back to the raw text when the title-case checker is unavailable.
    """
    text = title or summary
    checker = _title_case_checker()
    return text if checker is None else checker.expected_title(text)


def _check_values(values: dict[str, str]) -> list[str]:
    """Validate parsed front-matter values against the closed schema."""
    errors: list[str] = []
    issue = values.get("issue", "")
    if not issue.isdigit() or int(issue) < 1:
        errors.append(f"issue must be a positive issue number, got {issue!r}")
    summary = values.get("summary", "")
    if not summary:
        errors.append("summary is required and must be non-empty")
    elif "|" in summary:
        # spec_changelog splits table cells on every pipe, escaped or not.
        errors.append("summary contains '|', a markdown column break; reword it")
    state = values.get("dl_state")
    if state is not None and state not in development_log.VALID_STATES:
        errors.append(
            f"dl_state {state!r} is not one of {sorted(development_log.VALID_STATES)}"
        )
    if state in development_log.ACTIVE_STATES and not values.get("next_step"):
        errors.append(f"next_step is required when dl_state is {state!r}")
    if state in development_log.BRANCH_STATES and not values.get("branch"):
        errors.append(f"branch is required when dl_state is {state!r}")
    if state in development_log.PAUSED_STATES:
        errors.append(
            f"dl_state {state!r} needs a Parked reason the fragment schema does "
            "not carry; park the entry by editing the development log directly"
        )
    title = values.get("title")
    checker = _title_case_checker() if title else None
    if title and checker is not None:
        expected = checker.expected_title(title)
        if expected != title:
            # The collated DL heading is checked by document_title_case.
            errors.append(f"title must be title case: {title!r} -> {expected!r}")
    for key, value in values.items():
        if "\n" in value:
            errors.append(f"{key} must be a single line")
        if development_log.PLACEHOLDER.search(value):
            errors.append(f"{key} holds an unedited placeholder: {value!r}")
    return errors


def parse(text: str, name: str) -> tuple[Fragment | None, list[str]]:
    """Parse fragment ``text`` whose file name is ``name``.

    Returns the fragment (``None`` when invalid) and every finding.
    """
    errors: list[str] = []
    for pattern, secret_type in handoff_validator.SECRET_PATTERNS:
        if pattern.search(text):
            errors.append(f"potential {secret_type} (secret) detected; remove it")
    if not text.startswith("---"):
        return None, [*errors, "missing YAML frontmatter (must start with '---')"]
    match = FRONT_MATTER.match(text)
    if match is None:
        return None, [*errors, "front matter has no closing '---' line"]
    values, meta_errors = _parse_meta(match.group("meta"))
    errors.extend(meta_errors)
    errors.extend(_check_values(values))
    name_match = FILE_NAME.match(name)
    if name_match is None:
        errors.append(
            f"file name {name!r} must be '<issue>.md' or '<issue>-<slug>.md' "
            "(lowercase slug)"
        )
    elif values.get("issue", "").isdigit() and name_match.group("issue") != str(
        int(values["issue"])
    ):
        errors.append(
            f"file name {name!r} must start with the fragment issue {values['issue']}"
        )
    if errors:
        return None, errors
    fragment = Fragment(
        issue=int(values["issue"]),
        summary=values["summary"],
        dl_state=values.get("dl_state") or None,
        next_step=values.get("next_step") or None,
        title=values.get("title") or None,
        owner=values.get("owner") or None,
        branch=values.get("branch") or None,
        paths=values.get("paths") or None,
        handoff=text[match.end() :].strip(),
    )
    return fragment, []


def validate_fragment_file(path: Path) -> list[str]:
    """Return human-readable findings for the fragment at ``path`` (empty = ok)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"unable to read fragment: {exc}"]
    return parse(text, path.name)[1]


def load_fragment(path: Path) -> Fragment:
    """Load and validate one fragment. Raises :class:`FragmentError` if invalid."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FragmentError(f"{path}: unable to read fragment: {exc}") from exc
    fragment, errors = parse(text, path.name)
    if fragment is None:
        raise FragmentError(f"{path}: " + "; ".join(errors))
    return fragment


def find_fragments(repo_root: Path) -> list[Path]:
    """Every fragment under ``<repo_root>/changes/``, sorted by name."""
    directory = repo_root / CHANGES_DIR
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.glob("*.md") if is_fragment_path(f"{CHANGES_DIR}/{p.name}")
    )


def slugify(text: str, max_length: int = 40) -> str:
    """Lowercase ``text`` into a file-name slug of at most ``max_length``."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].rstrip("-")


def render_fragment(
    issue: int,
    summary: str,
    *,
    handoff: str = "",
    **optional: str | None,
) -> str:
    """Render fragment text. String values are written JSON-quoted (valid YAML)."""
    lines = ["---", f"issue: {issue}", f"summary: {json.dumps(summary)}"]
    for key in OPTIONAL_KEYS:
        value = optional.get(key)
        if value:
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    lines.append("---")
    body = f"\n{handoff.strip()}\n" if handoff.strip() else ""
    return "\n".join(lines) + "\n" + body
