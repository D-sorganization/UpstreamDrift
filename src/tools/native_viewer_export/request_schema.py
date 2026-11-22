"""One request schema for the native viewer exporter (#15150 G7, C11).

The schema is derived from the exporter's own ``cli.build_parser()`` so a new
command-line flag is reachable from REST and MCP without further edits.
``request_schema_settings.json`` only places flags into groups (with a
maturity level and source notes), lists flags that make no sense remotely,
and names the roots a remote request may read below and write below.
Options fixed when the bundle is built (the "Shaft" group) are derived from
the spec builder's parser and reported as stage ``bundle_build``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
from functools import cache
from pathlib import Path
from typing import Any, NoReturn

from src.config.capability_state import VALID_MATURITY_LEVELS
from src.tools.native_viewer_export.cli import build_parser, build_settings

_SETTINGS_PATH = Path(__file__).with_name("request_schema_settings.json")
_REPO_ROOT = Path(__file__).resolve().parents[3]
EXPORT_STAGE = "export"


@cache
def load_schema_settings() -> dict[str, Any]:
    """The settings document, checked against ``VALID_MATURITY_LEVELS``.

    Raises:
        ValueError: the maturity order or a group maturity is not canonical.
    """
    settings = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    if set(settings["maturity_order"]) != VALID_MATURITY_LEVELS:
        raise ValueError("maturity_order must list exactly VALID_MATURITY_LEVELS")
    for group in [*settings["groups"], settings["fallback_group"]]:
        if group["maturity"] not in VALID_MATURITY_LEVELS:
            raise ValueError(f"group {group['name']!r} has an unknown maturity")
    return settings


def _titles(parser: argparse.ArgumentParser) -> dict[int, str]:
    """Argparse group title of each action, keyed by ``id(action)``."""
    return {
        id(action): str(group.title)
        for group in parser._action_groups
        for action in group._group_actions
    }


def _group_of(dest: str, title: str) -> dict[str, Any]:
    """The first settings group whose rules match, else the fallback group."""
    settings = load_schema_settings()
    for group in settings["groups"]:
        if (
            dest in group.get("dests", ())
            or title in group.get("argparse_groups", ())
            or dest.startswith(tuple(group.get("dest_prefixes", ())))
        ):
            return group
    return dict(settings["fallback_group"])


def _kind(action: argparse.Action) -> str:
    """JSON-facing kind of ``action``: bool, flag, list, path, number, ..."""
    if isinstance(action, argparse.BooleanOptionalAction):
        return "bool"
    if action.nargs == 0:
        return "flag"
    if action.nargs not in (None, "?") or type(action).__name__ == "_AppendAction":
        return "list"
    if action.type is Path:
        return "path"
    if action.type is int:
        return "integer"
    return "number" if action.type is float else "string"


def _json_safe(value: Any) -> Any:
    """``value`` as plain JSON (paths and tuples become strings and lists)."""
    if value is argparse.SUPPRESS:
        return None
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _remote(dest: str, kind: str, stage: str) -> str:
    if stage != EXPORT_STAGE:
        return "bundle_build_only"
    if dest in load_schema_settings()["remote_exclusions"]:
        return "excluded"
    return "sandboxed_path" if kind == "path" else "allowed"


def option_schema(
    parser: argparse.ArgumentParser | None = None, stage: str = EXPORT_STAGE
) -> list[dict[str, Any]]:
    """One entry per option of ``parser`` (default: the exporter's parser).

    Postcondition: every action except ``--help`` appears exactly once.
    """
    parser = build_parser() if parser is None else parser
    if not isinstance(parser, argparse.ArgumentParser):
        raise TypeError("parser must be an argparse.ArgumentParser")
    titles = _titles(parser)
    entries = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        group, kind = _group_of(action.dest, titles.get(id(action), "")), _kind(action)
        entries.append(
            {
                "name": action.dest,
                "flags": list(action.option_strings),
                "kind": kind,
                "default": _json_safe(action.default),
                "choices": _json_safe(action.choices),
                "help": action.help or "",
                "required": bool(action.required),
                "group": group["name"],
                "maturity": group["maturity"],
                "stage": stage,
                "remote": _remote(action.dest, kind, stage),
            }
        )
    return entries


def _stage_options() -> list[dict[str, Any]]:
    """Options of the non-export stages, restricted to their listed groups."""
    entries = []
    for name, stage in load_schema_settings()["stages"].items():
        module, _, factory = stage["parser"].partition(":")
        parser = getattr(importlib.import_module(module), factory)()
        options = option_schema(parser, stage=name)
        entries += [o for o in options if o["group"] in stage["groups"]]
    return entries


def _rank(maturity: str) -> int:
    order = load_schema_settings()["maturity_order"]
    if maturity not in order:
        raise ValueError(f"min_maturity must be one of {order}, got {maturity!r}")
    return int(order.index(maturity))


def describe(request: dict[str, Any] | None = None) -> dict[str, Any]:
    """The full schema: groups (maturity, sources) and options.

    ``request["min_maturity"]`` (optional) hides groups below that level.

    Raises:
        ValueError: ``min_maturity`` is not a canonical maturity level.
    """
    settings = load_schema_settings()
    floor = _rank((request or {}).get("min_maturity") or settings["maturity_order"][0])
    groups = [
        {k: g[k] for k in ("name", "maturity", "sources", "limitations") if k in g}
        for g in [*settings["groups"], settings["fallback_group"]]
        if _rank(g["maturity"]) >= floor
    ]
    shown = {g["name"] for g in groups}
    options = [o for o in option_schema() + _stage_options() if o["group"] in shown]
    return {
        "schema": settings["schema_id"],
        "maturity_order": settings["maturity_order"],
        "groups": groups,
        "options": options,
    }


def allowed_roots() -> tuple[tuple[Path, ...], Path]:
    """Resolved input roots and output root (environment over settings)."""
    roots = load_schema_settings()["roots"]
    env_inputs = os.environ.get(roots["input_roots_env"], "")
    inputs = (
        env_inputs.split(os.pathsep) if env_inputs else roots["default_input_roots"]
    )
    output = os.environ.get(roots["output_root_env"]) or roots["default_output_root"]
    resolve = lambda p: (_REPO_ROOT / p).resolve()  # noqa: E731 - absolute p wins
    return tuple(resolve(p) for p in inputs if p), resolve(output)


def contained_path(value: Any, *, output: bool) -> str:
    """``value`` resolved below an allowed root (symlinks followed).

    Raises:
        TypeError: ``value`` is not a string.
        ValueError: the resolved path leaves every allowed root.
    """
    if not isinstance(value, str) or not value:
        raise TypeError("path options take a non-empty string")
    inputs, out_root = allowed_roots()
    for base in (out_root,) if output else inputs:
        resolved = (base / value).resolve()
        if resolved.is_relative_to(base):
            return str(resolved)
    where = "output root" if output else "input roots"
    raise ValueError(f"path {value!r} is outside the allowed {where}")


def _argv_for(action: argparse.Action, value: Any) -> list[str]:
    """Command-line words that set ``action`` to ``value``."""
    flag, kind = action.option_strings[0], _kind(action)
    if kind in ("bool", "flag"):
        if not isinstance(value, bool):
            raise TypeError(f"{action.dest!r} takes true or false")
        if kind == "bool" and not value:
            return [action.option_strings[1]]
        return [flag] if value else []
    if kind == "path":
        output = action.dest in load_schema_settings()["output_options"]
        return [flag, contained_path(value, output=output)]
    if kind == "list":
        values = [str(v) for v in (value if isinstance(value, list) else [value])]
        if type(action).__name__ == "_AppendAction":
            return [word for v in values for word in (flag, v)]
        return [flag, *values]
    return [flag, str(value)]


def _rejection(name: str) -> str:
    """Why option ``name`` cannot be set in an export request."""
    reason = load_schema_settings()["remote_exclusions"].get(name)
    if reason is not None:
        return f"option {name!r} is not accepted remotely: {reason}"
    if name in {o["name"] for o in _stage_options()}:
        return f"option {name!r} is set when the bundle is built, not at export"
    return f"unknown option {name!r}"


def _parse_strict(parser: argparse.ArgumentParser, argv: list[str]) -> Any:
    """``parser.parse_args(argv)`` raising ``ValueError`` instead of exiting."""

    def _fail(message: str) -> NoReturn:
        raise ValueError(f"invalid export request: {message}")

    parser.error = _fail  # type: ignore[method-assign]
    return parser.parse_args(argv)


def validate(request: dict[str, Any]) -> dict[str, Any]:
    """Check an export request and return the equivalent command line.

    Precondition: ``request["options"]`` maps schema option names to values
    (``None`` means the default). Postcondition: ``argv`` parses with the
    exporter's parser and converts to ``ExportSettings``.

    Raises:
        TypeError: the request or a value has the wrong type.
        ValueError: an unknown, excluded or bundle-build option, a path
            outside the allowed roots, or a value the exporter rejects.
    """
    options = request.get("options") if isinstance(request, dict) else None
    if not isinstance(options, dict):
        raise TypeError("request must be an object with an 'options' object")
    parser = build_parser()
    actions = {a.dest: a for a in parser._actions}
    argv: list[str] = []
    for name, value in options.items():
        if name not in actions or name in load_schema_settings()["remote_exclusions"]:
            raise ValueError(_rejection(name))
        if value is not None:
            argv += _argv_for(actions[name], value)
    try:
        build_settings(_parse_strict(parser, argv))
    except FileNotFoundError as exc:
        raise ValueError(f"invalid export request: {exc}") from exc
    prefix = load_schema_settings()["command_prefix"]
    return {
        "schema": load_schema_settings()["schema_id"],
        "argv": argv,
        "command_line": shlex.join([*prefix, *argv]),
    }
