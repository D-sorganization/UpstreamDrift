#!/usr/bin/env python3
"""Verify every in-tree testpath, tests/ subdir, and suite marker is CI-reached or excluded (#15106)."""

from __future__ import annotations

import ast
from collections.abc import Set as AbstractSet
import json
import pathlib
import re
import sys
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python <3.11 fallback
    import tomli as tomllib  # type: ignore[import-not-found]

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import scripts.ci.check_suite_marker_ratchet as ratchet  # noqa: E402
from tests.helpers.suite_marker_reach import infer_directory_suite_marker  # noqa: E402
from tests.support.suite_markers import SUITE_MARKERS  # noqa: E402

CONFIG_PATH = REPO_ROOT / "scripts" / "config" / "unrun_test_suite_exclusions.json"
_SECTIONS = ("intree_testpaths", "tests_subdirs", "suite_markers")
_MARKER_RE = re.compile(r'(?:^|\s)-m\s+(?:"([^"]+)"|\'([^\']+)\'|([^\s\\]+))')


def _workflow_exec_text(root: pathlib.Path) -> str:
    wf_dir = root / ".github" / "workflows"
    if not wf_dir.is_dir():
        return ""
    return "\n".join(
        re.sub(r"--ignore=\S+", "", raw)
        for wf in sorted(wf_dir.glob("*.yml"))
        for raw in wf.read_text(encoding="utf-8", errors="replace").splitlines()
        if not raw.lstrip().startswith("#")
    )


def _selected_markers(wf_text: str) -> set[str]:
    exprs = [next(g for g in m.groups() if g) for m in _MARKER_RE.finditer(wf_text)]
    return {
        mk
        for mk in SUITE_MARKERS
        if any(
            not re.search(r"\bnot\s+$", e[: m.start()])
            for e in exprs
            for m in re.finditer(rf"\b{re.escape(mk)}\b", e)
        )
    }


def _scan_dir_markers(directory: pathlib.Path) -> tuple[bool, set[str], int]:
    files = list(ratchet.iter_test_files(directory))
    markers: set[str] = set()
    unmarked = 0
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        mod_m = ratchet._module_suite_markers(tree)
        markers.update(mod_m)
        for stmt in tree.body:
            if ratchet._is_test_function(stmt):
                fn_m = mod_m | ratchet._decorator_suite_markers(stmt)
                markers.update(fn_m)
                unmarked += int(not fn_m)
            elif ratchet._is_test_class(stmt):
                cls_m = mod_m | ratchet._decorator_suite_markers(stmt)
                markers.update(cls_m)
                for fn in filter(ratchet._is_test_function, stmt.body):
                    m = cls_m | ratchet._decorator_suite_markers(fn)
                    markers.update(m)
                    unmarked += int(not m)
    return bool(files), markers, unmarked


def _valid_entry(sec: str, val: Any) -> bool:
    if sec != "intree_testpaths":
        return isinstance(val, str) and bool(val.strip())
    if not isinstance(val, dict):
        return False
    cap, reason = val.get("max_unmarked_tests"), val.get("reason")
    ok_cap = isinstance(cap, int) and not isinstance(cap, bool) and cap >= 0
    return ok_cap and isinstance(reason, str) and bool(reason.strip())


def _load_config(cfg_path: pathlib.Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not cfg_path.is_file():
        return {}, [f"Missing exclusion inventory: {cfg_path}"]
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}, [f"Invalid JSON object in {cfg_path}"]
    errors: list[str] = []
    parsed: dict[str, dict[str, Any]] = {}
    for sec in _SECTIONS:
        raw = payload.get(sec)
        if not isinstance(raw, dict):
            errors.append(f"Section {sec!r} must be a JSON object")
            parsed[sec] = {}
            continue
        for k, v in raw.items():
            if not isinstance(k, str) or not k or not _valid_entry(sec, v):
                errors.append(f"Invalid exclusion ({sec}, {k})")
        parsed[sec] = raw
    return parsed, errors


def _reconcile(
    root: pathlib.Path,
    sec: str,
    cfg: dict[str, Any],
    active: AbstractSet[str],
    out: list[str],
) -> None:
    for tgt in sorted(active - set(cfg)):
        out.append(f"Unrun {sec} lacks exclusion: {tgt}")
    for tgt in sorted(set(cfg) - active):
        note = (
            " (missing)" if sec != "suite_markers" and not (root / tgt).is_dir() else ""
        )
        out.append(f"Stale exclusion ({sec}, {tgt}){note}")


def find_unrun_suite_violations(
    repo_root: pathlib.Path, config_path: pathlib.Path | None = None
) -> list[str]:
    """Return contract violations across testpaths, tests/ subdirs, and markers."""
    if not isinstance(repo_root, pathlib.Path) or not repo_root.is_dir():
        raise ValueError(f"repo_root must be an existing directory: {repo_root}")
    cfg, violations = _load_config(
        config_path or repo_root / "scripts" / "config" / CONFIG_PATH.name
    )
    wf_text = _workflow_exec_text(repo_root)
    py_data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    active_tp: set[str] = set()
    intree_cfg = cfg.get("intree_testpaths", {})
    for tp in py_data["tool"]["pytest"]["ini_options"]["testpaths"]:
        if tp == "tests":
            continue
        tp_dir = repo_root / tp
        if not tp_dir.is_dir():
            violations.append(f"Dead testpath: {tp}")
            continue
        _, _, unmarked = _scan_dir_markers(tp_dir)
        if unmarked == 0 or tp in wf_text:
            continue
        active_tp.add(tp)
        entry = intree_cfg.get(tp)
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("max_unmarked_tests"), int)
            and unmarked > entry["max_unmarked_tests"]
        ):
            violations.append(
                f"Testpath {tp} grew unmarked ({unmarked} > {entry['max_unmarked_tests']})"
            )

    active_dirs: set[str] = set()
    tests_root = repo_root / "tests"
    if tests_root.is_dir():
        for sub in sorted(tests_root.iterdir()):
            if not sub.is_dir() or sub.name.startswith((".", "__")):
                continue
            has_files, markers, _ = _scan_dir_markers(sub)
            rel = f"tests/{sub.name}"
            if has_files and not (
                infer_directory_suite_marker(f"{rel}/test_x.py")
                or rel in wf_text
                or "unit" in markers
            ):
                active_dirs.add(rel)

    active_markers = SUITE_MARKERS - _selected_markers(wf_text)
    for sec, active in (
        ("intree_testpaths", active_tp),
        ("tests_subdirs", active_dirs),
        ("suite_markers", active_markers),
    ):
        _reconcile(repo_root, sec, cfg.get(sec, {}), active, violations)
    return violations


def check_unrun_suite_exclusions(
    repo_root: pathlib.Path = REPO_ROOT, config_path: pathlib.Path = CONFIG_PATH
) -> tuple[bool, str]:
    """Run the unrun suite exclusion contract and return (passed, summary)."""
    violations = find_unrun_suite_violations(repo_root, config_path)
    if violations:
        return False, "Unrun suite exclusion contract FAILED:\n  - " + "\n  - ".join(
            violations
        )
    cfg, _ = _load_config(config_path)
    count = sum(len(cfg.get(s, {})) for s in _SECTIONS)
    return True, f"Unrun suite exclusion contract PASSED ({count} explicit exclusions)."


def main() -> int:
    passed, msg = check_unrun_suite_exclusions(REPO_ROOT, CONFIG_PATH)
    sys.stdout.write(msg + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
