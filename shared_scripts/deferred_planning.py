"""Validate repository-owned future plans; never infer completion from deferral.

This module is network-free. A migration caller must fetch published bytes from
the owning repository's default branch, then call ``closure_payload`` immediately
before applying the reviewed GitHub disposition. The module never closes issues.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ENTRY_FIELDS = {
    "id",
    "title",
    "source_issue",
    "source_snapshot",
    "record",
    "disposition",
    "kind",
    "status",
    "rationale",
    "prerequisites",
    "acceptance",
    "board",
    "activation_issue",
}
REPOSITORY = re.compile(r"D-sorganization/[A-Za-z0-9_.-]+\Z")


class PlanningError(ValueError):
    """The plan is incomplete, unsafe to migrate, or violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanningError(message)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_migration_pr_body(body: str) -> None:
    """Reject empty text and issue-closing phrases, including negated phrases.

    GitHub can interpret a closing keyword inside prose such as "do not
    auto-close". A migration must use references and separately record a
    not-planned disposition after verifying default-branch publication.
    """
    _require(_text(body), "migration PR body must be nonempty text")
    closing = re.search(
        r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+"
        r"(?:(?:[\w.-]+/[\w.-]+)?#\d+"
        r"|https://github\.com/[\w.-]+/[\w.-]+/issues/\d+)",
        body,
        re.IGNORECASE,
    )
    _require(closing is None, "migration PR must reference issues without closing them")


def _object(value: object, keys: set[str], context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{context}: expected object")
    assert isinstance(value, dict)  # validated above; type narrowing only
    _require(set(value) == keys, f"{context}: missing or unknown fields")
    return value


def _file(root: Path, name: object) -> Path:
    _require(_text(name), "artifact path must be text")
    assert isinstance(name, str)
    _require(not any(c in name for c in ("\\", ":")), "nonportable artifact path")
    relative = Path(name)
    _require(not relative.is_absolute() and ".." not in relative.parts, "unsafe path")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()), "artifact escapes planning root")
    _require(path.is_file(), f"missing artifact: {name}")
    return path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlanningError(f"cannot read JSON: {path.name}") from exc


def activation_ready(entry: dict[str, Any]) -> bool:
    """Resource-ready for Board-approved execution, not scientifically validated.

    Evidence references still require human/Board review. Their presence does not
    authenticate an artifact or authorize a coding agent to perform a human study.
    """
    board = entry.get("board", {})
    resources = entry.get("prerequisites", [])
    return bool(
        isinstance(board, dict)
        and board.get("decision") == "approved"
        and _text(board.get("evidence"))
        and isinstance(resources, list)
        and resources
        and all(isinstance(r, dict) and _text(r.get("evidence")) for r in resources)
    )


def _validate_entry(root: Path, entry: object, repository: str) -> dict[str, Any]:
    item = _object(entry, ENTRY_FIELDS, "entry")
    for field in ("id", "title", "rationale"):
        _require(_text(item[field]), f"{field}: nonempty text required")
    _require(bool(re.fullmatch(r"DV-[1-9][0-9]*", item["id"])), "invalid plan id")
    prefix = f"https://github.com/{repository}/issues/"
    issue = item["source_issue"]
    _require(isinstance(issue, str) and issue.startswith(prefix), "wrong source repo")
    _require(
        bool(re.fullmatch(r"[1-9][0-9]*", issue[len(prefix) :])), "invalid issue URL"
    )
    _require(item["disposition"] in ("defer", "split"), "invalid disposition")
    _require(
        item["kind"]
        in (
            "physical-measurement",
            "human-validation",
            "external-review",
            "research-idea",
        ),
        "invalid kind",
    )
    _require(
        item["status"] in ("deferred", "ready", "activated", "declined"),
        "invalid status",
    )
    for field in ("acceptance", "prerequisites"):
        _require(
            isinstance(item[field], list) and bool(item[field]), f"{field}: required"
        )
    _require(all(_text(x) for x in item["acceptance"]), "empty acceptance criterion")
    for prerequisite in item["prerequisites"]:
        resource = _object(prerequisite, {"name", "evidence"}, "prerequisite")
        _require(_text(resource["name"]), "empty prerequisite")
        _require(
            resource["evidence"] is None or _text(resource["evidence"]),
            "invalid evidence",
        )
    board = _object(item["board"], {"decision", "evidence"}, "board")
    _require(
        board["decision"] in ("pending", "approved", "rejected"),
        "invalid Board decision",
    )
    _require(
        board["evidence"] is None or _text(board["evidence"]), "invalid Board evidence"
    )
    if board["decision"] != "pending":
        _require(_text(board["evidence"]), "Board decision requires a durable receipt")
    if item["status"] in ("ready", "activated"):
        _require(activation_ready(item), "activation prerequisites not satisfied")
    if item["status"] == "declined":
        _require(
            board["decision"] == "rejected", "declined plan requires Board decision"
        )
    activation = item["activation_issue"]
    if item["status"] == "activated":
        _require(
            isinstance(activation, str)
            and bool(
                re.fullmatch(
                    re.escape(prefix) + r"[1-9][0-9]*",
                    activation,
                )
            ),
            "activated plan requires owning-repository issue",
        )
    else:
        _require(activation is None, "inactive plan cannot declare activation issue")
    record = _file(root, item["record"])
    _require(
        record.suffix == ".md" and bool(record.read_text(encoding="utf-8").strip()),
        "empty record",
    )
    snapshot = _read_json(_file(root, item["source_snapshot"]))
    _require(isinstance(snapshot, dict), "source snapshot must be an object")
    for field in ("title", "body", "updated_at", "html_url"):
        _require(_text(snapshot.get(field)), f"source snapshot missing {field}")
    _require(snapshot["html_url"] == issue, "snapshot issue mismatch")
    _require(
        str(snapshot.get("number")) == issue[len(prefix) :], "snapshot number mismatch"
    )
    return item


def validate_catalog(root: Path) -> dict[str, Any]:
    """Read the strict v1 catalog, contained files, and preserved source snapshots."""
    catalog = _object(
        _read_json(root / "catalog.json"),
        {
            "schema_version",
            "repository",
            "entries",
        },
        "catalog",
    )
    _require(
        type(catalog["schema_version"]) is int and catalog["schema_version"] == 1,
        "unsupported schema",
    )
    repository = catalog["repository"]
    _require(
        isinstance(repository, str) and bool(REPOSITORY.fullmatch(repository)),
        "invalid repository",
    )
    _require(isinstance(catalog["entries"], list), "entries must be a list")
    ids: set[str] = set()
    sources: set[str] = set()
    for value in catalog["entries"]:
        item = _validate_entry(root, value, repository)
        _require(item["id"] not in ids, "duplicate plan id")
        _require(item["source_issue"] not in sources, "duplicate source issue")
        ids.add(item["id"])
        sources.add(item["source_issue"])
    return catalog


def closure_payload(
    root: Path,
    plan_id: str,
    live_issue: dict[str, Any],
    published: dict[str, bytes],
) -> dict[str, str]:
    """Fail closed unless an unchanged external-only issue has a durable plan.

    ``published`` must contain bytes fetched from the owning repository's default
    branch at a recorded commit, keyed by paths relative to the planning root.
    Callers must apply the exempt ``roadmap`` label and post the durable record
    link before this payload is PATCHed; never use a closing keyword in the PR.
    GitHub has no conditional issue PATCH here, so recheck immediately before it.
    """
    catalog = validate_catalog(root)
    entries = [e for e in catalog["entries"] if e["id"] == plan_id]
    _require(len(entries) == 1, "unknown plan")
    entry = entries[0]
    _require(entry["disposition"] == "defer", "split issue retains software work")
    _require(entry["status"] == "deferred", "only deferred plans can migrate")
    for name in ("catalog.json", entry["record"], entry["source_snapshot"]):
        _require(
            published.get(name) == _file(root, name).read_bytes(),
            f"unverified published artifact: {name}",
        )
    source = _read_json(_file(root, entry["source_snapshot"]))
    _require(live_issue.get("state") == "open", "issue is not open")
    # Comments and labels change updated_at without changing scope. Preserve the
    # snapshot timestamp for audit, but compare the actual source text/identity.
    for field in ("number", "html_url", "title", "body"):
        _require(live_issue.get(field) == source.get(field), f"source changed: {field}")
    labels = live_issue.get("labels", [])
    _require(isinstance(labels, list), "invalid issue labels")
    names = {x.get("name", "") if isinstance(x, dict) else x for x in labels}
    _require(
        not names.intersection({"do-not-automate", "claim:user", "claim:local"}),
        "protected issue",
    )
    return {"state": "closed", "state_reason": "not_planned"}


def main() -> int:
    """Validate only. This command does not mutate GitHub or local files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="docs/development/planning directory")
    args = parser.parse_args()
    try:
        catalog = validate_catalog(args.root)
    except PlanningError as exc:
        parser.exit(1, f"Invalid planning catalog: {exc}\n")
    print(f"Valid: {catalog['repository']} ({len(catalog['entries'])} plans)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
