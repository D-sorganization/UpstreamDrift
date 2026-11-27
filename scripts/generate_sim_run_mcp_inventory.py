"""Generate the committed ``ud-sim-mcp`` capability inventory document (#15020).

Usage:
    python3 -m scripts.generate_sim_run_mcp_inventory [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.sim_run.agent_api import (  # noqa: E402
    agent_api_env_docs,
    build_capability_inventory,
)

INVENTORY_MD = (
    REPO_ROOT / "docs" / "development" / "sim_run_mcp_capability_inventory.md"
)
INVENTORY_JSON = (
    REPO_ROOT / "docs" / "development" / "sim_run_mcp_capability_inventory.json"
)


def _render_markdown(inventory: dict[str, object]) -> str:
    env = agent_api_env_docs()
    lines = [
        "# Ud-Sim-MCP Capability Inventory",
        "",
        "<!-- Generated: python3 -m scripts.generate_sim_run_mcp_inventory -->",
        "",
        "Child task of [#15020](https://github.com/D-sorganization/UpstreamDrift/issues/15020).",
        "Statuses are **`available`** or **`unavailable`** only; nothing is labeled failed.",
        "",
        f"- Server: `{inventory['server']}`",
        f"- Schema: `{inventory['schema']}`",
        "",
        "## Authentication Environment",
        "",
        f"- Canonical token: `{env['auth_canonical']}` (alias: `{env['auth_aliases']}`)",
        (
            f"- Canonical Private Mode: `{env['private_mode_canonical']}` "
            f"(alias: `{env['private_mode_aliases']}`)"
        ),
        "",
        "Listener authentication hardening is tracked in private Repository_Management",
        "records; public docs do not describe that surface.",
        "",
        "## Simulation Engines",
        "",
        "| Engine | Description | Runtime Status | Unavailable Reason |",
        "| :--- | :--- | :---: | :--- |",
    ]
    for row in inventory["engines"]:
        reason = row.get("unavailable_reason") or "—"
        lines.append(
            f"| `{row['engine_type']}` | {row['description']} | "
            f"{row['runtime_status']} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Putting Tools",
            "",
            "| Tool | Runtime Status | Unavailable Reason |",
            "| :--- | :---: | :--- |",
        ]
    )
    for row in inventory["putting_tools"]:
        reason = row.get("unavailable_reason") or "—"
        lines.append(f"| `{row['id']}` | {row['runtime_status']} | {reason} |")
    agent_caps = inventory.get("agent_capabilities") or []
    if agent_caps:
        lines.extend(
            [
                "",
                "## Agent Capabilities",
                "",
                "| Tool | Runtime Status | Unavailable Reason |",
                "| :--- | :---: | :--- |",
            ]
        )
        for row in agent_caps:
            reason = row.get("unavailable_reason") or "—"
            lines.append(f"| `{row['id']}` | {row['runtime_status']} | {reason} |")
    lines.append("")
    return "\n".join(lines)


def generate() -> tuple[str, str]:
    inventory = build_capability_inventory()
    markdown = _render_markdown(inventory)
    json_text = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    return markdown, json_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when committed artifacts differ from the live inventory.",
    )
    args = parser.parse_args(argv)

    markdown, json_text = generate()

    if args.check:
        stale: list[str] = []
        if (
            not INVENTORY_MD.is_file()
            or INVENTORY_MD.read_text(encoding="utf-8") != markdown
        ):
            stale.append(str(INVENTORY_MD.relative_to(REPO_ROOT)))
        if (
            not INVENTORY_JSON.is_file()
            or INVENTORY_JSON.read_text(encoding="utf-8") != json_text
        ):
            stale.append(str(INVENTORY_JSON.relative_to(REPO_ROOT)))
        if stale:
            print(
                "Capability inventory is stale; run "
                "`python3 -m scripts.generate_sim_run_mcp_inventory`:\n"
                + "\n".join(f"  - {path}" for path in stale),
                file=sys.stderr,
            )
            return 1
        return 0

    INVENTORY_MD.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY_MD.write_text(markdown, encoding="utf-8")
    INVENTORY_JSON.write_text(json_text, encoding="utf-8")
    print(f"Wrote {INVENTORY_MD.relative_to(REPO_ROOT)}")
    print(f"Wrote {INVENTORY_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
