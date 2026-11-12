#!/usr/bin/env python3
# mypy: ignore-errors
"""Run the vendored Fleet Agent API conformance template (UpstreamDrift #15022)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEMPLATE = (
    Path(__file__).resolve().parent
    / "config"
    / "fleet_agent_api_conformance"
    / "v1"
    / "template.json"
)


def _load_template(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_server_command(template: dict[str, Any]) -> list[str]:
    declared = (template.get("server") or {}).get("command", "ud-sim-mcp")
    resolved = shutil.which(declared)
    if resolved:
        return [resolved]
    return [sys.executable, "-m", "src.sim_run.mcp"]


def _assert_payload(assertions: list[str], payload: dict[str, Any]) -> list[str]:
    from src.sim_run.fleet_agent_api import (
        CONTRACT_VERSION,
        assert_run_record_conformance,
        map_sim_run_record_v1,
    )

    failures: list[str] = []
    for item in assertions:
        if item == "contract_version":
            if payload.get("contract_version") != CONTRACT_VERSION:
                failures.append("missing or invalid contract_version")
            continue
        if item == "request_schema":
            if "request_schema" not in payload:
                failures.append("missing request_schema")
            continue
        if item == "sim_run_record_v1":
            failures.extend(assert_run_record_conformance(payload))
            continue
        if item == "capability_record_v1":
            if payload.get("schema") != "ud-agent-capability-run/v1":
                failures.append("missing ud-agent-capability-run/v1 schema")
            if payload.get("status") not in ("succeeded", "failed"):
                failures.append(
                    f"unexpected capability status {payload.get('status')!r}"
                )
            continue
        if item.startswith("status:"):
            want = item.split(":", 1)[1]
            got = payload.get("status")
            if got != want and payload.get("schema_version"):
                mapped = map_sim_run_record_v1(payload)
                got = mapped.get("status")
            if got != want:
                failures.append(f"status {got!r} != {want!r}")
            continue
        if item == "provenance.contract_version":
            prov = payload.get("provenance") or {}
            if not prov.get("contract_version"):
                mapped = map_sim_run_record_v1(payload)
                prov = mapped.get("provenance") or {}
            if prov.get("contract_version") != CONTRACT_VERSION:
                failures.append("missing provenance.contract_version")
            continue
    return failures


async def _run_stdio_checks(template: dict[str, Any], repo_root: Path) -> list[str]:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        return [f"MCP SDK not installed: {exc}"]

    command = _resolve_server_command(template)
    server = StdioServerParameters(
        command=command[0],
        args=command[1:],
        env={**os.environ, "UPSTREAMDRIFT_PRIVATE_MODE": "0"},
    )
    failures: list[str] = []
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            required = set(template.get("required_tools", []))
            missing = required - names
            if missing:
                failures.append(f"missing tools: {sorted(missing)}")

            for check in template.get("checks", []):
                if check.get("kind") == "handshake":
                    continue
                if check.get("kind") == "list_tools":
                    continue
                tool = check["tool"]
                arguments = check.get("arguments", {})
                try:
                    response = await session.call_tool(tool, arguments)
                except Exception as exc:  # noqa: BLE001 — record any MCP tool failure
                    failures.append(f"{check['id']}: {exc}")
                    continue
                if response.isError:
                    failures.append(f"{check['id']}: tool error")
                    continue
                text = response.content[0].text if response.content else "{}"
                payload = json.loads(text)
                failures.extend(
                    f"{check['id']}: {msg}"
                    for msg in _assert_payload(check.get("assertions", []), payload)
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=TEMPLATE,
        help="Conformance template JSON path.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    template = _load_template(args.template)
    failures = asyncio.run(_run_stdio_checks(template, args.repo_root.resolve()))
    if failures:
        for line in failures:
            logger.error("%s", line)
        return 1
    logger.info(
        "agent API conformance: %d checks passed", len(template.get("checks", []))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
