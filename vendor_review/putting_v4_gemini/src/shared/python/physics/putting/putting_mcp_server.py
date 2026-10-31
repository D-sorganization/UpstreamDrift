"""Model Context Protocol (MCP) & JSON-RPC 2.0 server for autonomous agentic putting control.

Part of the UpstreamDrift 3D Counterbalanced Putting Dynamics Suite (v4.0).
Enables external AI agents and automation harnesses to inspect tools, run 100/1000-putt
Monte Carlo batches, compute player-specific Strokes Gained curves, calibrate via the
USGA Stimpmeter ramp, and export/load simulation states via JSON-RPC 2.0.

Architecture budgets enforced:
- <= 500 LOC per file
- <= 100 LOC per function
- <= 8 parameters per function
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from .green_surface_undulation import GreenUndulationProfile
    from .stimpmeter_reader import solve_player_optimal_green_read
    from .strokes_gained_analytics import (
        PlayerSkillProfile,
        decompose_putting_variance_channels,
        export_simulation_state_json,
        load_simulation_state_json,
        optimize_capture_speed_strategy,
        simulate_parallel_putt_batch,
    )
except ImportError:
    from green_surface_undulation import GreenUndulationProfile  # type: ignore[no-redef]
    from stimpmeter_reader import solve_player_optimal_green_read  # type: ignore[no-redef]
    from strokes_gained_analytics import (  # type: ignore[no-redef]
        PlayerSkillProfile,
        decompose_putting_variance_channels,
        export_simulation_state_json,
        load_simulation_state_json,
        optimize_capture_speed_strategy,
        simulate_parallel_putt_batch,
    )


class PuttingMCPServer:
    """Stateless/stateful MCP & JSON-RPC 2.0 controller for the 3D Putting Suite."""

    def __init__(self) -> None:
        self._last_state: Dict[str, Any] = {}

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return MCP tool definitions for agentic discovery."""
        return [
            {
                "name": "run_parallel_putt_batch",
                "description": "Simulate 100-1000 putts with surface roughness, skill noise, and counter-torque.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "distance_ft": {"type": "number", "default": 12.0},
                        "n_trials": {"type": "integer", "default": 250},
                        "entry_speed_m_s": {"type": "number", "default": 0.58},
                        "slope_y_pct": {"type": "number", "default": 2.2},
                        "stimp_rating": {"type": "number", "default": 11.0},
                        "tau_preload_nm": {"type": "number", "default": 0.60},
                        "green_read_sd_pct": {"type": "number", "default": 0.25},
                        "surface_roughness_mm": {"type": "number", "default": 0.35},
                    },
                },
            },
            {
                "name": "compute_player_strokes_gained",
                "description": "Compute player-specific Bellman expected strokes E_player(d) across 0-45 ft and optimal capture speed.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "distance_ft": {"type": "number", "default": 12.0},
                        "slope_y_pct": {"type": "number", "default": 2.2},
                        "tau_preload_nm": {"type": "number", "default": 0.60},
                        "base_face_sd_deg": {"type": "number", "default": 0.48},
                    },
                },
            },
            {
                "name": "run_stimpmeter_green_read",
                "description": "Roll USGA Stimpmeter calibration balls and compute optimal player green read.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "distance_ft": {"type": "number", "default": 12.0},
                        "stimp_rating": {"type": "number", "default": 11.0},
                        "slope_y_pct": {"type": "number", "default": 2.2},
                        "tau_preload_nm": {"type": "number", "default": 0.60},
                    },
                },
            },
            {
                "name": "decompose_variance_6_channels",
                "description": "Quantify 18-hole Strokes Gained impact across all 6 putting variability channels.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "distance_ft": {"type": "number", "default": 15.0},
                        "slope_y_pct": {"type": "number", "default": 2.2},
                        "tau_preload_nm": {"type": "number", "default": 0.60},
                    },
                },
            },
            {
                "name": "export_or_load_simulation_json",
                "description": "Export current simulation state to JSON or load a provided JSON payload.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["export", "load"], "default": "export"},
                        "json_payload": {"type": "string"},
                    },
                },
            },
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Dispatch an MCP tool call by name and return structured results."""
        args = arguments or {}
        skill = PlayerSkillProfile(
            tau_preload_nm=float(args.get("tau_preload_nm", 0.60)),
            base_face_sd_deg=float(args.get("base_face_sd_deg", 0.48)),
            green_read_sd_pct=float(args.get("green_read_sd_pct", 0.25)),
            surface_roughness_mm=float(args.get("surface_roughness_mm", 0.35)),
        )
        if name == "run_parallel_putt_batch":
            res = simulate_parallel_putt_batch(
                distance_ft=float(args.get("distance_ft", 12.0)),
                n_trials=int(args.get("n_trials", 250)),
                entry_speed_m_s=float(args.get("entry_speed_m_s", 0.58)),
                slope_y_pct=float(args.get("slope_y_pct", 2.2)),
                stimp_rating=float(args.get("stimp_rating", 11.0)),
                skill=skill,
            )
            self._last_state = res
            return res
        if name == "compute_player_strokes_gained":
            res = optimize_capture_speed_strategy(
                distance_ft=float(args.get("distance_ft", 12.0)),
                slope_y_pct=float(args.get("slope_y_pct", 2.2)),
                skill=skill,
                stimp_rating=float(args.get("stimp_rating", 11.0)),
                n_trials=int(args.get("n_trials", 200)),
            )
            self._last_state = res
            return res
        if name == "run_stimpmeter_green_read":
            prof = GreenUndulationProfile(
                slope_y_pct=float(args.get("slope_y_pct", 2.2)),
                roughness_rms_mm=float(args.get("surface_roughness_mm", 0.35)),
            )
            res = solve_player_optimal_green_read(
                distance_ft=float(args.get("distance_ft", 12.0)),
                stimp_rating=float(args.get("stimp_rating", 11.0)),
                profile=prof,
                skill=skill,
            )
            self._last_state = res
            return res
        if name == "decompose_variance_6_channels":
            res = decompose_putting_variance_channels(
                distance_ft=float(args.get("distance_ft", 15.0)),
                slope_y_pct=float(args.get("slope_y_pct", 2.2)),
                skill=skill,
                n_trials=int(args.get("n_trials", 200)),
            )
            self._last_state = res
            return res
        if name == "export_or_load_simulation_json":
            mode = str(args.get("mode", "export"))
            if mode == "load":
                loaded = load_simulation_state_json(str(args.get("json_payload", "{}")))
                self._last_state = loaded
                return {"status": "loaded", "state": loaded}
            return {"status": "exported", "json": export_simulation_state_json(self._last_state)}
        raise ValueError(f"Unknown MCP tool: {name}")

    def handle_jsonrpc_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a standard JSON-RPC 2.0 / MCP request dictionary."""
        req_id = request.get("id", 1)
        method = request.get("method", "")
        params = request.get("params", {}) or {}
        try:
            if method == "tools/list":
                result: Any = {"tools": self.list_tools()}
            elif method == "tools/call":
                result = self.call_tool(str(params.get("name", "")), params.get("arguments", {}))
            else:
                raise ValueError(f"Unsupported JSON-RPC method: {method}")
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": str(exc)},
            }
