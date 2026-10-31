# Developer & Agent Turnover Specification — 3D Putting Dynamics & Undulating Green Roll Suite (v4.0)

## 1. UpstreamDrift Repository Layout (`upstreamdrift_putting_v4_package.zip`)
```text
src/shared/python/physics/putting/
├── __init__.py                        # Public package facade
├── turf_properties.py                 # Zero-allocation @precondition (D-11803), USGA friction, Holmes/Penner capture
├── grip_nullspace_stabilization.py    # 4-point grasp matrix A_couple, Moore-Penrose u_task, Null-Space u_null, ZTCF/ZVCF
├── green_surface_undulation.py        # Non-planar multi-tier green z(x,y), analytical grad_z(x,y), turf micro-roughness
├── strokes_gained_analytics.py        # 100/1000-putt batch engine, Bellman E_player(d) (0-45 ft), capture speed optimizer, 6-channel variance
├── stimpmeter_reader.py               # Virtual USGA Stimpmeter V-groove ramp rollout & optimal player green-read solver
├── putting_mcp_server.py              # MCP & JSON-RPC 2.0 server (PuttingMCPServer)
└── putting_dynamics_v3.py             # Unified engine facade
tests/unit/physics/putting/
├── test_putting_ci_suite.py           # 6 CI/AST/MCP verification tests (Ran 6 tests — OK)
├── test_putting_dynamics_suite.py     # 15 core physics verification checks (All 15 passed)
└── test_putting_dynamics_v3.py        # 28 comprehensive TDD/DbC/LoD/DRY unit tests (Ran 28 tests — OK)
```

## 2. Programmatic MCP / JSON-RPC 2.0 & Browser API Usage
### Browser Studio (`window.PuttingSimulatorAPI`)
```javascript
// 1. Run 1,000 parallel putts with custom preload, distance, roughness, and green-reading error
window.PuttingSimulatorAPI.setState({
  puttDistFt: 15.0,
  tauPreload: 0.60,
  captureSpeedMs: 0.65,
  surfaceRoughnessMm: 0.35,
  greenReadErrorPct: 0.25
});
const stats = window.PuttingSimulatorAPI.runBatch(1000);

// 2. Dispatch via JSON-RPC 2.0 / MCP protocol
const resp = window.PuttingSimulatorAPI.handleJsonRpc({
  jsonrpc: "2.0",
  id: 1,
  method: "tools/call",
  params: { name: "run_stimpmeter_calibration", arguments: {} }
});

// 3. Export or Load Simulation JSON
const savedJson = window.PuttingSimulatorAPI.exportResultsJSON();
window.PuttingSimulatorAPI.loadResultsJSON(savedJson);
```

### Python MCP Server (`PuttingMCPServer`)
```python
from putting_mcp_server import PuttingMCPServer

server = PuttingMCPServer()
tools = server.list_tools()
response = server.handle_jsonrpc_request({
    "jsonrpc": "2.0",
    "id": 101,
    "method": "tools/call",
    "params": {
        "name": "compute_player_strokes_gained",
        "arguments": {"distance_ft": 12.0, "slope_y_pct": 2.2, "tau_preload_nm": 0.60}
    }
})
```
