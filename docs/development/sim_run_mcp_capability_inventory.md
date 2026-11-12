# Ud-Sim-MCP Capability Inventory

<!-- Generated: python3 -m scripts.generate_sim_run_mcp_inventory -->

Child task of [#15020](https://github.com/D-sorganization/UpstreamDrift/issues/15020).
Statuses are **`available`** or **`unavailable`** only; nothing is labeled failed.

- Server: `ud-sim-mcp`
- Schema: `ud-sim-mcp-capability-inventory/v1`

## Authentication Environment

- Canonical token: `UPSTREAMDRIFT_AGENT_API_TOKEN` (alias: `UD_SIM_AUTH_TOKEN`)
- Canonical Private Mode: `UPSTREAMDRIFT_PRIVATE_MODE` (alias: `UD_PRIVATE_MODE`)

Listener authentication hardening is tracked in private Repository_Management
records; public docs do not describe that surface.

## Simulation Engines

| Engine | Description | Runtime Status | Unavailable Reason |
| :--- | :--- | :---: | :--- |
| `pendulum` | Double pendulum dynamics benchmark | available | — |
| `mujoco` | MuJoCo physics engine | available | — |
| `pinocchio` | Pinocchio rigid body algorithms | unavailable | optional engine dependency not installed (No module named 'pinocchio') |
| `drake` | Drake multibody plant simulation | unavailable | optional engine dependency not installed (No module named 'pydrake') |
| `opensim` | OpenSim musculoskeletal biomechanics | unavailable | optional engine dependency not installed (No module named 'opensim') |

## Putting Tools

| Tool | Runtime Status | Unavailable Reason |
| :--- | :---: | :--- |
| `optimize_putt` | available | — |
| `putting_misread_cost` | available | — |
| `putting_library_study` | available | — |
| `putting_variance_attribution` | available | — |
| `putting_simulate_putt` | available | — |
| `putting_read_green` | available | — |
| `putting_stimpmeter` | available | — |
| `putting_capture_speed` | available | — |

## Agent Capabilities

| Tool | Runtime Status | Unavailable Reason |
| :--- | :---: | :--- |
| `run_counterfactual_ztcf_golden` | available | — |
| `run_golf_simulator_golden_shot` | available | — |
