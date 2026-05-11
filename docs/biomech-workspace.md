# Biomechanics Workspace Setup

UpstreamDrift relies on external sibling repositories for biomechanical models:
- `MuJoCo_Models`
- `Drake_Models`
- `OpenSim_Models`
- `Pinocchio_Models`
- `Movement-Optimizer`

## Local Development (Editable Mode)

To set up your workspace for local development across all repositories, clone the sibling repos next to `UpstreamDrift`, then run:

```bash
# In UpstreamDrift root
./scripts/setup_biomech_workspace.sh
```

This will run `pip install -e ../<Repo>` for all sibling repos. UpstreamDrift will automatically resolve models directly from the sibling checkouts using the `model_pack:resolve()` hook.

## CI & Headless (Vendored Mode)

In CI environments, UpstreamDrift resolves models from hermetic snapshots stored in `vendor/biomech-models/`. 

To update a vendored snapshot from a sibling release, use:

```bash
python scripts/update_biomech_vendor.py --repo MuJoCo_Models --ref v1.4.0
```
