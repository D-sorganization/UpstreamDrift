# Quick Reference: Which GUI Should I Use?

## Decision Tree

```
Do you need to customize the humanoid's appearance (colors, size)?
├─ YES → Use Humanoid Golf Launcher (humanoid_launcher.py)
│         - Color customization ✅
│         - Height/weight scaling ✅
│         - Equipment parameters ✅
│
└─ NO → Do you need to edit poses or analyze multiple models?
    ├─ YES → Use Advanced Golf Analysis Window (python -m mujoco_humanoid_golf)
    │         - Interactive pose editing ✅
    │         - Drag-and-drop manipulation ✅
    │         - Multiple models ✅
    │         - Biomechanical analysis ✅
    │
    └─ NO → Do you need batch processing?
        └─ YES → Use CLI Runner (python -m mujoco_humanoid_golf.cli_runner)
                  - Headless batch processing ✅
                  - Command-line automation ✅
```

## Common Workflows

### Workflow 1: Create Custom Humanoid Simulation
1. Launch `humanoid_launcher.py`
2. Customize colors in "Appearance" tab
3. Set height/weight in "Appearance" tab
4. Configure club in "Equipment" tab
5. Choose control mode in "Simulation" tab
6. Run simulation (headless or live view)
7. Open generated video/data

### Workflow 2: Interactive Pose Design
1. Launch `python -m mujoco_humanoid_golf`
2. Select desired model from dropdown
3. Adjust joint angles with sliders OR
4. Enable drag mode and click-drag body segments
5. Save state when satisfied
6. Use saved state in future simulations

### Workflow 3: Batch Analysis
1. Create control configuration JSON
2. Run `python -m mujoco_humanoid_golf.cli_runner --model full_body --control-config config.json`
3. Analyze output CSV/JSON files

## File Locations

### Humanoid Launcher
- **GUI**: `engines/physics_engines/mujoco/python/humanoid_launcher.py`
- **Config**: `engines/physics_engines/mujoco/docker/src/simulation_config.json`
- **Outputs**: `engines/physics_engines/mujoco/docker/src/humanoid_golf.mp4` and `golf_data.csv`

### Advanced Analysis
- **GUI**: `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/advanced_gui.py`
- **Entry**: `python -m mujoco_humanoid_golf`
- **States**: User-specified `.pkl` files

### CLI Runner
- **Module**: `engines/physics_engines/mujoco/python/mujoco_humanoid_golf/cli_runner.py`
- **Entry**: `python -m mujoco_humanoid_golf.cli_runner`
- **Outputs**: User-specified paths

## Key Differences

| Need | Humanoid Launcher | Advanced Analysis |
|------|------------------|-------------------|
| Change body colors | ✅ | ❌ |
| Change height/weight | ✅ | ❌ |
| Edit poses interactively | ❌ | ✅ |
| Drag body segments | ❌ | ✅ |
| Multiple models | ❌ | ✅ |
| Docker-based | ✅ | ❌ |
| Biomechanical metrics | ❌ | ✅ |

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| Segmentation fault | Uncheck "Live Interactive View" |
| Pixelated display | Already fixed in latest version |
| Colors not applied | Use Humanoid Launcher, not Advanced Analysis |
| Can't find output files | Check `docker/src/` directory |
| Docker build fails | Click "UPDATE ENV" button |

---

**For full documentation, see [GUI_ARCHITECTURE.md](./GUI_ARCHITECTURE.md)**
