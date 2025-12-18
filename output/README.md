# Golf Modeling Suite Output Directory

This directory contains all simulation results, analysis outputs, and generated reports from the Golf Modeling Suite.

## Directory Structure

```
output/
├── simulations/          # Raw simulation results
│   ├── mujoco/          # MuJoCo engine results
│   ├── drake/           # Drake engine results
│   ├── pinocchio/       # Pinocchio engine results
│   └── matlab/          # MATLAB Simscape results
├── analysis/            # Processed analysis results
│   ├── biomechanics/    # Biomechanical analysis
│   ├── trajectories/    # Ball trajectory analysis
│   ├── optimization/    # Swing optimization results
│   └── comparisons/     # Cross-engine comparisons
├── exports/             # Exported data and media
│   ├── videos/          # Simulation videos
│   ├── images/          # Plots and visualizations
│   ├── data/           # Exported datasets (CSV, JSON, HDF5)
│   └── c3d/            # Motion capture exports
├── reports/             # Generated reports
│   ├── pdf/            # PDF reports
│   ├── html/           # HTML reports
│   └── presentations/   # Presentation materials
└── cache/              # Temporary and cache files
    ├── models/         # Cached model files
    ├── computations/   # Cached computation results
    └── temp/           # Temporary files
```

## File Naming Conventions

### Simulation Results
- Format: `{engine}_{model}_{timestamp}_{parameters}.{ext}`
- Example: `mujoco_golf_swing_20241218_120000_speed100mph.csv`

### Analysis Results
- Format: `analysis_{type}_{timestamp}_{description}.{ext}`
- Example: `analysis_biomechanics_20241218_120000_muscle_activation.json`

### Exports
- Videos: `{engine}_{description}_{timestamp}.mp4`
- Images: `{type}_{description}_{timestamp}.png`
- Data: `export_{format}_{timestamp}_{description}.{ext}`

### Reports
- Format: `report_{type}_{timestamp}_{description}.{ext}`
- Example: `report_swing_analysis_20241218_120000_optimization_study.pdf`

## Data Formats

### Supported Formats
- **CSV**: Tabular data, time series
- **JSON**: Metadata, configuration, structured results
- **HDF5**: Large datasets, hierarchical data
- **Pickle**: Python objects, complex data structures
- **C3D**: Motion capture data
- **MP4**: Video exports
- **PNG/JPG**: Images and plots
- **PDF**: Reports and documentation

### Data Schema
All simulation results follow a standardized schema:

```json
{
  "metadata": {
    "engine": "mujoco|drake|pinocchio|matlab",
    "model": "golf_swing|pendulum|biomechanical",
    "timestamp": "ISO 8601 format",
    "version": "suite version",
    "parameters": {...}
  },
  "results": {
    "ball_trajectory": [...],
    "club_motion": [...],
    "biomechanics": {...},
    "performance_metrics": {...}
  }
}
```

## Usage Examples

### Accessing Results Programmatically

```python
from golf_modeling_suite.output import OutputManager

# Initialize output manager
output = OutputManager()

# List available simulations
simulations = output.get_simulation_list()

# Load specific simulation
results = output.load_simulation("mujoco_golf_swing_20241218_120000.csv")

# Export analysis
output.export_analysis_report(analysis_data, "swing_optimization")
```

### Command Line Access

```bash
# List recent simulations
golf-suite output list --recent 10

# Export simulation data
golf-suite output export --simulation sim_001 --format csv

# Generate report
golf-suite output report --type biomechanics --output pdf
```

## Cleanup and Maintenance

### Automatic Cleanup
- Temporary files are cleaned automatically after 24 hours
- Cache files are cleaned when they exceed 1GB total size
- Old simulation results are archived after 30 days (configurable)

### Manual Cleanup
```python
from golf_modeling_suite.output import OutputManager

output = OutputManager()

# Clean old files (older than 30 days)
output.cleanup_old_files(max_age_days=30)

# Clear cache
output.clear_cache()

# Archive old results
output.archive_old_results(archive_path="/path/to/archive")
```

## Configuration

Output behavior can be configured in `shared/python/config/output_config.yaml`:

```yaml
output:
  base_directory: "output"
  auto_cleanup: true
  max_cache_size_gb: 1.0
  archive_after_days: 30
  default_formats:
    simulation: "csv"
    analysis: "json"
    export: "hdf5"
  compression:
    enabled: true
    level: 6
```

## Best Practices

1. **Organize by Project**: Create subdirectories for different research projects
2. **Use Descriptive Names**: Include key parameters in filenames
3. **Regular Cleanup**: Archive or delete old results regularly
4. **Backup Important Results**: Keep copies of significant findings
5. **Document Analysis**: Include metadata and documentation with results
6. **Version Control**: Track analysis scripts and configurations separately

## Troubleshooting

### Common Issues

**Disk Space**: Monitor output directory size, enable auto-cleanup
```bash
du -sh output/
```

**Permission Errors**: Ensure write permissions to output directory
```bash
chmod -R 755 output/
```

**Corrupted Files**: Use validation tools to check file integrity
```python
output.validate_simulation_file("simulation.csv")
```

**Missing Results**: Check simulation logs and error messages
```python
output.get_simulation_logs("simulation_id")
```