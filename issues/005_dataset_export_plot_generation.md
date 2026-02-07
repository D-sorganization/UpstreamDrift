# Issue: Dataset Export and Plot Generation Tab

## Summary
Create integrated dataset export functionality with a plot generation system for all
simulation data. This integrates the data processor from the shared tools folder into
the main workflow, providing a unified tab/interface for exporting simulation data
and generating standard plot sets.

## Motivation
Users need to export simulation datasets in multiple formats (CSV, HDF5, MAT, C3D)
and generate standardized plots (kinematics, kinetics, energy, phase portraits) after
each simulation. The existing `data_explorer_app.py` and `export.py` modules provide
building blocks but are not integrated into the main workflow.

## Requirements
- [ ] Unified export interface supporting CSV, JSON, HDF5, MAT, C3D
- [ ] Standard plot sets: kinematics, kinetics, energy, phase portrait, comparison
- [ ] Configurable plot generation (select which plots to create)
- [ ] Batch export for dataset generator outputs
- [ ] API endpoints for export and plot generation
- [ ] Integration with `data_explorer_app.py` data discovery
- [ ] Plot templates for common analysis scenarios

## Acceptance Criteria
- Export works for all supported formats
- Standard plot sets generated correctly
- API endpoints functional
- Integrates with dataset generator
- Tests for export and plot generation

## Labels
`enhancement`, `data-processing`, `visualization`, `shared`
