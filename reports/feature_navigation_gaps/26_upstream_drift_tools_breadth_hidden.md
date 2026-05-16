---
title: UpstreamDrift Tools calculator suite breadth is hidden
labels: feature, priority/P3
---

## Problem

The UpstreamDrift Tools (`src/shared/python/upstream_drift_tools/`) contains a comprehensive process-engineering calculator suite:

- **Process calculators**: pressure drop, syngas compression, scrubber, PSA, acid gas dewpoint, flare, baghouse, WGS reactor, electrode advancement
- **Thermo**: steam tables, properties, vapor pressure
- **Mechanical**: pipe database, fittings, friction factors
- **Electrical**: electrical model
- **Data processing**: operations, widget
- **Unit conversion**: widget and preferences manager
- **Lab UI**: full workspace with file navigation, command history, tab system, reporting

Only the `Data Processor` tile surfaces this suite. The calculator breadth is invisible.

## Classification

**Borderline**: The calculators are domain-specific (process engineering) and may not match the core biomechanics audience. However, they are fully functional tools.

## Acceptance Criteria

- [ ] Expand the `Data Processor` tile description to mention calculators
- [ ] Consider adding an `upstream_drift_tools` category or expanding the `tool` category description
- [ ] Add `process_calculators` to the Data Processor tile `capabilities` array
- [ ] Low priority: could add individual calculator tiles if demand exists
