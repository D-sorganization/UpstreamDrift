---
title: Putting Green miscategorized as physics_engine (should be simulation)
labels: bug, ui, priority/P2
---

## Problem

The Putting Green tile has `category: physics_engine` but it's actually a specialized simulation tool (putting green physics). The `simulation` sidebar category currently shows zero tiles. Moving Putting Green to the `simulation` category would populate it and more accurately reflect its nature.

## Current

```json
{
  ""id"": ""putting_green"",
  ""category"": ""physics_engine"",
  ...
}
```

## Suggested

```json
{
  ""id"": ""putting_green"",
  ""category"": ""simulation"",
  ...
}
```

## Acceptance Criteria

- [ ] Change Putting Green category from `physics_engine` to `simulation`
- [ ] Verify the Simulation sidebar button now shows at least one tile
- [ ] Update the Putting Green description to clarify it's a simulation tool
