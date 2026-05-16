"# Feature Navigation Gap Review: Internal vs. Exposed

## Classification Criteria

**Tile-worthy** (exposed in launcher): Standalone user-facing feature with its own GUI
or meaningful entry point. Users would explicitly seek it out.

**Internal library** (no tile): Underlying module consumed by other features. Users
interact with it indirectly through other tiles or APIs.

**Borderline**: Has some standalone value but is also used by other features. Needs
judgment call on whether the standalone use case justifies a tile.
