# Field Metadata Registry

Source of truth for every user-facing input's label, tooltip, units, valid
range, default, default source, example, and producer/consumer edges.
Consumed by:

- **PyQt6**: `HelpfulField` wrapper in `src/shared/python/ui/helpful_field.py`
  _(forthcoming — Phase 0.2 of epic [#5968](https://github.com/D-sorganization/UpstreamDrift/issues/5968))_.
- **React**: `<HelpfulField>` in `ui/src/components/ux/HelpfulField.tsx`
  _(forthcoming — Phase 0.3)_.
- **Coverage ratchet**: `scripts/ci/check_ux_coverage_ratchet.py` flags
  any bare `QSpinBox`/`QDoubleSpinBox`/`QComboBox`/`QSlider`/`QLineEdit`
  in Python and any bare `<input>`/`<select>`/`<textarea>` in TSX that
  is not wrapped in a `HelpfulField`.

## Why this exists

Today, tooltip copy is scattered across ~376 `setToolTip(...)` call
sites of inconsistent quality. The registry centralises help copy in
one YAML file (`configs/ux/field_metadata.yaml`) that:

1. Non-coders can review and edit.
2. Has a single validation pass (DbC) — every entry has a label, a
   non-empty tooltip, units, a sane default, and an attribution.
3. Drives both desktop and web UIs (DRY across UI frameworks).
4. Declares cross-field dependencies (`consumers` / `producers`) so
   the UI can show "Linked to…" badges (Phase 3).

## Schema

```yaml
fields:
  - id: simulation.timestep # dotted lowercase, immutable once shipped
    label: Timestep # short headline shown next to the input
    short_help: One sentence. # tooltip, <= 80 chars
    long_help: | # popover body, Markdown
      Free-form explanation that may span paragraphs.
    units: s # symbol or null
    valid_range: [1.0e-6, 1.0] # [min, max] for numerics
    #                              # or [enum, values] for enums
    #                              # or null for free-form text
    default: 0.002
    default_source: MuJoCo recommended for humanoid (mujoco docs, 2024).
    consumers: [] # downstream field ids
    producers: [] # upstream field ids
    example: "0.002"
```

See `configs/ux/field_metadata.yaml` for working examples.

## Worked example — adding a new field

Say you add a "max wall-clock budget" input to the simulation page.

1. **Pick a stable id**: `simulation.max_wallclock_s`. Dotted,
   lowercase, snake_case segments. Cannot change once shipped.
2. **Add the YAML entry** to `configs/ux/field_metadata.yaml`:

   ```yaml
   - id: simulation.max_wallclock_s
     label: Max wall-clock
     short_help: Abort the run after this many real-time seconds.
     long_help: |
       The simulation aborts and surfaces `simulation_timeout` (see
       `configs/ux/error_messages.yaml`) if it exceeds this budget.
       0 means no limit.
     units: s
     valid_range: [0.0, 3600.0]
     default: 60.0
     default_source: 1-minute default — matches CI per-test timeout.
     consumers: []
     producers: []
     example: "60.0"
   ```

3. **Run the seed test** to confirm the YAML still parses:

   ```bash
   python3 -m pytest tests/unit/ux/test_seed_configs.py -q
   ```

4. **Wrap the input** in `HelpfulField` (Phase 0.2 / 0.3, forthcoming).
   Until that ships, leaving the input bare is allowed if the
   coverage ratchet baseline still tolerates it (every new bare input
   increments the count, which the ratchet will catch).

## Validation rules

The registry enforces these at load time. If your PR's YAML fails any
of them, the seed-config test fails:

- `id` must match `^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$`.
- `label`, `short_help`, `long_help`, `default_source` are non-empty.
- `short_help` is at most 80 characters.
- Numeric `valid_range` must be ordered (`min <= max`).
- `default` must lie within `valid_range` (numeric) or be a member of
  the enum.
- Every id in `consumers` / `producers` must reference a real field.
- The consumer graph must be acyclic.
- If A lists B as a consumer and B's `producers` is non-empty, B's
  `producers` must include A (symmetry check).

## Cross-references

- Epic: [#5968 — Idiot-Proof UX](https://github.com/D-sorganization/UpstreamDrift/issues/5968)
- Code: `src/shared/python/ux/field_metadata.py`
- YAML: `configs/ux/field_metadata.yaml`
- Tests: `tests/unit/ux/test_field_metadata.py`, `tests/unit/ux/test_seed_configs.py`
- Ratchet: `scripts/ci/check_ux_coverage_ratchet.py`
- Baseline: `scripts/config/ux_field_coverage_baseline.json`
