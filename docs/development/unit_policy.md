# Fleet Unit Policy for Tool Development

> **Scope.** All tools under `src/tools/` modeling physical chains in the Golf Modeling Suite must adhere to this unit policy (issue #8886).

---

## 1. Core Principles

1. **Strict SI Computation Everywhere:**
   All internal physical simulations, models, estimators, physics engines, and kinematics layers must store, compute, and exchange state strictly in SI units:

   - Distance/Position: meters ($m$)
   - Speed/Velocity: meters per second ($m/s$)
   - Mass: kilograms ($kg$)
   - Angle/Rotation: radians ($rad$) (or degrees where explicit in APIs, e.g. loft angles)
   - Force: Newtons ($N$)
   - Torque: Newton-meters ($N\cdot m$)

2. **Display-Layer Separation (LoD):**
   Tools must never force internal simulation math to run in Imperial units. Display-level conversions (e.g. $mph$, $yd$, $ft$, $lb$, $deg$) belong strictly in UI controls and result formatters.

3. **Single Source of Truth:**
   Use the shared conversion helpers in `src/shared/python/ui/units.py`:
   - `UnitSystem`: enum with `METRIC` and `IMPERIAL`.
   - `get_unit_preference()` / `set_unit_preference()`: reads/writes the user preference from `UserPreferences.unit_system` (persisted in `preferences.json`).
   - `to_display_distance`, `from_display_distance`, `distance_suffix`, `format_distance`
   - `to_display_speed`, `from_display_speed`, `speed_suffix`, `format_speed`
   - `to_display_mass`, `from_display_mass`, `mass_suffix`
   - `format_spin`

---

## 2. Widget Requirements for Tool Authors

When building or refactoring a tool widget:

1. **Unit-System Awareness:**
   The widget constructor should accept an optional `unit_system: UnitSystem | None = None` keyword argument, falling back to `get_unit_preference()` when omitted:

   ```python
   class MyToolWidget(QWidget):
       def __init__(self, parent: QWidget | None = None, *, unit_system: UnitSystem | None = None) -> None:
           super().__init__(parent)
           self._unit_system = unit_system if unit_system is not None else get_unit_preference()
           self._build_ui()
   ```

2. **Dynamic Switching:**
   Implement `unit_system` property and `set_unit_system(self, system: UnitSystem) -> None`. When switching:

   - Read current value in old unit system and convert to SI.
   - Set new ranges, suffixes, and decimals on spinboxes.
   - Convert SI value into the new display system and update spinboxes.
   - Re-render any active result panels.

3. **Honest Tooltips and Labels:**
   Tooltips and accessibility descriptions must update whenever display units or angle modes change (e.g. Pose Studio updating slider tooltips between `deg` and `rad`).

4. **Result Panes (No Mixed Systems Without Dual Display):**
   No result pane may display mixed systems inconsistently. When displaying metrics:
   - The user's active unit system is shown as the primary readout.
   - The alternate system is shown in parentheses as a secondary readout:
     - Metric mode: `220.0 m (240.6 yd)`, `70.0 m/s (156.6 mph)`
     - Imperial mode: `240.6 yd (220.0 m)`, `156.6 mph (70.0 m/s)`
   - Headline provenance labels (`ProvenanceValueLabel`) reflect the user's active display units.
