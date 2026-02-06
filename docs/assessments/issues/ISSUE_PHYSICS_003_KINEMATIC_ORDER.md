---
type: physics-risk
severity: high
status: open
created: 2026-02-05
file: src/shared/python/kinematic_sequence.py
---

# High: Hardcoded Kinematic Sequence Order

## Description

The `KinematicSequenceAnalyzer` hardcodes the expected segment order as `['Pelvis', 'Torso', 'Arm', 'Club']`.

```python
self.expected_order = expected_order or ["Pelvis", "Torso", "Arm", "Club"]
```

This poses two risks:

1.  **Scientific Validity:** Not all effective swings follow this specific proximal-to-distal order (e.g., short game shots, specific elite variations). Hardcoding it limits the tool's diagnostic utility.
2.  **Legal/IP:** This specific sequence methodology is closely associated with TPI (Titleist Performance Institute) patents. Hardcoding it into the core logic may invite infringement claims if used commercially.

## Expected Behavior

The analyzer should be agnostic to the segments and their order, allowing the user to define the expected sequence via configuration.

## Recommended Fix

1.  Remove the default hardcoded list. Require `expected_order` to be passed during initialization or method call.
2.  Refactor `analyze` to accept an arbitrary list of segments to check.
3.  Rename metrics like "Sequence Consistency" to more generic statistical terms if necessary to avoid trademark issues.
