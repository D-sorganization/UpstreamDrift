Add an F09 native MyoSuite muscle-excitation consumer for versioned T01 bundles. The adapter applies exact normalized excitation directly to native MuJoCo controls with full state restore, explicit frame-skip/ZOH timing, strict provider/model identity, and no Gym observation or feedback path. The actual elbow fixture remains test-only; required driver/iron rows and the six-engine parity denominator stay unqualified.

Wrapper-state capture and restoration share one chain-walk implementation,
with focused coverage for the supported wrapper order and state values.
