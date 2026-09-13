# Saved Replay Velocity Diagnostic

This diagnostic compares the saved run52 scalar trajectory with its finest
manifold trajectory. It performs forward kinematics and coordinate algebra only;
no new integration, optimization, production change or acceptance decision occurs.

## Findings

- Largest native rate error: **0.0046527873 rad/s**, LSInputX at
  **0.7861111111 s**. Scalar rate 397.7982678 rad/s versus manifold 397.7936151 rad/s.
  LSInputZ differs by 0.0046458535 rad/s at the same sample.
- Largest physical frame angular-velocity error: **0.00020800065 rad/s**, frame LS
  at the same time. Scalar angular speed 52.7284394 rad/s; manifold 52.7282385 rad/s.
- Largest physical frame-origin linear-velocity error: **0.00002283562 m/s**,
  frame Hip at 0.85 s, versus scalar speed3.1342888 m/s.
- The left-shoulder middle angle is -1.6509766517 rad, **0.0801803 rad** from its
  nearest pole. The sampled chart condition number is **24.9304**. Both
  trajectories retain the same middle branch.
- The three-component native rate-error norm 0.0065756153 rad/s becomes
  0.00038114869 rad/s under the scalar screw-axis map E. This is an actual
  amplification of **17.2521**, close to the inverse-map bound **17.6426**.

Chart conditioning therefore explains substantial amplification of the native
coordinate error. It does not make the residual physical error disappear or
establish which numerical integration is more accurate. These are sampled
360-Hz diagnostics, not bounds on conditioning between samples.

The script independently verifies the exact decomposition
`E_scalar * delta_v = delta_omega - (E_manifold - E_scalar) * v_manifold`.
At the peak, relative joint angular-velocity difference norm is 0.00021630958
rad/s and the configuration-dependent map term is 0.00032579549 rad/s. Small
configuration differences acting on very large native rates therefore matter
alongside inverse-chart amplification. Do not diagnose a new branch flip merely
from the larger native-rate discrepancy.

Keep existing acceptance gates. Establish scalar-reference convergence and then
refine adaptive manifold tolerances while measuring both native and physical
velocity discrepancies. The current report does not justify waiving a gate or
substituting physical-rate agreement for the required native-rate agreement.

## Reproduction

`diagnostic.json` records input and script SHA-256 values, Pinocchio 4.1.0, every
coordinate's peak, physical velocity peaks, and full group conditioning/error
series. Physical twists are world-aligned, evaluated at corresponding native
frame origins; components are linear then angular. The source data remain in
root's immutable run52 archive.

```powershell
scp inspect_velocity.py controltower:/C:/Users/diete/inspect_manifold_velocity_10043_17.py
ssh -o BatchMode=yes controltower 'wsl -d ControlTower-Runner -- env PYTHONPATH=/home/dieterolson/native-manifold-10043-16 /home/dieterolson/simscape-pinocchio-9967/.venv/bin/python /mnt/c/Users/diete/inspect_manifold_velocity_10043_17.py --model /mnt/c/Users/diete/native_geometry_spec_9967.json --candidate /mnt/c/Users/diete/native-ms-fit-9967-19/returned-candidate.json --scalar /mnt/c/Users/diete/native-manifold-replay-9967-52/scalar.npz --manifold /mnt/c/Users/diete/native-manifold-replay-9967-52/manifold-1.npz --output /mnt/c/Users/diete/native-manifold-velocity-repeat.json'
```

Run the scp command from this artifact directory, or supply the full local script
path. The source archive preserves exact script/report bytes before formatting.
Ruff passes. The decomposition assertion passed at all 307 samples for all three
rotational groups. No production files were modified for this diagnostic.
