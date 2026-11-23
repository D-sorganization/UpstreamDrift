# Bracketed Contact Root Kernel

This research kernel refines a supplied contact bracket against an integrated
state callback; see the [calculation note](contact_root.tex). Its MuJoCo test
uses a simplified sliding rectangular face and a fixed sphere. It does not
certify a best-match swing, a first contact, a complete search, or an impact.

To use it with the driver, the caller must supply the actual finite clubface
surface in a consistent outward orientation, sphere clearance, strict interior
and front-side classification, point velocity, and re-evaluation of the same
continuous production trajectory. Fixed actuator holds must not move when the
bracket is refined. A sign transition can be caused by a discontinuity;
residual and approach checks do not prove the callback is continuous. Scans can
miss brief or tangent contact. The kernel does not provide the smooth miss
penalty or claim any anatomical, speed, torque, or physical-acceptance gates.

Reproduce headlessly with `python -m pytest -q
 tests/unit/motion_matching/test_contact_root.py` after setting
`QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg`, and `MUJOCO_GL=disable`.
The finite-face fixture uses MuJoCo when installed and skips otherwise.
