# Golf Modeling Portfolio Demo

This is the narrow portfolio path for a hiring reviewer who wants one
reproducible golf-modeling story instead of the full multi-engine surface. It
uses the supported MuJoCo/default dependency profile plus the Rust-backed ball
flight kernel used by the current Python flight example. Optional engines such
as Drake, Pinocchio, OpenSim, and MyoSuite are intentionally out of scope.

## What This Demonstrates

- A measured driver launch condition can be turned into a source-bounded ball
  flight calculation.
- The model separates measured inputs, environmental assumptions, and simulated
  outputs.
- The tabular inputs and outputs can be reused by downstream analysis,
  model-comparison, or feature-engineering workflows.
- The result is a physics demonstration, not a validated coaching prescription.

## Setup

From a fresh clone:

```bash
git clone https://github.com/D-sorganization/UpstreamDrift.git
cd UpstreamDrift
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,rust]"
```

If the packaged `upstream_physics` wheel is not available for the local Python
version, build the local binding instead:

```bash
python -m pip install maturin
cd rust_core/upstream-physics
python -m maturin develop --features python
cd ../..
```

## Run

Run the default driver-shot flight example:

```bash
python examples/basic_flight_simulation.py
```

Expected runtime is under 10 seconds on a normal developer laptop after
dependencies are installed. The command uses the Rust-backed trajectory
integrator through `src.shared.python.physics.ball_flight_physics`.

For environments where the Rust extension cannot be installed, use this
lightweight smoke check to verify the documented Python import path:

```bash
python -c "from src.shared.python.physics.ball_flight_physics import BallFlightSimulator, LaunchConditions; print(BallFlightSimulator.__name__, LaunchConditions.__name__)"
```

Expected smoke-check output:

```text
BallFlightSimulator LaunchConditions
```

## Input Fixture

The demo uses one driver-launch condition:

| Quantity      |        Value | Role                     |
| ------------- | -----------: | ------------------------ |
| Ball speed    |     70.0 m/s | Measured input           |
| Launch angle  |     12.0 deg | Measured input           |
| Backspin      |     2700 rpm | Measured input           |
| Air density   | 1.225 kg/m^3 | Environmental assumption |
| Gravity       |   9.81 m/s^2 | Environmental assumption |
| Ball mass     |    0.0459 kg | Custom ball property     |
| Ball diameter |    0.04267 m | Custom ball property     |
| cd0           |         0.25 | Custom ball property     |
| cd1           |          0.0 | Custom ball property     |
| cd2           |          0.0 | Custom ball property     |
| cl0           |         0.15 | Custom ball property     |
| cl1           |          0.0 | Custom ball property     |
| cl2           |          0.0 | Custom ball property     |

These values describe a plausible driver shot. They are not presented as
measured data from a named player, launch monitor, or published study.

## Inspectable Output

The committed reference fixture is
[`golf_modeling_demo_output.csv`](golf_modeling_demo_output.csv).

| Output         |    Reference value | Interpretation                                          |
| -------------- | -----------------: | ------------------------------------------------------- |
| Carry distance | 186.9 m / 204.4 yd | Simulated landing range for the stated launch condition |
| Peak height    |             42.4 m | Simulated apex height                                   |
| Flight time    |              7.6 s | Simulated time aloft                                    |

Reviewers should treat these as model-conditioned outputs and sample fixture
values. They are useful for checking output shape, units, and assumptions before
rerunning the Rust-backed command in a fully provisioned environment; they are
not a claim that the model has been calibrated against TrackMan, radar, or
outdoor range data.

## Troubleshooting

- If `python examples/basic_flight_simulation.py` reports
  `upstream-physics Rust kernel not found`, install the `rust` extra or build
  the local binding with `maturin` as shown above.
- If MuJoCo imports fail on Windows, verify Visual C++ runtime availability and
  use the default `pip install -e ".[dev]"` profile before optional engines.
- Do not install OpenSim or MyoSuite for this demo. They are heavier
  biomechanics integrations and are not required for this portfolio path.

## Limitations

- The demo starts from a prescribed launch condition; it does not infer ball
  launch from a measured full-body swing.
- The reference table is not a validation against measured radar or launch
  monitor data.
- Coaching implications such as "change attack angle" or "change wrist torque"
  would require measured player data and a validated inverse model.
- Optional cross-engine agreement is intentionally excluded to keep the path
  short, reproducible, and honest.
