# Pinocchio Engine

## Overview

Pinocchio is a library for fast rigid body dynamics algorithms based on the Featherstone arithmetic. It is particularly efficient for computing kinematics and dynamics derivatives.

## Key Features in Suite

- Extremely fast recursive algorithms (RNEA, ABA, CRBA).
- Python bindings for rapid prototyping.
- Integration with other robotics tools.

## Usage

Located in `src/engines/physics_engines/pinocchio/`.

The engine adapter is `PinocchioPhysicsEngine` in
`src/engines/physics_engines/pinocchio/python/pinocchio_physics_engine.py`.
Launch the Pinocchio dashboard from the repository root:

```bash
python -m src.engines.physics_engines.pinocchio.python
```

## Qualified Optional Motion Runtime

The numerical Pinocchio, Pink, and Crocoddyl integration is an optional Linux
runtime. The portable version manifest is
`scripts/config/motion_runtime/environment.yml`; the exact conda-forge lock
for Linux is `scripts/config/motion_runtime/linux-64.explicit.txt`.

Create the qualified environment with micromamba from this directory:

```bash
micromamba create -y -n upstream-motion-runtime \
  --file scripts/config/motion_runtime/linux-64.explicit.txt
micromamba run -n upstream-motion-runtime \
  python scripts/ci/check_motion_runtime.py --receipt /tmp/motion-runtime-receipt.json
```

The lock contains the conda-forge `linux-64` packages used for qualification,
including the `_x86_64-microarch-level=3` package. It therefore requires an
x86-64-v3 capable CPU. `environment.yml` records package versions without
build strings and is useful when a portable solve is preferred; it does not
provide the same binary reproducibility as the explicit lock.

The checker runs imports and solver probes in fresh subprocesses. The lock
pins Pinocchio 4.1.0, Pink 4.4.0, Crocoddyl 3.2.1, qpsolvers 4.13.0, and
quadprog 0.1.13; the checker verifies that the installed packages import and
that their APIs pass the repository Crocoddyl ABI probe, a Pink hard posture
equality, and Pink rejection of an infeasible equality QP. It records bounded
stdout/stderr, package source hashes, the repository revision, and the numeric
tolerances in its JSON receipt. A passing receipt establishes runtime
capability only; it does not accept a full-body model, fitting result, or
renderer, and cannot by itself prove all native ABI failure modes absent.

Keep these packages in one conda-forge stack. Mixing PyPI cmeel Crocoddyl or
Pinocchio libraries with another Pinocchio ABI can load duplicate native
libraries and cause incorrect derivatives or a process crash. MuJoCo and GUI
dependencies remain outside this optional numerical runtime lock.
