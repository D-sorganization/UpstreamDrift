---
issue: 11595
summary: "ci: give every dtolnay/rust-toolchain job a per-job RUSTUP_HOME (${{ github.workspace }}/.rustup-home) so rustup never upgrades the runner image's partially installed ~/.rustup toolchain in place; contract test in tests/ci/test_unit_gate_rust_kernel.py."
---
