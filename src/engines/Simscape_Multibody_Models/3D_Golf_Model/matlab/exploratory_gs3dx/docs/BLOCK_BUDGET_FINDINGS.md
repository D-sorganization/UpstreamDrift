# Block Budget and License Limit Probe Findings

Exploratory Simscape workspace block-budget analysis and Home-license boundary verification for epic [#10950](https://github.com/D-sorganization/UpstreamDrift/issues/10950) (issue [#10953](https://github.com/D-sorganization/UpstreamDrift/issues/10953)).

## Executive Summary

The MATLAB R2025b Home license allows **at most 1,000** nonvirtual blocks, counted exactly as `find_system(...,'Virtual','off')` counts them (each Simscape converter counts as one block). An orchestrator re-run on 2026-09-26 confirmed that 999 Gains + 1 Constant = 1,000 simulate and that 1,001 is rejected at compile time.

Key findings from empirical measurements on `GS3DX_Baseline`:

- **Total Nonvirtual Blocks**: 672
- **Simscape Library Blocks**: 489 (72.8 % of nonvirtual total)
- **Standard Simulink Blocks**: 183 (27.2 % of nonvirtual total)
- **Converter Internal Blocks**: 306 (45.5 % of nonvirtual total; 279 PS-Simulink, 27 Simulink-PS)
- **Top-Level Equivalent Count**: 672
- **Home-License Headroom**: 328 blocks remaining before hitting the 1,000-block ceiling

## Measured Baseline Block Budget

The block budget measured for `GS3DX_Baseline` (derived from `GolfSwing3D_Kinetic.slx`) using `gs3dx_block_budget('GS3DX_Baseline')` is recorded in `block_budget_GS3DX_Baseline.json`.

| Metric                 | Value            | Description                                                                |
| :--------------------- | :--------------- | :------------------------------------------------------------------------- |
| `model`                | `GS3DX_Baseline` | Evaluated top-level exploratory model                                      |
| `nonvirtual_total`     | 672              | Total nonvirtual blocks across top model and referenced subsystems         |
| `converter_internal`   | 306              | Nonvirtual blocks (`PMIOPort`) inside `nesl_utility` converters            |
| `top_level_equivalent` | 672              | Effective nonvirtual block count when each converter is counted as 1 block |
| `simscape_blocks`      | 489              | Nonvirtual blocks from `sm_lib`, `fl_lib`, `nesl_utility`, or `ee_lib`     |
| License Limit          | 1,000            | MathWorks Home-license limit on nonvirtual blocks                          |
| Available Headroom     | 328              | Difference between license limit and baseline count                        |

## Nonvirtual Block-Type Breakdown

The top 20 block types by descending count in `GS3DX_Baseline`:

| Rank | BlockType                | ReferenceBlock                                                          | Count |
| :--- | :----------------------- | :---------------------------------------------------------------------- | :---- |
| 1    | `PMIOPort`               | `nesl_utility/PS-Simulink Converter/input`                              | 279   |
| 2    | `PMIOPort`               | _(Unreferenced port)_                                                   | 93    |
| 3    | `SimscapeMultibodyBlock` | `sm_lib/Frames and Transforms/Transform Sensor`                         | 28    |
| 4    | `PMIOPort`               | `nesl_utility/Simulink-PS Converter/output`                             | 27    |
| 5    | `SimscapeMultibodyBlock` | `sm_lib/Body Elements/Cylindrical Solid`                                | 23    |
| 6    | `SimscapeBlock`          | `fl_lib/Mechanical/Mechanical Sources/Ideal Torque Source`              | 21    |
| 7    | `SimscapeBlock`          | `fl_lib/Mechanical/Multibody Interfaces/Rotational Multibody Interface` | 21    |
| 8    | `SimscapeBlock`          | `fl_lib/Mechanical/Rotational Elements/Mechanical Rotational Reference` | 21    |
| 9    | `Product`                | _(Built-in)_                                                            | 20    |
| 10   | `S-Function`             | _(Built-in)_                                                            | 19    |
| 11   | `SubSystem`              | _(Built-in nonvirtual)_                                                 | 19    |
| 12   | `SimscapeMultibodyBlock` | `sm_lib/Frames and Transforms/World Frame`                              | 16    |
| 13   | `SimscapeMultibodyBlock` | `sm_lib/Body Elements/Inertia Sensor`                                   | 14    |
| 14   | `SimscapeMultibodyBlock` | `sm_lib/Frames and Transforms/Rigid Transform`                          | 13    |
| 15   | `Integrator`             | _(Built-in)_                                                            | 10    |
| 16   | `SimscapeMultibodyBlock` | `sm_lib/Body Elements/Spherical Solid`                                  | 8     |
| 17   | `SimscapeMultibodyBlock` | `sm_lib/Joints/Revolute Joint`                                          | 5     |
| 18   | `SimscapeMultibodyBlock` | `sm_lib/Joints/Universal Joint`                                         | 5     |
| 19   | `Sum`                    | _(Built-in)_                                                            | 5     |
| 20   | `Product`                | `matrix_library/Cross Product/Element Product`                          | 3     |

Notice that the 306 converter internal blocks (279 in rank 1 + 27 in rank 4) account for nearly half of the entire baseline nonvirtual block count.

## Home-License Limit Probe Results

The license limit was experimentally tested using `gs3dx_license_limit_probe.m`. The tool constructs synthetic in-memory models across two suites:

1. Chained Simulink Gain blocks with N = [900, 1000, 1100]
2. Chained Simscape Converter pairs with N = [400, 500, 600] around a `Solver Configuration` block

All synthetic models are created in memory and closed without saving to disk.

| Probe Type      | Parameter N | Nonvirtual Total | Status    | Simulation Duration (s) | Error Message                         |
| :-------------- | :---------- | :--------------- | :-------- | :---------------------- | :------------------------------------ |
| `Gain`          | 900         | 901              | `Success` | 2.65                    | None                                  |
| `Gain`          | 999         | 1000             | `Success` | —                       | None (orchestrator re-run)            |
| `Gain`          | 1000        | 1001             | `Error`   | 0.36                    | Limit exceeded (see exact text below) |
| `Gain`          | 1100        | 1101             | `Error`   | 0.36                    | Limit exceeded (see exact text below) |
| `ConverterPair` | 400         | 802              | `Success` | 3.52                    | None                                  |
| `ConverterPair` | 500         | 1002             | `Error`   | 3.02                    | Limit exceeded (see exact text below) |
| `ConverterPair` | 600         | 1202             | `Error`   | 3.96                    | Limit exceeded (see exact text below) |

_Note on nonvirtual totals_:

- In the Gain probe, the model contains N Gain blocks (nonvirtual) and 1 Constant block (nonvirtual). The Terminator block is virtual (`Virtual = 'on'`), giving `N + 1` nonvirtual blocks. At N = 900, `NonvirtualTotal = 901 <= 1000` (Success). At N = 1000, `NonvirtualTotal = 1001 > 1000` (Error).
- In the Converter probe, each pair contributes 2 nonvirtual blocks (one `PMIOPort` per converter). The model also contains 1 Constant and 1 Solver Configuration block. At N = 400 pairs, `NonvirtualTotal = 802 <= 1000` (Success). At N = 500 pairs, `NonvirtualTotal = 1002 > 1000` (Error).

## Exact License Limit Error Message

When the nonvirtual block count exceeds 1,000, MATLAB R2025b throws the following exact error text upon `sim()`:

```text
Number of blocks in the block diagram '<model_name>' and all models it references exceeds the license limit of 1000 nonvirtual blocks.
```

This error is issued during the compilation phase before numerical integration begins.

## Analysis and Implications for GS3DX Model Lineage

1. **Baseline Viability**:
   `GS3DX_Baseline` compiles and simulates cleanly under the Home license because its 672 nonvirtual blocks sit comfortably below the 1,000-block ceiling (328 blocks headroom).

2. **Full-Body Expansion Risk**:
   Adding a full lower body (pelvis, hips, knees, ankles, feet, contact mechanics) in `GS3DX_FullBody` (#10957, #10958) would easily exceed the 328-block margin if added directly onto the baseline.

3. **Necessity of Slimming Phase**:
   Issue #10954 (`GS3DX_Slim`) must eliminate redundant logging converters and duplicate frames. Because converters alone consume 306 blocks (with 279 dedicated to logging outputs), trimming unused logging channels will reclaim over 200 blocks, expanding headroom to >500 blocks for the subsequent kinematic and dynamic stages.

## Verification and Reproducibility

To re-run the budget evaluation and probe:

```matlab
cd src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx
info = gs3dx_setup();
report = gs3dx_block_budget('GS3DX_Baseline', json_path=fullfile(info.docs_dir, 'block_budget_GS3DX_Baseline.json'));
results = gs3dx_license_limit_probe(verbose=true);
runtests('tests')
```
