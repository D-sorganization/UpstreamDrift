## Usage Guide - Optimized Golf Swing Analysis System

### Table of Contents

1. [Getting Started](#getting-started)
2. [Running Your First Analysis](#running-your-first-analysis)
3. [Understanding the Output](#understanding-the-output)
4. [Advanced Configuration](#advanced-configuration)
5. [Performance Tuning](#performance-tuning)
6. [Common Workflows](#common-workflows)

---

## Getting Started

### Prerequisites

Before running the analysis, ensure you have:

1. ✅ MATLAB R2020a or later
2. ✅ Simulink installed
3. ✅ Original GolfSwing model accessible
4. ✅ (Optional) Parallel Computing Toolbox for speedup

### Installation

No installation required! Simply navigate to the directory:

```matlab
cd /path/to/2D_Golf_Model/matlab_optimized
```

---

## Running Your First Analysis

### Option 1: GUI (Easiest)

```matlab
% From matlab_optimized directory
launch_gui()
```

Then:

1. Check desired options (Parallel Processing, Checkpoints, Generate Plots)
2. Click "▶️ RUN COMPLETE ANALYSIS"
3. Watch progress in status log
4. View results in Results tab

### Option 2: Command Line (Fastest)

```matlab
% From matlab_optimized directory
addpath(genpath(pwd))
[BASE, ZTCF, DELTA, ZVCF] = run_analysis();
```

This will:

- Use parallel processing (if available)
- Enable checkpointing
- Generate all plots
- Save all data tables

### Option 3: Custom Configuration

```matlab
[BASE, ZTCF, DELTA, ZVCF] = run_analysis(...
    'use_parallel', true, ...      % Enable/disable parallelization
    'use_checkpoints', true, ...   % Enable/disable checkpointing
    'generate_plots', false, ...   % Skip plot generation for speed
    'verbose', true);              % Show detailed progress
```

---

## Understanding the Output

### Console Output

You'll see structured output like:

```
╔══════════════════════════════════════════════════════════════════╗
║   OPTIMIZED 2D GOLF SWING ANALYSIS SYSTEM                        ║
║   Zero Torque Counterfactual (ZTCF) Analysis Pipeline           ║
╚══════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1: Model Initialization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 Initializing Simulink model...
✅ Model initialized successfully

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2: Base Data Generation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Generating baseline simulation data...
✅ Baseline data generated: 2800 rows, 186 variables
   ⏱️  Stage completed in 3.45 seconds

... and so on
```

### Data Tables

After completion, you'll have these tables in memory and saved to `data/output/`:

#### BASE Tables

- **BASE**: Full resolution (0.0001s timestep, ~2800 rows)
- **BASEQ**: Plot resolution (0.0025s timestep, ~112 rows)
- Contains: All forces, torques, work, power, impulse from complete simulation

#### ZTCF Tables

- **ZTCF**: First retained sample at joint-torque removal
- **ZTCFQ**: Plot resolution
- Retains gravity, motion-dependent dynamics, and modeled compliance at the sampled state; it is not a physiological "passive forces only" label

#### DELTA Tables

- **DELTA = BASE - ZTCF**: Algebraic table residual
- **DELTAQ**: Plot resolution
- Does not isolate muscle-generated torque or additive power because force and velocity columns are both subtracted

#### ZVCF Tables

- **ZVCFTable**: Static pose analysis (all velocities = 0)
- **ZVCFTableQ**: Plot resolution
- Shows the configured zero-velocity response; gravity inclusion depends on the model setting, so it is not automatically a pure torque term

### Table Structure

Each table has ~186+ variables including:

**Kinematics:**

- Time
- Positions (x, y, z for all joints)
- Velocities (linear and angular)
- Accelerations

**Forces/Torques:**

- Joint forces (LS, RS, LE, RE, LW, RW)
- Total hand forces
- Equivalent couples
- Net forces

**Calculated Quantities:**

- Angular/Linear Work
- Angular/Linear Power
- Angular/Linear Impulse
- Total Work/Power
- Stored table totals; fractional contribution ratios are not implemented by `calculate_total_work_power.m`

### Generated Plots

If plot generation is enabled, you'll find plots in `data/plots/`:

```
data/plots/
├── BASE_Charts/
│   ├── BASE_Plot_Angular_Work.fig
│   ├── BASE_Plot_Angular_Work.png
│   ├── BASE_Plot_Linear_Work.fig
│   └── ...
├── ZTCF_Charts/
│   └── ...
├── DELTA_Charts/
│   └── ...
└── ZVCF_Charts/
    └── ...
```

---

## Advanced Configuration

### Modifying Simulation Parameters

Edit `config/simulation_config.m`:

```matlab
% Change simulation duration
config.stop_time = 0.30;  % Default: 0.28

% Change ZTCF resolution
config.ztcf_end_time = 50;     % More time points
config.ztcf_time_scale = 200;  % Finer resolution

% Change data resolution
config.base_sample_time = 0.00005;  % Higher resolution
config.q_sample_time = 0.001;       % Finer Q-table resolution
```

### Modifying Plot Appearance

Edit `config/plot_config.m`:

```matlab
% Change figure size
config.figure_width = 1200;
config.figure_height = 800;

% Change line properties
config.line_width = 2.0;
config.font_size = 14;

% Change colors
config.colors.LS = [1, 0, 0];  % Red for left shoulder

% Change export formats
config.export_formats = {'fig', 'png', 'pdf'};
```

### Custom Output Directories

```matlab
config = simulation_config();
config.output_path = '/path/to/my/data';
config.plots_path = '/path/to/my/plots';
config.cache_path = '/path/to/my/cache';
```

---

## Performance Tuning

### Maximizing Speed

```matlab
% Run with parallelization, no plots
[BASE, ZTCF, DELTA, ZVCF] = run_analysis(...
    'use_parallel', true, ...
    'generate_plots', false, ...
    'use_checkpoints', false);  % Skip checkpointing overhead
```

**Expected runtime:** ~1 minute

### Maximizing Reliability

```matlab
% Run with checkpointing, serial mode (more stable)
[BASE, ZTCF, DELTA, ZVCF] = run_analysis(...
    'use_parallel', false, ...
    'use_checkpoints', true);
```

**Expected runtime:** ~3-4 minutes

### Parallel Pool Configuration

```matlab
% Use specific number of workers
config = simulation_config();
config.num_workers = 4;  % Use 4 cores

% Or auto-detect
config.num_workers = [];  % Use all available cores
```

### Memory Optimization

For large datasets or limited RAM:

```matlab
% Reduce ZTCF resolution
config = simulation_config();
config.ztcf_end_time = 14;  % Fewer time points
config.ztcf_time_scale = 50;

% Skip plot generation initially
run_analysis('generate_plots', false);

% Generate plots later if needed
load('data/output/BASEQ.mat');
load('data/output/ZTCFQ.mat');
% ... generate specific plots
```

---

## Common Workflows

### Workflow 1: Quick Data Generation

For fast iteration during model development:

```matlab
% Skip plots, use parallelization
[BASE, ZTCF, DELTA, ~] = run_analysis(...
    'use_parallel', true, ...
    'generate_plots', false);

% Analyze specific quantities
plot(BASE.Time, BASE.LWAngularWorkonClub);
title('Left Wrist Angular Work');
```

### Workflow 2: Full Analysis for Publication

For complete results with all plots:

```matlab
% Run complete analysis
run_analysis(...
    'use_parallel', true, ...
    'use_checkpoints', true, ...
    'generate_plots', true);

% All data and plots will be in:
% - data/output/ (tables)
% - data/plots/ (figures)
```

### Workflow 3: Resuming After Interruption

If analysis was interrupted:

```matlab
% Simply run again with checkpoints enabled
[BASE, ZTCF, DELTA, ZVCF] = run_analysis('use_checkpoints', true);
% Will resume from last completed stage
```

### Workflow 4: Parameter Sweep

Testing different model parameters:

```matlab
% Modify model parameter
config = simulation_config();
% ... modify config ...

% Run analysis
[BASE1, ZTCF1, ~, ~] = run_analysis('generate_plots', false);

% Change parameter
% ... modify again ...

% Run again
[BASE2, ZTCF2, ~, ~] = run_analysis('generate_plots', false);

% Compare
figure; hold on;
plot(BASE1.Time, BASE1.LWAngularWorkonClub, 'b-');
plot(BASE2.Time, BASE2.LWAngularWorkonClub, 'r--');
legend('Config 1', 'Config 2');
```

### Workflow 5: Custom Post-Processing

Using the generated tables for custom analysis:

```matlab
% Run analysis
[BASE, ZTCF, DELTA, ZVCF] = run_analysis();

% Custom analysis
% Example: Find time of maximum club head speed
[max_CHS, idx] = max(BASE.ClubHeadSpeed);
time_at_max = BASE.Time(idx);

fprintf('Max club head speed: %.2f m/s at %.3f seconds\n', ...
    max_CHS, time_at_max);

% Custom plot
figure;
subplot(2,1,1);
plot(BASE.Time, BASE.ClubHeadSpeed);
title('Club Head Speed');
hold on;
plot(time_at_max, max_CHS, 'ro', 'MarkerSize', 10);

subplot(2,1,2);
plot(BASE.Time, BASE.LWAngularWorkonClub + BASE.RWAngularWorkonClub);
title('Total Wrist Angular Work');
xline(time_at_max, 'r--', 'Max CHS');
```

---

## Tips and Tricks

### Viewing Table Contents

```matlab
% Load a table
load('data/output/BASE.mat');

% View variable names
BASE.Properties.VariableNames

% View first few rows
head(BASE, 10)

% Access specific variable
BASE.LWAngularWorkonClub

% Find specific time
idx = find(BASE.Time >= 0.15, 1);
BASE(idx, :)  % Row at ~0.15 seconds
```

### Comparing Datasets

```matlab
load('data/output/BASEQ.mat');
load('data/output/ZTCFQ.mat');
load('data/output/DELTAQ.mat');

figure;
hold on;
plot(BASEQ.Time, BASEQ.LWAngularWorkonClub, 'k-', 'LineWidth', 2);
plot(ZTCFQ.Time, ZTCFQ.LWAngularWorkonClub, 'g--', 'LineWidth', 2);
plot(DELTAQ.Time, DELTAQ.LWAngularWorkonClub, 'r:', 'LineWidth', 2);
legend('BASE (Total)', 'ZTCF (Passive)', 'DELTA (Active)');
grid on;
title('Decomposition of Left Wrist Angular Work');
```

### Batch Processing

```matlab
% Run multiple analyses with different configurations
configs = {'Config1', 'Config2', 'Config3'};
results = cell(length(configs), 1);

for i = 1:length(configs)
    fprintf('Running configuration: %s\n', configs{i});

    % Modify config for this run
    % ... your configuration changes ...

    % Run analysis
    [BASE, ZTCF, DELTA, ZVCF] = run_analysis('generate_plots', false);

    % Store results
    results{i} = struct('BASE', BASE, 'ZTCF', ZTCF, ...
                       'DELTA', DELTA, 'ZVCF', ZVCF);
end
```

---

## Troubleshooting

### Issue: "Parallel Pool Cannot Be Created"

**Solution:**

```matlab
run_analysis('use_parallel', false);
```

### Issue: "Out of Memory"

**Solutions:**

1. Reduce ZTCF resolution:

```matlab
config = simulation_config();
config.ztcf_end_time = 14;
```

2. Skip checkpointing:

```matlab
run_analysis('use_checkpoints', false);
```

3. Clear workspace between runs:

```matlab
clear all; close all;
run_analysis();
```

### Issue: "Model Not Found"

**Solution:**
Check model path:

```matlab
config = simulation_config();
disp(config.model_path);
% Verify this directory exists and contains GolfSwing model
```

### Issue: Analysis Very Slow

**Check parallelization:**

```matlab
pool = gcp('nocreate');
if isempty(pool)
    disp('Parallel pool not running');
    parpool('local');  % Start manually
end
```

---

Happy analyzing! 🏌️‍♂️📊
