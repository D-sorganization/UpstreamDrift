# Pressure Drop Calculator

## Overview

An advanced pressure drop calculation engine for combustion and gasification gases. Implements comprehensive pressure drop calculations using industry-standard correlations (Darcy-Weisbach, Colebrook-White, Fanning) with support for compressible flow corrections, multiple friction factor methods, fitting K-factors from Crane TP-410, and extensive gas property databases.

## Key Features

- Darcy-Weisbach equation for pipe friction losses
- Multiple friction factor correlations (Colebrook, Swamee-Jain, Churchill, Haaland)
- Compressible flow corrections for high pressure drop scenarios
- Extensive gas property database for syngas mixtures (H2, CO, CO2, CH4, etc.)
- ASME B36.10M standard pipe size database (1/2" to 24")
- Fitting and valve K-factor database (Crane TP-410)
- Two-K method for improved fitting accuracy at low Reynolds numbers
- Erosional velocity calculations per API RP 14E
- Real gas compressibility (Z-factor) corrections
- Flexible unit conversions (mass, molar, volumetric flows)

## Installation / Prerequisites

### Required Dependencies

```bash
pip install numpy
```

### Standard Library Dependencies (No Installation Required)

- `math`, `logging`, `dataclasses`

## Usage Instructions

### Quick Start

```python
from pressure_drop_calculator import calculate_pressure_drop, print_results

# Simple calculation with standard pipe
result = calculate_pressure_drop(
    pipe_size='4',
    pipe_schedule='40',
    pipe_length=100,           # meters
    flow_rate=1000,
    flow_unit='kg/h',
    pressure=10,               # bar
    temperature=500            # K
)

print(f"Pressure Drop: {result['pressure_drop_bar']:.4f} bar")
print_results(result)
```

### Custom Gas Composition

```python
# Syngas composition (mole fractions)
syngas = {'H2': 0.30, 'CO': 0.40, 'CO2': 0.15, 'N2': 0.10, 'CH4': 0.05}

result = calculate_pressure_drop(
    pipe_diameter=0.1543,      # 6" Schedule 40 (meters)
    pipe_length=50,
    gas_composition=syngas,
    flow_rate=2000,
    flow_unit='kg/h',
    pressure=25,               # bar
    temperature=800            # K
)
```

### With Fittings and Valves

```python
result = calculate_pressure_drop(
    pipe_size='6',
    pipe_schedule='40',
    pipe_length=100,
    flow_rate=5000,
    flow_unit='kg/h',
    pressure=20,
    temperature=750,
    gas_composition={'H2': 0.3, 'CO': 0.5, 'CO2': 0.2},
    fittings=[
        {'type': '90_elbow_std', 'quantity': 4},
        {'type': 'gate_valve_open', 'quantity': 2},
        {'type': 'tee_through_run', 'quantity': 1}
    ]
)
```

## Input Parameters

### Pipe Geometry

| Parameter        | Type  | Range                     | Units  | Description                                  |
| ---------------- | ----- | ------------------------- | ------ | -------------------------------------------- |
| pipe_size        | str   | 1/2 - 24                  | inches | Nominal pipe size (NPS)                      |
| pipe_schedule    | str   | 5S, 10S, 40, 80, 160, XXS | -      | Pipe schedule                                |
| pipe_diameter    | float | > 0                       | m      | Internal diameter (alternative to pipe_size) |
| pipe_length      | float | > 0                       | m      | Total pipe length                            |
| pipe_material    | str   | -                         | -      | Commercial Steel, Stainless, Cast Iron, etc. |
| pipe_roughness   | float | > 0                       | m      | Absolute roughness (overrides material)      |
| elevation_change | float | any                       | m      | Elevation change (+ = upward flow)           |

### Flow Conditions

| Parameter        | Type  | Range    | Units  | Description                                  |
| ---------------- | ----- | -------- | ------ | -------------------------------------------- |
| flow_rate        | float | > 0      | varies | Flow rate value                              |
| flow_unit        | str   | -        | -      | kg/h, kg/s, lb/hr, SCFM, ACFM, kmol/h, Nm3/h |
| pressure         | float | 0.1-1000 | bar    | Inlet pressure (absolute)                    |
| pressure_unit    | str   | -        | -      | bar, psi, Pa, kPa, atm                       |
| temperature      | float | 200-2000 | K      | Inlet temperature                            |
| temperature_unit | str   | -        | -      | K, C, F                                      |

### Gas Composition

| Parameter       | Type | Description                                        |
| --------------- | ---- | -------------------------------------------------- |
| gas_composition | dict | {component: mole_fraction}, auto-normalized to 1.0 |

**Available Gas Components:**
H2, CO, CO2, CH4, C2H6, C2H4, N2, O2, H2O, Ar, H2S, NH3, Air

### Calculation Options

| Parameter                  | Type | Default     | Description                                        |
| -------------------------- | ---- | ----------- | -------------------------------------------------- |
| friction_method            | str  | 'colebrook' | 'colebrook', 'swamee-jain', 'churchill', 'haaland' |
| compressibility_correction | bool | True        | Apply real gas Z-factor corrections                |
| standard_condition         | str  | 'STP'       | 'STP', 'NTP', 'SCFM' for volumetric conversions    |

## Output Format

### Primary Results

| Field               | Units | Description                |
| ------------------- | ----- | -------------------------- |
| pressure_drop_pa    | Pa    | Total pressure drop        |
| pressure_drop_bar   | bar   | Total pressure drop        |
| pressure_drop_psi   | psi   | Total pressure drop        |
| pressure_drop_kpa   | kPa   | Total pressure drop        |
| outlet_pressure_bar | bar   | Outlet pressure (absolute) |

### Pressure Drop Components

| Field                                | Units    | Description               |
| ------------------------------------ | -------- | ------------------------- |
| friction_loss_pa / friction_loss_bar | Pa / bar | Pipe wall friction losses |
| fitting_loss_pa / fitting_loss_bar   | Pa / bar | Fittings and valve losses |
| elevation_loss_pa                    | Pa       | Hydrostatic head change   |
| pressure_drop_per_100ft_pa           | Pa/100ft | Pressure gradient         |

### Flow Characteristics

| Field                                  | Units      | Description                            |
| -------------------------------------- | ---------- | -------------------------------------- |
| reynolds_number                        | -          | Reynolds number (Re)                   |
| friction_factor                        | -          | Darcy friction factor (f)              |
| flow_velocity_m_s / flow_velocity_ft_s | m/s / ft/s | Flow velocity                          |
| mach_number                            | -          | Mach number (M)                        |
| flow_regime                            | str        | 'laminar', 'transitional', 'turbulent' |

### Gas Properties

| Field                  | Units   | Description              |
| ---------------------- | ------- | ------------------------ |
| density_kg_m3          | kg/m3   | Gas mixture density      |
| viscosity_pa_s         | Pa-s    | Dynamic viscosity        |
| molecular_weight       | kg/kmol | Mixture molecular weight |
| compressibility_factor | -       | Real gas Z-factor        |

### Safety Metrics

| Field                  | Units | Description                         |
| ---------------------- | ----- | ----------------------------------- |
| erosional_velocity_m_s | m/s   | API RP 14E erosional velocity limit |
| erosion_ratio          | -     | V_actual / V_erosional              |
| erosion_ratio_percent  | %     | Erosion ratio as percentage         |
| velocity_pressure_pa   | Pa    | Dynamic pressure (rho\*V^2/2)       |
| warnings               | list  | Warning messages                    |

## Mathematical Models

### Darcy-Weisbach Equation

```
Delta_P = f * (L/D) * (rho * V^2 / 2)
```

Where:

- f = Darcy friction factor (dimensionless)
- L = pipe length (m)
- D = pipe internal diameter (m)
- rho = gas density (kg/m3)
- V = flow velocity (m/s)

### Fanning Friction Factor Relation

```
f_Darcy = 4 * f_Fanning
```

### Friction Factor Correlations

**Colebrook-White (Implicit, Iterative):**

```
1/sqrt(f) = -2.0 * log10(epsilon/(3.7*D) + 2.51/(Re*sqrt(f)))
```

**Swamee-Jain (Explicit, within 1% of Colebrook):**

```
f = 0.25 / [log10(epsilon/(3.7*D) + 5.74/Re^0.9)]^2
```

**Churchill (All Flow Regimes):**

```
f = 8 * [(8/Re)^12 + 1/(A + B)^1.5]^(1/12)
A = [-2.457 * ln((7/Re)^0.9 + 0.27*(epsilon/D))]^16
B = (37530/Re)^16
```

**Haaland (Explicit, within 1.5%):**

```
1/sqrt(f) = -1.8 * log10[(epsilon/D / 3.7)^1.11 + 6.9/Re]
```

**Laminar Flow (Re < 2300):**

```
f = 64 / Re
```

### Reynolds Number

```
Re = (rho * V * D) / mu
```

### Fitting Pressure Drop (K-Factor Method)

```
Delta_P_fitting = Sum(K_i * n_i) * (rho * V^2 / 2)
```

### Elevation Pressure Change

```
Delta_P_elevation = rho * g * Delta_h
```

Where g = 9.80665 m/s^2 (standard gravity)

### Compressible Flow Correction (Isothermal)

```
P1^2 - P2^2 = G^2 * (Z*R*T/M) * [f*L/D + Sum(K) + 2*ln(P1/P2)]
```

Where:

- G = mass flux (kg/(m2-s))
- Z = compressibility factor
- R = 8314.5 J/(kmol-K)
- M = molecular weight (kg/kmol)

### Erosional Velocity (API RP 14E)

```
V_erosion = C / sqrt(rho)
```

Where C = 100 (continuous), 125 (intermittent), 150 (non-corrosive)

## Fitting K-Factors (Crane TP-410)

| Fitting Type       | K-Factor | Fitting Type         | K-Factor |
| ------------------ | -------- | -------------------- | -------- |
| 90_elbow_std       | 0.75     | gate_valve_open      | 0.17     |
| 90_elbow_long      | 0.45     | globe_valve_open     | 6.00     |
| 45_elbow_std       | 0.35     | ball_valve_open      | 0.05     |
| tee_through_run    | 0.40     | check_valve_swing    | 2.50     |
| tee_through_branch | 1.50     | butterfly_valve_open | 0.25     |
| entrance_sharp     | 0.50     | exit_sharp           | 1.00     |

## Pipe Material Roughness

| Material         | epsilon (mm) | epsilon (m) |
| ---------------- | ------------ | ----------- |
| Commercial Steel | 0.045        | 0.000045    |
| Stainless Steel  | 0.015        | 0.000015    |
| Drawn Tubing     | 0.0015       | 0.0000015   |
| Cast Iron        | 0.26         | 0.00026     |
| Concrete         | 1.0-3.0      | 0.001-0.003 |
| PVC/Plastic      | 0.0015       | 0.0000015   |

## Example Usage

### Complete Syngas Pipeline Analysis

```python
from pressure_drop_calculator import (
    calculate_pressure_drop_syngas,
    print_results,
    compare_friction_methods
)

# Compare friction methods at operating conditions
compare_friction_methods(reynolds_number=100000, relative_roughness=0.0003)

# Syngas pipeline calculation
result = calculate_pressure_drop_syngas(
    pipe_size='8',
    pipe_schedule='40',
    pipe_length=500,
    flow_rate=10000,
    flow_unit='kg/h',
    pressure=30,
    temperature=650,
    H2_fraction=0.35,
    CO_fraction=0.40,
    CO2_fraction=0.15,
    N2_fraction=0.05,
    CH4_fraction=0.05,
    fittings=[
        {'type': '90_elbow_long', 'quantity': 6},
        {'type': 'gate_valve_open', 'quantity': 3},
        {'type': 'tee_through_branch', 'quantity': 2}
    ],
    friction_method='colebrook',
    compressibility_correction=True
)

# Print formatted results with recommendations
print_results(result, show_recommendations=True)

# Access specific values
print(f"Total Pressure Drop: {result['pressure_drop_bar']:.4f} bar")
print(f"Reynolds Number: {result['reynolds_number']:.0f}")
print(f"Erosion Ratio: {result['erosion_ratio_percent']:.1f}%")
```

## Troubleshooting

| Issue                        | Cause                            | Solution                               |
| ---------------------------- | -------------------------------- | -------------------------------------- |
| Unknown pipe size            | Size not in ASME database        | Use pipe_diameter directly in meters   |
| Unknown flow unit            | Unit not recognized              | Run list_flow_units() for options      |
| Unknown gas component        | Component not in database        | Run list_gas_components() for options  |
| Choked flow warning          | Pressure drop > inlet pressure   | Increase pipe diameter or reduce flow  |
| High erosion ratio           | Velocity exceeds erosional limit | Use larger pipe diameter               |
| Composition != 1.0           | Mole fractions don't sum to 1    | Composition auto-normalizes            |
| Negative outlet pressure     | Excessive pressure drop          | Check inputs; flow may be choked       |
| Large compressibility effect | High Delta P / P ratio           | Enable compressibility_correction=True |

## Helper Functions

| Function                            | Description                                       |
| ----------------------------------- | ------------------------------------------------- |
| show_help()                         | Display comprehensive help                        |
| list_gas_components()               | Show available gas components with properties     |
| list_fittings(category)             | Show fittings with K-factors (filter by category) |
| list_pipe_sizes()                   | Show standard ASME pipe sizes and schedules       |
| list_flow_units()                   | Show available flow rate units                    |
| list_materials()                    | Show pipe materials and roughness values          |
| compare_friction_methods(Re, eps_D) | Compare friction factor correlations              |
| validate_inputs(...)                | Pre-validate inputs before calculation            |

## Literature References

- Darcy, H. (1857), Weisbach, J. (1845): Pipe flow friction equation
- Colebrook, C.F. (1939): J. Inst. Civil Engineers, London, 11, 133-156
- Swamee, P.K., Jain, A.K. (1976): J. Hydraulics Division, ASCE, 102(5), 657-664
- Churchill, S.W. (1977): Chemical Engineering, 84(24), 91-92
- Haaland, S.E. (1983): J. Fluids Engineering, 105(1), 89-90
- Crane Technical Paper No. 410: Flow of Fluids Through Valves, Fittings, and Pipe
- Perry's Chemical Engineers' Handbook, 9th Edition
- API RP 14E: Offshore Production Piping Systems
- ASME B36.10M: Welded and Seamless Wrought Steel Pipe
- GPSA Engineering Data Book, 14th Edition
- Reid, Prausnitz, Poling: The Properties of Gases and Liquids, 5th Ed.

## Related Tools

- **Scrubber Calculator**: Packed column pressure drop and sizing
- **Flare Calculator**: Relief header and flare tip sizing
- **Baghouse Calculator**: Ductwork and filter pressure losses
- **Acid Gas Dewpoint Calculator**: Corrosion prevention in piping
