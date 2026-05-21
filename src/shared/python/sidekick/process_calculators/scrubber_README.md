# Scrubber Calculator

## Overview

A technical calculator for countercurrent packed bed scrubbers used in syngas acid gas removal applications. Implements industry-standard design methods for vessel sizing, pressure drop, mass transfer (NTU/HTU), heat transfer, and caustic consumption calculations.

## Key Features

- Vessel sizing based on gas velocity and flooding correlations
- Pressure drop calculation using Eckert's generalized correlation
- NTU/HTU-based mass transfer design for acid gas removal
- Heat transfer with water-side balance and approach temperature
- Stoichiometric caustic (NaOH) consumption calculation
- Comprehensive packing property database
- Henry's law constants for acid gas solubility
- Column diameter optimization

## Installation / Prerequisites

### Required Dependencies

```bash
pip install numpy
```

No additional dependencies required.

## Usage Instructions

### Basic Column Sizing

```python
from scrubber_calculator import (
    calculate_gas_density,
    calculate_flooding_velocity,
    calculate_column_diameter,
    calculate_pressure_drop,
    PACKING_DATABASE
)

# Gas properties
temp_k = 350  # K
pressure_pa = 150000  # Pa (1.5 bar)
mw_gas = 25  # kg/kmol (syngas mixture)

# Calculate gas density
rho_gas = calculate_gas_density(temp_k, pressure_pa, mw_gas)

# Get packing properties
packing = PACKING_DATABASE['Metal Pall Rings']

# Calculate flooding velocity
u_flood = calculate_flooding_velocity(
    liquid_mass_flux=5.0,     # kg/(m2-s)
    gas_density=rho_gas,
    liquid_density=1000,       # kg/m3 (water)
    packing=packing
)

# Size column diameter
sizing = calculate_column_diameter(
    gas_flow_kg_hr=5000,
    gas_density=rho_gas,
    flooding_velocity=u_flood,
    percent_of_flood=70
)

print(f"Column Diameter: {sizing['diameter_m']:.2f} m")
print(f"Design Velocity: {sizing['design_velocity_m_s']:.2f} m/s")
```

### Mass Transfer Design

```python
from scrubber_calculator import (
    calculate_ntu_removal,
    calculate_htu,
    calculate_required_packed_height
)

# Calculate NTU for HCl removal
ntu = calculate_ntu_removal(
    inlet_conc=0.002,   # 2000 ppm inlet
    outlet_conc=0.00001 # 10 ppm outlet (99.5% removal)
)

# Calculate HTU
htu = calculate_htu(
    gas_mass_flux=2.5,      # kg/(m2-s)
    liquid_mass_flux=5.0,   # kg/(m2-s)
    gas_density=rho_gas,
    packing=packing,
    kla=500                 # 1/hr overall mass transfer coefficient
)

# Required packed height
Z = calculate_required_packed_height(ntu, htu, safety_factor=1.2)
print(f"NTU: {ntu:.2f}")
print(f"HTU: {htu:.2f} m")
print(f"Packed Height: {Z:.2f} m")
```

## Input Parameters

### Gas Properties

| Parameter        | Type  | Range        | Units   | Description        |
| ---------------- | ----- | ------------ | ------- | ------------------ |
| temperature_k    | float | 273-500      | K       | Gas temperature    |
| pressure_pa      | float | 50000-500000 | Pa      | Operating pressure |
| molecular_weight | float | 2-50         | kg/kmol | Gas mixture MW     |

### Column Sizing

| Parameter        | Type  | Range | Units     | Description                 |
| ---------------- | ----- | ----- | --------- | --------------------------- |
| gas_flow_kg_hr   | float | > 0   | kg/hr     | Gas mass flow rate          |
| liquid_mass_flux | float | 1-20  | kg/(m2-s) | Liquid flux (L/G design)    |
| percent_of_flood | float | 50-85 | %         | Design fraction of flooding |

### Mass Transfer

| Parameter     | Type  | Range    | Units   | Description                       |
| ------------- | ----- | -------- | ------- | --------------------------------- |
| inlet_conc    | float | 0-1      | mol/mol | Inlet pollutant concentration     |
| outlet_conc   | float | 0-1      | mol/mol | Target outlet concentration       |
| kla           | float | 100-2000 | 1/hr    | Overall mass transfer coefficient |
| safety_factor | float | 1.1-1.5  | -       | Design safety factor              |

## Output Format

### Column Diameter Results

| Field               | Units | Description                 |
| ------------------- | ----- | --------------------------- |
| design_velocity_m_s | m/s   | Superficial gas velocity    |
| cross_section_m2    | m2    | Column cross-sectional area |
| diameter_m          | m     | Column diameter             |
| diameter_ft         | ft    | Column diameter (US units)  |

### Heat Transfer Results

| Field            | Units | Description                   |
| ---------------- | ----- | ----------------------------- |
| sensible_heat_kw | kW    | Sensible heat duty            |
| latent_heat_kw   | kW    | Latent heat from condensation |
| total_heat_kw    | kW    | Total heat duty               |

### Caustic Requirement Results

| Field               | Units | Description               |
| ------------------- | ----- | ------------------------- |
| naoh_pure_kg_hr     | kg/hr | Pure NaOH requirement     |
| naoh_solution_kg_hr | kg/hr | NaOH solution requirement |
| naoh_solution_L_hr  | L/hr  | Solution volume rate      |
| salt_produced_kg_hr | kg/hr | Total salt produced       |

## Mathematical Models

### Gas Density (Ideal Gas Law)

```
rho = (P * MW) / (R * T)
```

### Gas Viscosity (Sutherland's Formula)

```
mu = mu_ref * (T/T_ref)^1.5 * (T_ref + S) / (T + S)
```

Where S = 110.4 K (Sutherland constant for air-like gases)

### Flooding Velocity (Eckert Correlation)

```
Flow Parameter: X = (L/G) * sqrt(rho_G / rho_L)
Capacity Parameter: Y_flood = C_flood * exp(-1.5 * X^0.5)
Flooding Mass Flux: G'_flood = sqrt(Y_flood * rho_G * rho_L * g / (F * mu_L^0.1))
Flooding Velocity: u_flood = G'_flood / rho_G
```

### Pressure Drop (Eckert Generalized)

```
Y = (G'^2 * F * mu_L^0.1) / (rho_G * rho_L * g)
dP/dZ = alpha * Y^beta * (1 + gamma * X)
```

Typical coefficients: alpha = 85 Pa/m, beta = 1.1, gamma = 3.5

### Number of Transfer Units (NTU)

For irreversible absorption (chemical scrubbing):

```
NTU = ln(y_in / y_out)
```

### Height of Transfer Unit (HTU)

```
HTU = C_H / (kla * a * (L/G)^n)
```

Where:

- C_H = packing HTU constant
- a = specific surface area (m2/m3)
- n = packing exponent

### Packed Height

```
Z = NTU * HTU * SF
```

Where SF = safety factor (typically 1.1-1.3)

### Caustic Stoichiometry

| Reaction                    | NaOH Ratio |
| --------------------------- | ---------- |
| HCl + NaOH -> NaCl + H2O    | 1:1        |
| SO2 + 2NaOH -> Na2SO3 + H2O | 1:2        |
| H2S + 2NaOH -> Na2S + 2H2O  | 1:2        |
| HF + NaOH -> NaF + H2O      | 1:1        |
| CO2 + 2NaOH -> Na2CO3 + H2O | 1:2        |

### Heat Transfer Duty

```
Q_sensible = m_gas * Cp * (T_in - T_out)
Q_latent = m_water_condensed * h_fg
Q_total = Q_sensible + Q_latent
```

### Cooling Water Requirement

```
m_water = Q_total / (Cp_water * Delta_T_water)
```

## Packing Database

| Packing Type          | Material | Size (mm) | Surface (m2/m3) | Void Fraction | Packing Factor |
| --------------------- | -------- | --------- | --------------- | ------------- | -------------- |
| Ceramic Raschig Rings | Ceramic  | 50        | 95              | 0.74          | 155            |
| Metal Pall Rings      | SS       | 50        | 112             | 0.95          | 66             |
| Plastic Cascade Rings | PP       | 50        | 105             | 0.92          | 72             |
| Structured 250Y       | SS       | -         | 250             | 0.98          | 33             |

## Example Usage

### Complete Scrubber Design

```python
from scrubber_calculator import (
    calculate_gas_density,
    calculate_gas_viscosity,
    calculate_flooding_velocity,
    calculate_column_diameter,
    calculate_pressure_drop,
    calculate_ntu_removal,
    calculate_htu,
    calculate_required_packed_height,
    calculate_caustic_requirement,
    calculate_heat_transfer_duty,
    calculate_cooling_water_requirement,
    PACKING_DATABASE
)

# Operating conditions
T_gas = 400  # K
P = 200000   # Pa
MW = 22      # kg/kmol
gas_flow = 10000  # kg/hr

# Gas properties
rho_g = calculate_gas_density(T_gas, P, MW)
mu_g = calculate_gas_viscosity(T_gas, MW)

# Select packing
packing = PACKING_DATABASE['Metal Pall Rings']

# Flooding analysis
L_flux = 8.0  # kg/(m2-s)
u_flood = calculate_flooding_velocity(L_flux, rho_g, 1000, packing)

# Column sizing (70% of flood)
sizing = calculate_column_diameter(gas_flow, rho_g, u_flood, 70)

# Mass transfer for HCl removal
ntu = calculate_ntu_removal(0.001, 0.00001)  # 99% removal
htu = calculate_htu(gas_flow/3600/sizing['cross_section_m2'],
                    L_flux, rho_g, packing, 600)
Z = calculate_required_packed_height(ntu, htu, 1.2)

# Pressure drop
G_flux = (gas_flow/3600) / sizing['cross_section_m2']
dp = calculate_pressure_drop(
    sizing['design_velocity_m_s'], rho_g, L_flux, 1000, packing, Z
)

# Caustic requirement
caustic = calculate_caustic_requirement(
    {'HCl': 10},  # kg/hr removed
    caustic_concentration=25  # wt%
)

# Heat duty
heat = calculate_heat_transfer_duty(
    gas_flow, T_gas-273.15, 40, 500, 1100
)

# Cooling water
cw = calculate_cooling_water_requirement(
    heat['total_heat_kw'], 25, 5, 40
)

print("=== SCRUBBER DESIGN ===")
print(f"Column Diameter: {sizing['diameter_m']:.2f} m")
print(f"Packed Height: {Z:.2f} m")
print(f"Pressure Drop: {dp:.0f} Pa")
print(f"NaOH (25%): {caustic['naoh_solution_L_hr']:.1f} L/hr")
print(f"Heat Duty: {heat['total_heat_kw']:.1f} kW")
print(f"Cooling Water: {cw['water_flow_L_min']:.1f} L/min")
```

## Troubleshooting

| Issue                          | Cause                   | Solution                           |
| ------------------------------ | ----------------------- | ---------------------------------- |
| Negative outlet concentration  | NTU calculation error   | Ensure inlet > outlet              |
| Zero flooding velocity         | Invalid flow parameter  | Check L/G ratio and densities      |
| High pressure drop (> 2 kPa/m) | Near flooding           | Reduce gas velocity                |
| Infinite HTU                   | Zero kla                | Verify mass transfer coefficient   |
| No cooling possible            | Approach temp too small | Increase delta T or use more water |

## Design Guidelines

### Typical Operating Ranges

| Parameter            | Range                   |
| -------------------- | ----------------------- |
| Percent of flooding  | 60-80%                  |
| L/G ratio (mass)     | 2-10 kg liquid / kg gas |
| Pressure drop        | 200-800 Pa/m            |
| NaOH concentration   | 10-50 wt%               |
| Approach temperature | 5-15 C                  |

### Literature References

- Perry's Chemical Engineers' Handbook, 9th Edition
- Treybal, R.E., "Mass Transfer Operations", 3rd Edition
- Kohl, A.L., Nielsen, R.B., "Gas Purification", 5th Edition
- Strigle, R.F., "Packed Tower Design and Applications", 2nd Edition
- Eckert, J.S., Chem. Eng. Prog., Vol. 57, No. 9, 1961

## Related Tools

- **Acid Gas Dewpoint Calculator**: Determine safe operating temperatures
- **Flare Calculator**: Size emergency relief systems
- **Pressure Drop Calculator**: Piping to/from scrubber
- **Baghouse Calculator**: Pre-treatment particulate removal
