# Flare Calculator

## Overview

A core calculation engine for flare system design and analysis in oil and gas, refinery, and gasification applications. This tool sizes flares based on gas flow conditions, calculates thermal radiation zones, and evaluates combustion efficiency in accordance with API 521 guidelines.

## Key Features

- Flare tip diameter sizing based on exit velocity requirements
- Flare stack height calculation for radiation safety
- Heat release rate calculations from gas heating values
- Radiation zone mapping (lethal, damage, safe, comfort zones)
- Combustion efficiency estimation
- Support for multi-component gas mixtures
- API 521 compliant design methods

## Installation / Prerequisites

### Required Dependencies

```bash
pip install dataclasses  # Built-in for Python 3.7+
```

No external dependencies required. The calculator uses standard Python math libraries.

## Usage Instructions

### Basic Usage

```python
from flare_calculator import FlareCalculator

# Initialize calculator
calc = FlareCalculator()

# Define gas composition (mol%)
gas_composition = {
    'H2': 30.0,
    'CO': 40.0,
    'CH4': 10.0,
    'CO2': 15.0,
    'N2': 5.0
}

# Calculate flare design
design = calc.calculate_flare_size(
    total_flow=5000,           # kg/hr
    gas_composition=gas_composition,
    temperature=500,           # K
    pressure=2.0               # bar
)

print(f"Flare Height: {design.height:.1f} m")
print(f"Tip Diameter: {design.diameter:.3f} m")
print(f"Exit Velocity: {design.exit_velocity:.1f} m/s")
print(f"Heat Release: {design.heat_release:.0f} kW")
```

### Radiation Zone Analysis

```python
# Get radiation zones around flare
zones = calc.calculate_radiation_zones(design)

print(f"Lethal Zone (37.5 kW/m2): {zones['lethal']:.1f} m")
print(f"Damage Zone (12.5 kW/m2): {zones['damage']:.1f} m")
print(f"Safe Zone (1.6 kW/m2): {zones['safe']:.1f} m")
print(f"Comfort Zone (0.5 kW/m2): {zones['comfort']:.1f} m")
```

## Input Parameters

| Parameter       | Type  | Range    | Units | Description                         |
| --------------- | ----- | -------- | ----- | ----------------------------------- |
| total_flow      | float | > 0      | kg/hr | Total gas mass flow rate            |
| gas_composition | dict  | -        | mol%  | Gas composition (H2, CO, CH4, etc.) |
| temperature     | float | 200-1000 | K     | Gas temperature at flare inlet      |
| pressure        | float | 0.1-50   | bar   | Gas pressure at flare inlet         |

### Supported Gas Components

| Component | Molecular Weight (g/mol) | Heating Value (kJ/kg) | Cp (kJ/kg-K) |
| --------- | ------------------------ | --------------------- | ------------ |
| H2        | 2.016                    | 119,930               | 14.3         |
| CO        | 28.01                    | 10,100                | 1.04         |
| CH4       | 16.04                    | 50,010                | 2.22         |
| C2H6      | 30.07                    | 47,520                | 1.75         |
| C3H8      | 44.10                    | 46,360                | 1.67         |
| C4H10     | 58.12                    | 45,720                | 1.66         |
| H2S       | 34.08                    | 16,500                | 1.05         |
| N2        | 28.01                    | 0                     | 1.04         |
| CO2       | 44.01                    | 0                     | 0.84         |
| H2O       | 18.02                    | 0                     | 1.87         |

## Output Format

### FlareDesign Dataclass

| Field               | Units | Description                 |
| ------------------- | ----- | --------------------------- |
| height              | m     | Required flare stack height |
| diameter            | m     | Flare tip diameter          |
| exit_velocity       | m/s   | Gas exit velocity at tip    |
| heat_release        | kW    | Total heat release rate     |
| radiation_intensity | kW/m2 | Design radiation at grade   |

### Radiation Zones Dictionary

| Zone    | Intensity (kW/m2) | Description                |
| ------- | ----------------- | -------------------------- |
| lethal  | 37.5              | Fatal exposure in seconds  |
| damage  | 12.5              | Equipment/structure damage |
| safe    | 1.6               | Safe for personnel access  |
| comfort | 0.5               | No discomfort              |

## Mathematical Models

### Heat Release Rate

```
Q = m_dot * HV_mix / 3600
```

Where:

- Q = heat release rate (kW)
- m_dot = mass flow rate (kg/hr)
- HV_mix = mixture heating value (kJ/kg)

### Mixture Heating Value

```
HV_mix = Sum(y_i * HV_i)
```

Where y_i is the mole fraction of component i.

### Gas Density (Ideal Gas Law)

```
rho = (P * MW) / (R * T)
```

Where:

- P = pressure (Pa)
- MW = molecular weight (kg/kmol)
- R = universal gas constant (8314.5 J/kmol-K)
- T = temperature (K)

### Flare Tip Diameter (API 521 Method)

For smokeless operation, target exit velocity is typically 0.5 Mach or maximum 170 m/s.

```
A = m_dot / (rho * V_target)
D = sqrt(4 * A / pi)
```

Where:

- A = required cross-sectional area (m2)
- V_target = target exit velocity (m/s)
- D = tip diameter (m)

### Flare Height (Radiation Method)

Using simplified point source radiation model:

```
H = sqrt(epsilon * Q / (4 * pi * I_target))
```

Where:

- epsilon = flame emissivity (~0.3 for hydrocarbon flames)
- Q = heat release rate (kW)
- I_target = target radiation intensity (1.6 kW/m2 for safe access)

### Radiation Zone Distance

```
R = sqrt(epsilon * Q / (4 * pi * I))
```

Where I is the radiation intensity limit for each zone.

### Combustion Efficiency

Base efficiency of 98% adjusted for:

- High H2 content (> 50%): +1%
- High CO content (> 30%): -2%
- High H2S content (> 10%): -1%
- Low temperature (< 300K): -2%
- High temperature (> 500K): +1%

```
eta = 0.98 + adjustments
eta_final = max(0.95, min(0.999, eta))
```

## Example Usage

### Complete Flare System Design

```python
from flare_calculator import FlareCalculator

calc = FlareCalculator()

# Syngas flare for gasification plant
syngas = {
    'H2': 35.0,
    'CO': 45.0,
    'CH4': 5.0,
    'CO2': 10.0,
    'H2O': 3.0,
    'N2': 2.0
}

# Design for emergency relief scenario
design = calc.calculate_flare_size(
    total_flow=10000,  # kg/hr (emergency rate)
    gas_composition=syngas,
    temperature=573,   # 300C
    pressure=1.5       # bar
)

# Radiation analysis
zones = calc.calculate_radiation_zones(design)

# Combustion efficiency
efficiency = calc.calculate_combustion_efficiency(
    gas_composition=syngas,
    temperature=573,
    pressure=1.5
)

print("=== FLARE DESIGN SUMMARY ===")
print(f"Stack Height: {design.height:.1f} m")
print(f"Tip Diameter: {design.diameter*1000:.0f} mm")
print(f"Exit Velocity: {design.exit_velocity:.0f} m/s")
print(f"Heat Release: {design.heat_release/1000:.1f} MW")
print(f"Combustion Efficiency: {efficiency*100:.1f}%")
print(f"\n=== EXCLUSION ZONES ===")
print(f"Lethal Zone Radius: {zones['lethal']:.1f} m")
print(f"Damage Zone Radius: {zones['damage']:.1f} m")
print(f"Safe Access Radius: {zones['safe']:.1f} m")
```

## Troubleshooting

| Issue                  | Cause                | Solution                                  |
| ---------------------- | -------------------- | ----------------------------------------- |
| Zero heat release      | All inert components | Add combustible components to composition |
| Zero diameter          | Zero flow or density | Check flow rate and gas properties        |
| Very high stack height | High heat release    | Consider multiple flare tips              |
| Low efficiency         | High CO or H2S       | Expected for difficult gases              |
| Unrealistic density    | Extreme T or P       | Verify temperature and pressure inputs    |

## Design Guidelines (API 521)

### Exit Velocity Limits

| Condition           | Velocity Limit           |
| ------------------- | ------------------------ |
| Smokeless operation | 0.2-0.5 Mach             |
| Emergency relief    | < 0.8 Mach (sonic limit) |
| Typical design      | 120-170 m/s              |

### Radiation Limits (API 521 / API 2000)

| Location                  | Max Intensity (kW/m2) |
| ------------------------- | --------------------- |
| Personnel (8 hr exposure) | 1.58                  |
| Personnel (escape routes) | 6.31                  |
| Equipment/structures      | 15.77                 |
| Flare tip                 | 37.5                  |

### Minimum Flare Heights

- Industrial flares: typically 20-100 m
- Minimum height: 10 m (per most codes)
- Consider wind effects and flame tilt

## Related Tools

- **Baghouse Calculator**: Solid particulate removal
- **Scrubber Calculator**: Gas cleaning before flare
- **Pressure Drop Calculator**: Relief system piping
- **Acid Gas Dewpoint Calculator**: Corrosion in flare headers
