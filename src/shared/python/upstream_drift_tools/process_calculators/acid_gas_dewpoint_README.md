# Acid Gas Dewpoint Calculator

## Overview

A comprehensive calculator for predicting dewpoint temperatures of acid gases (HF, HCl, H2S) in syngas and water vapor mixtures. This tool is essential for preventing corrosion in gasification and syngas processing systems by ensuring operating temperatures remain above acid gas dewpoints.

## Key Features

- Multi-component acid gas dewpoint calculations (H2O, HF, HCl, H2S)
- Multiple vapor pressure calculation methods (Antoine, Extended Antoine, Thermo, CoolProp)
- Condensation risk assessment with safety margin evaluation
- Literature-based thermodynamic correlations from industry-standard sources
- Dewpoint curve generation for parametric analysis
- Optional PyQt6 GUI interface for interactive calculations

## Installation / Prerequisites

### Required Dependencies

```bash
pip install numpy pandas
```

### Optional Dependencies (Enhanced Accuracy)

```bash
pip install thermo CoolProp PyQt6
```

- **thermo**: Provides advanced thermodynamic property calculations
- **CoolProp**: High-accuracy equation of state calculations
- **PyQt6**: GUI interface for interactive use

## Usage Instructions

### Basic Usage

```python
from acid_gas_dewpoint_calculator import AcidGasDewpointCalculator, AcidGasComposition

# Initialize calculator
calc = AcidGasDewpointCalculator()

# Define composition
composition = AcidGasComposition(
    h2o=0.15,    # Water vapor mole fraction
    hf=0.001,    # Hydrogen fluoride mole fraction
    hcl=0.002,   # Hydrogen chloride mole fraction
    h2s=0.005    # Hydrogen sulfide mole fraction
)

# Calculate dewpoint
result = calc.calculate_dewpoint_mixture(
    temperature_c=150,
    pressure_bar=30,
    composition=composition,
    method='antoine'
)

print(f"Overall Dewpoint: {result.overall_dewpoint_c:.2f} C")
print(f"Limiting Component: {result.limiting_component}")
print(f"Condensation Risk: {result.condensation_risk}")
```

### Quick Calculation Function

```python
from acid_gas_dewpoint_calculator import quick_dewpoint_calculation

result = quick_dewpoint_calculation(
    temperature_c=150,
    pressure_bar=30,
    h2o_fraction=0.15,
    hf_fraction=0.001,
    hcl_fraction=0.002,
    h2s_fraction=0.005
)
```

## Input Parameters

| Parameter     | Type  | Range       | Units   | Description                                         |
| ------------- | ----- | ----------- | ------- | --------------------------------------------------- |
| temperature_c | float | -100 to 400 | C       | System temperature                                  |
| pressure_bar  | float | 0.1 to 300  | bar     | System pressure                                     |
| h2o           | float | 0 to 1      | mol/mol | Water vapor mole fraction                           |
| hf            | float | 0 to 1      | mol/mol | Hydrogen fluoride mole fraction                     |
| hcl           | float | 0 to 1      | mol/mol | Hydrogen chloride mole fraction                     |
| h2s           | float | 0 to 1      | mol/mol | Hydrogen sulfide mole fraction                      |
| method        | str   | -           | -       | 'antoine', 'extended_antoine', 'thermo', 'coolprop' |

## Output Format

The `DewpointResult` dataclass contains:

| Field                         | Units    | Description                              |
| ----------------------------- | -------- | ---------------------------------------- |
| temperature_c / temperature_k | C / K    | Input temperature                        |
| pressure_bar / pressure_pa    | bar / Pa | Input pressure                           |
| h2o_dewpoint_c                | C        | Water vapor dewpoint                     |
| hf_dewpoint_c                 | C        | HF dewpoint                              |
| hcl_dewpoint_c                | C        | HCl dewpoint                             |
| h2s_dewpoint_c                | C        | H2S dewpoint                             |
| overall_dewpoint_c            | C        | Highest component dewpoint               |
| limiting_component            | str      | Component determining overall dewpoint   |
| dewpoint_margin_c             | C        | Temperature above dewpoint               |
| condensation_risk             | str      | Risk classification                      |
| \*\_vapor_pressure_pa         | Pa       | Vapor pressures at operating temperature |
| \*\_partial_pressure_pa       | Pa       | Partial pressures of each component      |

## Mathematical Models

### Antoine Equation (Vapor Pressure)

```
log10(P) = A - B / (C + T)
```

Where:

- P = vapor pressure (mmHg)
- T = temperature (C)
- A, B, C = component-specific Antoine constants

**Antoine Constants (from Perry's 8th Ed.):**

| Component | A       | B       | C       |
| --------- | ------- | ------- | ------- |
| H2O       | 8.07131 | 1730.63 | 233.426 |
| HF        | 7.158   | 1111.0  | 235.0   |
| HCl       | 7.960   | 1118.0  | 240.0   |
| H2S       | 6.987   | 884.0   | 240.0   |

### Inverse Antoine (Dewpoint Temperature)

```
T_dewpoint = B / (A - log10(P_partial)) - C
```

### Partial Pressure Calculation

```
P_partial_i = y_i * P_total
```

Where y_i is the mole fraction of component i.

### Condensation Risk Classification

| Margin (C) | Risk Level                      |
| ---------- | ------------------------------- |
| < 0        | HIGH - Condensation occurring   |
| 0 - 10     | MEDIUM - Within 10C of dewpoint |
| 10 - 30    | LOW - Safe margin               |
| > 30       | VERY LOW - Large safety margin  |

## Example Usage

### Typical Syngas Analysis

```python
from acid_gas_dewpoint_calculator import (
    AcidGasDewpointCalculator,
    ACID_GAS_PRESETS,
    estimate_condensation_risk
)

calc = AcidGasDewpointCalculator()

# Use preset composition for coal gasification
composition = ACID_GAS_PRESETS['coal_gasification']

# Calculate at operating conditions
result = calc.calculate_dewpoint_mixture(
    temperature_c=180,
    pressure_bar=25,
    composition=composition
)

# Generate dewpoint curves for analysis
curves = calc.generate_dewpoint_curves(
    pressure_bar=25,
    composition=composition,
    temp_range=(-50, 200),
    num_points=100
)
```

## Troubleshooting

| Issue                            | Cause                                  | Solution                                        |
| -------------------------------- | -------------------------------------- | ----------------------------------------------- |
| NaN dewpoint values              | Component mole fraction is zero        | Only non-zero components have valid dewpoints   |
| Warnings about temperature range | Operating outside correlation validity | Use extended_antoine or thermo method           |
| CoolProp/Thermo not available    | Optional libraries not installed       | Install with `pip install CoolProp thermo`      |
| High condensation risk           | Operating too close to dewpoint        | Increase temperature or reduce acid gas content |

## Literature Sources

- Perry's Chemical Engineers' Handbook, 8th Edition
- NIST Chemistry WebBook
- CRC Handbook of Chemistry and Physics
- Journal of Chemical & Engineering Data (2001, 2003)
- Industrial & Engineering Chemistry Research (1995)
- IAPWS-IF97 Formulation (for water)

## Related Tools

- **Scrubber Calculator**: Design wet scrubbers for acid gas removal
- **Syngas Water Calculator**: Water-gas shift and steam calculations
- **Pressure Drop Calculator**: Pipe system pressure losses
