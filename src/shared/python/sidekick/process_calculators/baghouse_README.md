# Baghouse Calculator

## Overview

A core calculation engine for baghouse filter performance analysis, solid removal efficiency, and collection drum sizing. The calculator determines filter performance including solid removal rates, drum fill times, and air-to-cloth ratios for industrial dust collection applications in syngas and combustion systems.

## Key Features

- Solid removal rate calculations for carbon and ash
- Collection drum sizing and fill time estimation
- Air-to-cloth ratio calculations
- Heat loss and temperature drop modeling
- Volumetric flow rate conversions (ACFM/SCFM)
- Dual operation modes: full thermodynamic or simplified ideal gas calculations
- Support for variable gas compositions

## Installation / Prerequisites

### Required Dependencies

```bash
pip install dataclasses  # Built-in for Python 3.7+
```

### Optional Dependencies (Enhanced Thermodynamics)

```bash
pip install thermo  # For detailed thermodynamic calculations
```

When the `thermo` module is unavailable, the calculator automatically falls back to simplified ideal gas approximations.

## Usage Instructions

### Basic Usage

```python
from baghouse_calculator import BaghouseCalculator

# Initialize calculator
calc = BaghouseCalculator()

# Define gas composition (mole fractions)
composition = {
    'H2': 0.30,
    'CO': 0.40,
    'CO2': 0.15,
    'N2': 0.10,
    'CH4': 0.05
}

# Calculate baghouse performance
result = calc.calculate(
    gas_flow_kg_s=2.5,              # Gas mass flow rate [kg/s]
    inlet_temp_k=573.15,            # Inlet temperature [K] (300C)
    pressure_pa=101325,             # Pressure [Pa]
    composition=composition,         # Gas composition
    solid_carbon_in_kg_hr=50,       # Solid carbon input [kg/hr]
    ash_in_kg_hr=25,                # Ash input [kg/hr]
    carbon_removal_efficiency=0.99,  # 99% carbon removal
    ash_removal_efficiency=0.995,    # 99.5% ash removal
    heat_loss_w=50000,              # Heat loss [W]
    drum_volume_m3=2.0,             # Collection drum volume [m3]
    solid_density_kg_m3=800,        # Collected solids density [kg/m3]
    bag_area_ft2=5000               # Total bag filter area [ft2]
)
```

### Accessing Results

```python
print(f"Carbon Removed: {result.carbon_removed_rate:.2f} kg/hr")
print(f"Total Solids Removed: {result.total_solids_removed_rate:.2f} kg/hr")
print(f"Drum Fill Time: {result.drum_fill_time_days:.1f} days")
print(f"Air-to-Cloth Ratio: {result.air_to_cloth_ratio:.2f} ft/min")
print(f"Outlet Temperature: {result.outlet_temperature_c:.1f} C")
```

## Input Parameters

| Parameter                 | Type  | Range    | Units   | Description                          |
| ------------------------- | ----- | -------- | ------- | ------------------------------------ |
| gas_flow_kg_s             | float | > 0      | kg/s    | Gas mass flow rate                   |
| inlet_temp_k              | float | 273-1273 | K       | Inlet gas temperature                |
| pressure_pa               | float | > 0      | Pa      | Operating pressure                   |
| composition               | dict  | -        | mol/mol | Gas composition (mole fractions)     |
| solid_carbon_in_kg_hr     | float | >= 0     | kg/hr   | Solid carbon input rate              |
| ash_in_kg_hr              | float | >= 0     | kg/hr   | Ash input rate                       |
| carbon_removal_efficiency | float | 0-1      | -       | Carbon removal efficiency (fraction) |
| ash_removal_efficiency    | float | 0-1      | -       | Ash removal efficiency (fraction)    |
| heat_loss_w               | float | >= 0     | W       | Heat loss from baghouse              |
| drum_volume_m3            | float | > 0      | m3      | Collection drum volume               |
| solid_density_kg_m3       | float | > 0      | kg/m3   | Bulk density of collected solids     |
| bag_area_ft2              | float | > 0      | ft2     | Total bag filter surface area        |

## Output Format

The `BaghouseResult` dataclass contains:

| Field                       | Units  | Description                        |
| --------------------------- | ------ | ---------------------------------- |
| carbon_removed_rate         | kg/hr  | Carbon removal rate                |
| ash_removed_rate            | kg/hr  | Ash removal rate                   |
| total_solids_removed_rate   | kg/hr  | Total solids collected             |
| drum_fill_time_hours        | hours  | Time to fill collection drum       |
| drum_fill_time_days         | days   | Time to fill collection drum       |
| carbon_only_fill_time_hours | hours  | Fill time if only carbon collected |
| ash_only_fill_time_hours    | hours  | Fill time if only ash collected    |
| clean_gas_flow_rate         | kg/hr  | Clean gas mass flow rate           |
| flow_acfm                   | ACFM   | Actual cubic feet per minute       |
| flow_scfm                   | SCFM   | Standard cubic feet per minute     |
| air_to_cloth_ratio          | ft/min | Filtration velocity                |
| outlet_temperature_c        | C      | Outlet gas temperature             |
| ash_stream_composition      | dict   | Carbon/ash fractions in solids     |
| removal_efficiency          | dict   | Removal efficiencies (%)           |

## Mathematical Models

### Heat Balance and Temperature Drop

```
Delta_T = Q_loss / (m_dot * Cp)
T_outlet = T_inlet - Delta_T
```

Where:

- Q_loss = heat loss rate (W)
- m_dot = mass flow rate (kg/s)
- Cp = specific heat capacity (J/kg-K)

### Ideal Gas Heat Capacity Estimation

For syngas mixtures, weighted average Cp values are used:

| Component | Cp at ~500K (J/mol-K) |
| --------- | --------------------- |
| H2        | 29.1                  |
| CO        | 29.2                  |
| CO2       | 41.3                  |
| H2O       | 35.5                  |
| N2        | 29.5                  |
| CH4       | 44.5                  |

### Solid Removal Calculations

```
m_carbon_removed = m_carbon_in * eta_carbon
m_ash_removed = m_ash_in * eta_ash
m_total = m_carbon_removed + m_ash_removed
```

### Drum Fill Time

```
t_fill = (rho_solid * V_drum) / m_total_removed
```

Where:

- rho_solid = bulk density of collected solids (kg/m3)
- V_drum = drum volume (m3)

### Air-to-Cloth Ratio

```
A/C = Q_acfm / A_bag
```

Where:

- Q_acfm = actual volumetric flow rate (ft3/min)
- A_bag = total bag filter area (ft2)

**Typical A/C Ratios:**

| Application          | A/C Ratio (ft/min) |
| -------------------- | ------------------ |
| Pulse-jet cleaning   | 4-6                |
| Reverse-air cleaning | 2-3.5              |
| Shaker cleaning      | 2-3                |

### Volumetric Flow Rate (Ideal Gas)

```
Q_actual = (n_dot * R * T) / P
Q_standard = (n_dot * R * T_std) / P_std
```

Standard conditions: T_std = 273.15 K, P_std = 101325 Pa

## Example Usage

### Complete Baghouse Analysis

```python
from baghouse_calculator import BaghouseCalculator, BaghouseResult

calc = BaghouseCalculator()

# Coal gasification syngas
syngas = {
    'H2': 0.25,
    'CO': 0.35,
    'CO2': 0.20,
    'H2O': 0.10,
    'N2': 0.08,
    'CH4': 0.02
}

result = calc.calculate(
    gas_flow_kg_s=5.0,
    inlet_temp_k=623.15,  # 350C
    pressure_pa=150000,   # 1.5 bar
    composition=syngas,
    solid_carbon_in_kg_hr=100,
    ash_in_kg_hr=50,
    carbon_removal_efficiency=0.995,
    ash_removal_efficiency=0.999,
    heat_loss_w=75000,
    drum_volume_m3=3.0,
    solid_density_kg_m3=750,
    bag_area_ft2=8000
)

# Print comprehensive results
print("=== Baghouse Performance ===")
print(f"Gas Flow: {result.clean_gas_flow_rate:.0f} kg/hr")
print(f"ACFM: {result.flow_acfm:.0f}")
print(f"SCFM: {result.flow_scfm:.0f}")
print(f"A/C Ratio: {result.air_to_cloth_ratio:.2f} ft/min")
print(f"Outlet Temp: {result.outlet_temperature_c:.1f} C")
print(f"\n=== Solids Collection ===")
print(f"Carbon: {result.carbon_removed_rate:.1f} kg/hr")
print(f"Ash: {result.ash_removed_rate:.1f} kg/hr")
print(f"Total: {result.total_solids_removed_rate:.1f} kg/hr")
print(f"Drum Fill: {result.drum_fill_time_days:.1f} days")
```

## Troubleshooting

| Issue                 | Cause                       | Solution                           |
| --------------------- | --------------------------- | ---------------------------------- |
| Infinite fill time    | Zero solid input rates      | Verify solid loading inputs        |
| Zero A/C ratio        | Zero bag area               | Check bag_area_ft2 parameter       |
| No temperature drop   | Zero heat loss or zero Cp   | Verify heat_loss_w and composition |
| Thermo module warning | Optional dependency missing | Install with `pip install thermo`  |
| Unexpected flow rates | Incorrect composition       | Ensure composition sums to ~1.0    |

## Design Guidelines

### Recommended Air-to-Cloth Ratios

| Dust Type                | A/C (ft/min) |
| ------------------------ | ------------ |
| Light dust (< 1 gr/ft3)  | 5-6          |
| Medium dust (1-5 gr/ft3) | 3-5          |
| Heavy dust (> 5 gr/ft3)  | 2-3          |
| Sticky/hygroscopic       | 2-3          |

### Pressure Drop Considerations

Typical pressure drop: 4-8 inches w.c. (1-2 kPa)

Factors affecting pressure drop:

- Dust loading
- Cleaning frequency
- Filter media type
- Air-to-cloth ratio

## Related Tools

- **Flare Calculator**: Emergency relief system design
- **Scrubber Calculator**: Wet scrubber for gas cleaning
- **Pressure Drop Calculator**: Ductwork pressure losses
- **Acid Gas Dewpoint Calculator**: Corrosion prevention
