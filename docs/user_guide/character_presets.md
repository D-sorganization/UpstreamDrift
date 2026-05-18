# Character Presets Reference

This document provides a comprehensive reference for all available character presets in the Humanoid Character Builder.

## Overview

Character presets provide pre-configured anthropometric parameters for quickly creating humanoid characters with realistic body proportions. Each preset is based on anthropometric data from scientific sources.

## Available Presets

### Basic Body Types

| Preset     | Height (m) | Mass (kg) | Build Type | Gender  | Description                                   |
| ---------- | ---------- | --------- | ---------- | ------- | --------------------------------------------- |
| `athletic` | 1.80       | 80.0      | Mesomorph  | Neutral | Athletic, muscular build with broad shoulders |
| `average`  | 1.75       | 75.0      | Average    | Neutral | Average proportions                           |
| `heavy`    | 1.78       | 100.0     | Endomorph  | Neutral | Heavier build with higher body fat            |
| `lean`     | 1.82       | 70.0      | Ectomorph  | Neutral | Lean, tall build with long limbs              |
| `compact`  | 1.65       | 65.0      | Average    | Neutral | Shorter, compact build                        |
| `tall`     | 1.95       | 90.0      | Average    | Neutral | Tall build                                    |
| `minimal`  | 1.70       | 60.0      | Ectomorph  | Neutral | Minimal/lightweight build for testing         |

### Gender-Specific Presets

| Preset           | Height (m) | Mass (kg) | Build Type | Gender | Description                                |
| ---------------- | ---------- | --------- | ---------- | ------ | ------------------------------------------ |
| `male_average`   | 1.78       | 80.0      | Average    | Male   | Average male proportions                   |
| `female_average` | 1.65       | 62.0      | Average    | Female | Average female proportions                 |
| `tall_male`      | 1.93       | 88.0      | Mesomorph  | Male   | Tall male (95th percentile height, CDC)    |
| `petite_female`  | 1.55       | 52.0      | Average    | Female | Petite female (5th percentile height, CDC) |

### Age-Specific Presets

| Preset        | Height (m) | Mass (kg) | Build Type | Gender  | Description                                           |
| ------------- | ---------- | --------- | ---------- | ------- | ----------------------------------------------------- |
| `child_8yo`   | 1.28       | 26.0      | Ectomorph  | Neutral | 8-year-old child (CDC growth charts, 50th percentile) |
| `senior_70yo` | 1.70       | 72.0      | Average    | Neutral | 70-year-old senior (NHANES data, average)             |

### Sport-Specific Presets

| Preset              | Height (m) | Mass (kg) | Build Type | Gender | Description                                          |
| ------------------- | ---------- | --------- | ---------- | ------ | ---------------------------------------------------- |
| `golfer_pro`        | 1.83       | 82.0      | Mesomorph  | Male   | Professional golfer body type                        |
| `pro_golfer_male`   | 1.83       | 82.0      | Mesomorph  | Male   | Male professional golfer (PGA Tour anthropometry)    |
| `pro_golfer_female` | 1.68       | 64.0      | Mesomorph  | Female | Female professional golfer (LPGA Tour anthropometry) |

## Usage Examples

### Python API

```python
from humanoid_character_builder.presets.loader import load_body_preset, get_preset_info

# Load a preset
body_params = load_body_preset("athletic")

# Get preset information
info = get_preset_info("athletic")
print(info["description"])
print(info["citation"])

# Override specific parameters
body_params = load_body_preset("athletic", height_m=1.85, mass_kg=85.0)
```

### CLI

```bash
# Build a character using a preset
python -m humanoid_character_builder build --preset athletic --output my_character.urdf

# List available presets
python -m humanoid_character_builder presets list
```

## Anthropometric Sources

### CDC Growth Charts

The CDC Growth Charts are used for pediatric anthropometric data. These charts represent national reference data for children in the United States.

- **Source**: CDC Growth Charts 2000
- **URL**: https://www.cdc.gov/growthcharts/cdc_charts.htm
- **Used by**: `child_8yo`

### NHANES Data

The National Health and Nutrition Examination Survey (NHANES) provides comprehensive anthropometric data for the US population.

- **Source**: NHANES Anthropometric Data
- **URL**: https://www.cdc.gov/nchs/nhanes/
- **Used by**: `senior_70yo`

### CDC Anthropometric Reference Data

CDC Anthropometric Reference Data provides statistics for adult populations including percentile distributions.

- **Source**: CDC Anthropometric Reference Data (Series 3, Number 39)
- **URL**: https://www.cdc.gov/nchs/data/series/sr_03/sr03_039.pdf
- **Used by**: `tall_male`, `petite_female`

### Sport-Specific Studies

Professional golfer anthropometry is derived from published sports science research.

- **Source**: Hume, P. A., et al. (2005). Anthropometric profiling of professional golfers.
- **Used by**: `pro_golfer_male`, `pro_golfer_female`, `golfer_pro`

## Build Types

The character builder supports three somatotype build categories:

| Build Type  | Description           | Characteristics                                 |
| ----------- | --------------------- | ----------------------------------------------- |
| `ECTOMORPH` | Lean and slender      | Lower body mass, longer limbs relative to torso |
| `MESOMORPH` | Muscular and athletic | Higher muscularity, broader shoulders           |
| `ENDOMORPH` | Rounder and softer    | Higher body fat, wider hips                     |

## Custom Presets

You can create custom presets by saving `BodyParameters` to a YAML or JSON file:

```python
from humanoid_character_builder.presets.loader import save_preset_to_file
from humanoid_character_builder.core.body_parameters import BodyParameters, BuildType, GenderModel

params = BodyParameters(
    name="custom_athlete",
    height_m=1.85,
    mass_kg=85.0,
    build_type=BuildType.MESOMORPH,
    gender_model=GenderModel.MALE,
    muscularity=0.75,
    body_fat_factor=0.10,
)

save_preset_to_file(params, "presets/my_custom_preset.yaml")
```

## Expected Use Cases

| Use Case                        | Recommended Presets                                |
| ------------------------------- | -------------------------------------------------- |
| General biomechanics simulation | `average`, `male_average`, `female_average`        |
| Sports performance analysis     | `athletic`, `pro_golfer_male`, `pro_golfer_female` |
| Pediatric studies               | `child_8yo`                                        |
| Geriatric studies               | `senior_70yo`                                      |
| Ergonomics testing              | `compact`, `petite_female`, `tall_male`            |
| Lightweight robotics            | `minimal`, `lean`                                  |
| Heavy load testing              | `heavy`                                            |

## Contributing New Presets

To contribute new presets:

1. Create the preset definition with anthropometrically valid parameters
2. Document the data source with a citable reference
3. Add the preset to `src/shared/python/humanoid_character_builder/presets/loader.py`
4. Update this documentation with the preset details
5. Add tests for the new preset
