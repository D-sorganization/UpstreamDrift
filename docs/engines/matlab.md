# MATLAB Engine

The MATLAB integration allows the Golf Modeling Suite to leverage existing Simscape Multibody models and high-fidelity legacy code.

## ✨ Key Features

- **Simscape Multibody**: Physical modeling blocks for biomechanics.
- **Legacy Compatibility**: Reuse of validated specialized golf models.
- **Symbolic Math**: MATLAB's symbolic toolbox capabilities.

## ⚠️ Prerequisites

To use the MATLAB engine, you must have:

1. **MATLAB Installation**: R2022b or later recommended.
2. **MATLAB Engine API for Python**: Installed via `pip install matlabengine` or from the MATLAB root.

## 📁 Directory Structure

```
src/engines/Simscape_Multibody_Models/
├── 2D_Golf_Model/
│   ├── matlab/      # 2D planar models (GolfSwing.slx, ...)
│   └── python/      # Python bridge code
└── 3D_Golf_Model/
    ├── matlab/      # 3D spatial models
    └── python/      # Python bridge code
```

## Integration Details

The suite starts a MATLAB session in the background `matlab.engine.start_matlab()`. This process can take 30-60 seconds to initialize.

### Data Exchange

Data is passed between Python and MATLAB using:

- **Inputs**: Scalars and Lists converted to MATLAB Arrays.
- **Outputs**: MATLAB Arrays converted back to NumPy/Lists.

### Performance Note

Due to the overhead of IPC (Inter-Process Communication), the MATLAB engine is best used for high-fidelity, single-run validations rather than tight-loop optimization.
