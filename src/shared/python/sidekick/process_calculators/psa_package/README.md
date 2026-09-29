# PSA Analysis Package

Two-stage Pressure Swing Adsorption (PSA) system analysis tools for hydrogen purification.

## Contents

### Core Module

- `psa_model.py` - Core calculation module with algebraic PSA model
- `__init__.py` - Package exports

### User Interfaces

- `psa_gui.py` - PyQt6 desktop GUI for interactive analysis
- `psa_webapp.py` - Streamlit web application (password-protected)
- `psa_calculator.html` - Standalone HTML/JavaScript calculator (no server required)

### Analysis Notebooks

- `psa_analysis.ipynb` - Jupyter notebook with LaTeX equations and plots
- `psa_analysis_colab.ipynb` - Google Colab-compatible notebook (embedded model)

### Testing

- `test_psa_model.py` - Test suite (39 tests)

### Resources

- `PSA System PFD.jpg` - Process flow diagram

## Installation

```bash
# Install dependencies
pip install numpy matplotlib PyQt6

# Optional: For web app
pip install streamlit plotly pandas

# Optional: For notebooks
pip install jupyter
```

## Usage

### Desktop GUI

```bash
python psa_gui.py
```

### Web Application

```bash
streamlit run psa_webapp.py
# Default password: "password"
```

### HTML Calculator

Open `psa_calculator.html` in any web browser. No server required.
Default password: "password"

### Jupyter Notebook

```bash
jupyter notebook psa_analysis.ipynb
```

### Run Tests

```bash
python -m pytest test_psa_model.py -v
```

### Import as Module

```python
from psa_package import PSAModel, calculate_sensitivity

# Create model with default parameters
model = PSAModel(
    total_feed_scfm=1100.0,
    s2_tail_recycle_frac=1.0,
    product_recycle_frac=0.0,
)

# Calculate results
results = model.calculate()
print(f"H2 Recovery: {results.h2_recovery_pct:.2f}%")
print(f"H2 Purity: {results.h2_purity_pct:.4f}%")
print(f"Net Product: {results.total_net_product_scfm:.1f} SCFM")
```

## Model Validation

All calculations validated against Excel reference model within 1e-10 relative tolerance.

### Key Equation

The algebraic solution for mixed feed flow eliminates circular references:

```text
M_i = F_i / [1 - (1-R1_i) * (R2_i * r_tail + (1-R2_i) * r_prod)]
```

Where:

- `M_i` = Mixed feed flow for component i
- `F_i` = Fresh feed flow for component i
- `R1_i` = Stage 1 removal fraction for component i
- `R2_i` = Stage 2 removal fraction for component i
- `r_tail` = S2 tail recycle fraction
- `r_prod` = Product recycle fraction

## Features

### All Applications

- Auto-calculate on value changes (no calculate button needed)
- Pre-calculated plots for immediate display
- Configurable plot options (lines, markers, number of points)

### Safety Analysis

- O2 concentration tracking in S2 tail gas
- Flammability status indicators (Safe, Caution, Flammable, Critical)
- Critical threshold: O2 > 2% with H2 > 4%

### Sensitivity Analysis

- H2 recovery vs recycle fractions
- Net product flow vs recycle fractions
- O2 safety vs Stage 1 removal
- 3D surface plots
- Contour maps

## CI/CD Integration

### Linting

```bash
# Format code
ruff format psa_model.py psa_gui.py psa_webapp.py test_psa_model.py

# Check linting
ruff check psa_model.py psa_gui.py psa_webapp.py test_psa_model.py

# Type checking
mypy psa_model.py psa_gui.py psa_webapp.py test_psa_model.py
```

### Testing with Coverage

```bash
python -m pytest test_psa_model.py -v --cov=psa_model --cov-report=html
```

### Exported Symbols

The package exports the following from `psa_model.py`:

- `DEFAULT_COMPONENTS` - Default component data
- `ComponentData` - TypedDict for component properties
- `PSAModel` - Main calculation model
- `PSAResults` - Calculation results dataclass
- `StreamFlows` - Flow data dataclass
- `StreamCompositions` - Composition data dataclass
- `calculate_sensitivity` - Sensitivity analysis function
- `calculate_o2_safety_analysis` - O2 safety analysis function
- `get_flammability_status` - Flammability status checker

## Security Notes

The web app and HTML calculator use SHA-256 password hashing. To change the password:

```python
import hashlib
new_hash = hashlib.sha256("your_new_password".encode()).hexdigest()
# Update PASSWORD_HASH in psa_webapp.py and psa_calculator.html
```

## License

Internal use only. Contact administrator for distribution permissions.
