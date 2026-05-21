"""Deprecated alias for the sidekick package.

The runtime code lives at ``sidekick.*``. This shim re-exports every
public symbol so existing ``from upstream_drift_tools.X import Y``
imports continue to work during the migration window. A
``DeprecationWarning`` is emitted on first import.

Downstream consumers should migrate to ``sidekick`` at their own pace.
This shim will be removed in a future major release (tracked by a
separate issue).
"""

# isort: skip_file
import sys
import warnings

warnings.warn(
    "upstream_drift_tools is deprecated and will be removed in a future release. "
    "Import from sidekick instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import the canonical package so all its submodules are registered.
import sidekick  # noqa: E402
import sidekick.calculators as calculators  # noqa: E402, F401
import sidekick.data_processing as data_processing  # noqa: E402, F401
import sidekick.lab as lab  # noqa: E402, F401
import sidekick.process_calculators as process_calculators  # noqa: E402, F401
import sidekick.theme as theme  # noqa: E402, F401
import sidekick.ui as ui  # noqa: E402, F401
import sidekick.utils as utils  # noqa: E402, F401
from sidekick import (  # noqa: E402, F401
    CalculationResult,
    Calculator,
    DataTransformer,
    InputValidator,
    ProcessCalculator,
    StateSerializable,
    UnitConverter,
    ValidationResult,
)

# Mirror every sidekick.* module already in sys.modules under the old name.
# This makes `import upstream_drift_tools.X` and
# `from upstream_drift_tools.X import Y` resolve to the canonical objects.
_PREFIX = "sidekick."
_OLD_PREFIX = "upstream_drift_tools."
for _name, _mod in list(sys.modules.items()):
    if _name.startswith(_PREFIX):
        _alias = _OLD_PREFIX + _name[len(_PREFIX) :]
        sys.modules.setdefault(_alias, _mod)

__version__ = sidekick.__version__

__all__ = [
    # Protocols
    "Calculator",
    "DataTransformer",
    "ProcessCalculator",
    "StateSerializable",
    "UnitConverter",
    # Data classes
    "CalculationResult",
    "ValidationResult",
    # Validation
    "InputValidator",
    # Subpackages (explicit for discovery)
    "calculators",
    "data_processing",
    "lab",
    "process_calculators",
    "theme",
    "ui",
    "utils",
]
