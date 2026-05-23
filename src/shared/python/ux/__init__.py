"""UX infrastructure for the Idiot-Proof UX epic (#5968).

This package holds the pure-Python, framework-agnostic foundation that
PyQt6 widgets and React components both consume:

* :mod:`field_metadata` — typed metadata for every user-facing input
  (label, tooltip, units, valid range, default, producers/consumers).
* :mod:`provenance` — ``ProvenanceRecord`` describing where any
  displayed value came from (formula, inputs, run id).
* :mod:`preflight` — checklist + severity model for pre-action gates.
* :mod:`error_envelope` — ``UserFacingError`` envelope replacing raw
  exception strings at the API boundary.

No Qt or React imports live here; widget/component wrappers in
``src.shared.python.ui`` and ``ui/src/components/ux`` consume this
package.
"""

from src.shared.python.ux.error_envelope import (
    ErrorCatalog,
    UserFacingError,
    UserFacingErrorError,
    load_error_catalog,
)
from src.shared.python.ux.field_metadata import (
    FieldMetadata,
    FieldMetadataError,
    FieldRegistry,
    load_registry,
)
from src.shared.python.ux.preflight import (
    PreflightCheck,
    PreflightError,
    PreflightResult,
    Severity,
    run_preflight,
)
from src.shared.python.ux.provenance import (
    ProvenanceError,
    ProvenanceRecord,
    ProvenanceValue,
)

__all__ = [
    "ErrorCatalog",
    "FieldMetadata",
    "FieldMetadataError",
    "FieldRegistry",
    "PreflightCheck",
    "PreflightError",
    "PreflightResult",
    "ProvenanceError",
    "ProvenanceRecord",
    "ProvenanceValue",
    "Severity",
    "UserFacingError",
    "UserFacingErrorError",
    "load_error_catalog",
    "load_registry",
    "run_preflight",
]
