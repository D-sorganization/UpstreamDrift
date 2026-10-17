"""Reader of the open-fp swing file format and its LAN client (OF-02/OF-06, #13061).

Reads the documented ``<id>.json`` + ``<id>.npz`` swing pair and its CSV export
into a :class:`~src.shared.python.force_plates.contracts.VerticalPlateTrial`.
The read-only :class:`OpenFpClient` (OF-06) fetches the same swings over HTTP.
Depends on the force-plate contract, never the reverse.  open-fp is AGPL-3.0;
this package implements the documented file format and imports no open-fp code.
"""

from __future__ import annotations

from src.shared.python.force_plates.open_fp_format.client import (
    HTTP_TIME_QUANTUM_S,
    OpenFpClient,
    OpenFpClientError,
    UnknownSwingError,
)
from src.shared.python.force_plates.open_fp_format.reader import (
    PlateDeclarations,
    list_swings,
    read_swing,
    read_swing_csv,
)

__all__ = [
    "HTTP_TIME_QUANTUM_S",
    "OpenFpClient",
    "OpenFpClientError",
    "PlateDeclarations",
    "UnknownSwingError",
    "list_swings",
    "read_swing",
    "read_swing_csv",
]
