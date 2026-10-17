"""Reader of the open-fp swing file format and its LAN client (OF-02/OF-06/OF-09, #13061).

Reads the documented ``<id>.json`` + ``<id>.npz`` swing pair and its CSV export
into a :class:`~src.shared.python.force_plates.contracts.VerticalPlateTrial`.
The read-only :class:`OpenFpClient` (OF-06) fetches the same swings over HTTP.  The optional ``LiveBridge`` (OF-09) relays the live
Socket.IO feed to realtime IPC and never writes to the board.
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
from src.shared.python.force_plates.open_fp_format.live import (
    LIVE_CHANNEL,
    BackoffPolicy,
    LiveBridge,
    backoff_delays,
    register_live_channel,
)
from src.shared.python.force_plates.open_fp_format.reader import (
    PlateDeclarations,
    list_swings,
    read_swing,
    read_swing_csv,
)

__all__ = [
    "HTTP_TIME_QUANTUM_S",
    "LIVE_CHANNEL",
    "BackoffPolicy",
    "LiveBridge",
    "OpenFpClient",
    "OpenFpClientError",
    "PlateDeclarations",
    "UnknownSwingError",
    "backoff_delays",
    "list_swings",
    "read_swing",
    "register_live_channel",
    "read_swing_csv",
]
