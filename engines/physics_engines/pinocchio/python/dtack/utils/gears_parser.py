"""Parser for Gears Motion Capture files (.gpcap)."""

from __future__ import annotations

import logging
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

logger = logging.getLogger(__name__)


class GearsParser:
    """Parser for proprietary Gears .gpcap binary files."""

    @staticmethod
    def load(file_path: Path | str) -> dict[str, npt.NDArray[np.float64]]:
        """Load .gpcap file.

        Analysis of file format (from probe):
        - Binary format with mixed ASCII/Wide-char strings.
        - Contains 'Skeleton' header.
        - Contains marker names like 'WaistLeft', 'WaistRight', 'HeadTop'.
        - Data appears to be float32 or float64 streams interleaved or following.

        Currently this parser is a STUB. Full reverse engineering of the binary
        layout is required, or a vendor DLL.

        Args:
            file_path: Path to .gpcap file

        Returns:
            Dictionary with 'markers' (Dict[str, array]).

        Raises:
            RuntimeError: Always raised until implementation is complete.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            msg = f"File not found: {file_path}"
            raise FileNotFoundError(msg)

        logger.warning("GearsParser is experimental/stub.")

        # NOTE: (Future Feature) Implement binary parsing based on offsets found.
        # Structure seems to be: [Len][String: MarkerName] ... [Data]

        msg = (
            "Gears .gpcap parser not yet implemented. "
            "File format requires reverse engineering. "
            "Please convert to C3D or MAT using Gears software."
        )
        raise RuntimeError(msg)
