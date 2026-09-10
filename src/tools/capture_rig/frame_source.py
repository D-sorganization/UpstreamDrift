"""Minimal random-access image source shared by recordings and model playback."""

from typing import Protocol

import numpy as np
import numpy.typing as npt


class FrameSource(Protocol):
    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def fps(self) -> float: ...

    @property
    def frame_count(self) -> int: ...

    def read(self, index: int) -> npt.NDArray[np.uint8] | None: ...

    def close(self) -> None: ...
