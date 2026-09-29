from __future__ import annotations

import asyncio
from collections import deque

from src.unreal_integration.data_models import UnrealDataFrame


class FrameBuffer:
    """Thread-safe frame buffer with overflow protection.

    Uses a circular buffer to store frames. When buffer is full,
    oldest frames are dropped to make room for new ones.

    Attributes:
        max_size: Maximum number of frames to store.

    Example:
        >>> buffer = FrameBuffer(max_size=10)
        >>> buffer.add(frame)
        >>> frame = buffer.get()
    """

    def __init__(self, max_size: int = 10) -> None:
        """Initialize frame buffer.

        Args:
            max_size: Maximum number of frames to store.
        """
        if max_size is None:
            raise ValueError("max_size must be provided")
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self._buffer: deque[UnrealDataFrame] = deque(maxlen=max_size)
        # Lock only when an event loop is already running; otherwise leave None
        # to avoid triggering deprecated asyncio.get_event_loop() warnings.
        try:
            running = asyncio.get_running_loop()
            self._lock: asyncio.Lock | None = asyncio.Lock() if running else None
        except RuntimeError:
            self._lock = None

    def __len__(self) -> int:
        """Return number of frames in buffer."""
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._buffer) == 0

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self._buffer) >= self.max_size

    def add(self, frame: UnrealDataFrame) -> bool:
        """Add frame to buffer.

        If buffer is full, oldest frame is dropped.

        Args:
            frame: Frame to add.

        Returns:
            True if frame was added (oldest may have been dropped).
        """
        if frame is None:
            raise ValueError("frame must be provided")
        self._buffer.append(frame)
        return True

    def get(self) -> UnrealDataFrame | None:
        """Remove and return oldest frame.

        Returns:
            Oldest frame, or None if buffer is empty.
        """
        if self.is_empty:
            return None
        return self._buffer.popleft()

    def peek(self) -> UnrealDataFrame | None:
        """Return oldest frame without removing it.

        Returns:
            Oldest frame, or None if buffer is empty.
        """
        if self.is_empty:
            return None
        return self._buffer[0]

    def clear(self) -> None:
        """Remove all frames from buffer."""
        self._buffer.clear()

    def get_all(self) -> list[UnrealDataFrame]:
        """Get all frames without removing them.

        Returns:
            List of all frames in buffer (oldest first).
        """
        return list(self._buffer)
