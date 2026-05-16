"""AppEvent dataclass — the atomic unit stored in HistoryStore."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AppEvent:
    """A single recorded application event.

    Attributes:
        type: Short string identifier (e.g. ``"simulation_run"``).
        timestamp: UTC datetime when the event was recorded.
        payload: Arbitrary key-value context for the event.
    """

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation.

        Returns:
            dict with ``type``, ``timestamp`` (ISO 8601), and ``payload``.
        """
        return {
            "type": self.type,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
        }
