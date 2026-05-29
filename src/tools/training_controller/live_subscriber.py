"""Realtime-channel subscriber that decodes training-progress payloads.

The training backend publishes per-job metric and status updates onto
``training/<job_id>/progress`` via
:class:`training.runtime.RealtimeChannelProgressSink`. The dashboard's
GUI follow-up will not bind to that channel directly; instead it
constructs a :class:`TrainingJobLiveSubscriber`, registers two
callbacks (one for metrics, one for status), and lets this module own
the JSON decoding, validation, and lifecycle.

Keeping the realtime / decode glue in a pure-Python module means the
follow-up PyQt PR only has to marshal callback invocations onto the GUI
thread — it does not need to know the wire format.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.training import TrainingStatus
from src.shared.python.training.persistence import training_metric_from_dict
from src.shared.python.training.metrics import TrainingMetric
from src.shared.python.training.runtime.progress_sinks import training_channel_for

__all__ = [
    "MetricCallback",
    "StatusCallback",
    "TrainingJobLiveSubscriber",
]


logger = get_logger(__name__)


MetricCallback = Callable[[TrainingMetric], None]
"""Callback fired for every decoded :class:`TrainingMetric`."""

StatusCallback = Callable[[TrainingStatus, "str | None"], None]
"""Callback fired for every decoded status change.

The second positional argument is the optional message attached to the
status emission (e.g. failure reason). ``None`` when not supplied.
"""


class TrainingJobLiveSubscriber:
    """Subscribes to a job's realtime progress channel.

    The subscription is created lazily inside :meth:`start` so importing
    this module never touches :mod:`src.shared.python.realtime` — that
    keeps the dashboard's startup cost flat and lets headless tests
    patch the realtime facade with a stub before the subscriber is
    instantiated.

    Args:
        job_id: Identifier of the job whose progress this subscriber
            consumes. Used to derive the channel name via
            :func:`training_channel_for`.
        on_metric: Callback fired for each decoded
            :class:`TrainingMetric`. Optional — leave ``None`` to drop
            metric events.
        on_status: Callback fired for each decoded status change.
            Optional — leave ``None`` to drop status events.

    Raises:
        ValueError: When ``job_id`` is empty.
        TypeError: When the callbacks are not callable.
    """

    __slots__ = (
        "_channel",
        "_job_id",
        "_lock",
        "_on_metric",
        "_on_status",
        "_subscription",
    )

    def __init__(
        self,
        job_id: str,
        *,
        on_metric: MetricCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> None:
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")
        if on_metric is not None and not callable(on_metric):
            raise TypeError("on_metric must be callable or None")
        if on_status is not None and not callable(on_status):
            raise TypeError("on_status must be callable or None")
        self._job_id = job_id
        self._channel = training_channel_for(job_id)
        self._on_metric = on_metric
        self._on_status = on_status
        self._subscription: Any | None = None
        self._lock = threading.Lock()

    @property
    def job_id(self) -> str:
        """Job identifier this subscriber is bound to."""

        return self._job_id

    @property
    def channel(self) -> str:
        """Canonical realtime channel name (see :func:`training_channel_for`)."""

        return self._channel

    @property
    def is_started(self) -> bool:
        """``True`` while a realtime subscription is active."""

        with self._lock:
            return self._subscription is not None

    def start(self) -> None:
        """Register the realtime subscription. Idempotent.

        Imports :mod:`src.shared.python.realtime` lazily so the
        subscriber stays headless-safe at import time.
        """

        from src.shared.python import realtime  # noqa: PLC0415 - lazy

        with self._lock:
            if self._subscription is not None:
                return
            # Re-register the channel so subscribers can boot up before the
            # producer side. ``register_channel`` is idempotent for matching
            # descriptors.
            realtime.register_channel(
                self._channel,
                description=(f"Training-job progress stream for {self._job_id!r}"),
            )
            self._subscription = realtime.subscribe(self._channel, self._dispatch)

    def stop(self) -> None:
        """Tear the subscription down. Idempotent; safe to call from any thread."""

        with self._lock:
            subscription = self._subscription
            self._subscription = None
        if subscription is None:
            return
        unsubscribe = getattr(subscription, "unsubscribe", None)
        if unsubscribe is None:
            return
        try:
            unsubscribe()
        except (RuntimeError, OSError, ValueError):
            logger.exception(
                "TrainingJobLiveSubscriber.stop: unsubscribe failed for %s",
                self._channel,
            )

    def _dispatch(self, payload: Any) -> None:
        """Decode and fan a single realtime payload to the user callbacks."""

        if not isinstance(payload, dict):
            logger.debug(
                "training subscriber: dropping non-dict payload on %s",
                self._channel,
            )
            return
        event = payload.get("event")
        if event == "metric":
            self._dispatch_metric(payload)
        elif event == "status":
            self._dispatch_status(payload)
        else:
            logger.debug(
                "training subscriber: dropping unknown event %r on %s",
                event,
                self._channel,
            )

    def _dispatch_metric(self, payload: dict[str, Any]) -> None:
        callback = self._on_metric
        if callback is None:
            return
        body = payload.get("metric")
        if not isinstance(body, dict):
            logger.debug(
                "training subscriber: metric event missing 'metric' dict on %s",
                self._channel,
            )
            return
        try:
            metric = training_metric_from_dict(body)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning(
                "training subscriber: failed to decode metric on %s: %s",
                self._channel,
                exc,
            )
            return
        try:
            callback(metric)
        except (RuntimeError, ValueError, TypeError, OSError, LookupError):
            logger.exception(
                "training subscriber on_metric callback raised for %s",
                self._channel,
            )

    def _dispatch_status(self, payload: dict[str, Any]) -> None:
        callback = self._on_status
        if callback is None:
            return
        status_raw = payload.get("status")
        if not isinstance(status_raw, str):
            logger.debug(
                "training subscriber: status event missing 'status' string on %s",
                self._channel,
            )
            return
        try:
            status = TrainingStatus(status_raw)
        except ValueError:
            logger.warning(
                "training subscriber: unknown status %r on %s",
                status_raw,
                self._channel,
            )
            return
        message_raw = payload.get("message")
        message: str | None
        if message_raw is None:
            message = None
        elif isinstance(message_raw, str):
            message = message_raw
        else:
            logger.debug(
                "training subscriber: ignoring non-string status message on %s",
                self._channel,
            )
            message = None
        try:
            callback(status, message)
        except (RuntimeError, ValueError, TypeError, OSError, LookupError):
            logger.exception(
                "training subscriber on_status callback raised for %s",
                self._channel,
            )
