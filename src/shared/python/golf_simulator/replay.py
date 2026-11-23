"""Monotonic replay clock and single-impact submission coordinator.

Follows TDD, DbC, Law of Demeter, and DRY.
Acceptance criteria from Issue #10195 (GS-06).
"""

from __future__ import annotations

import datetime
import logging
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from src.shared.python.golf_simulator.contracts import (
    PreparedShot,
    ReplayFrame,
    ReplayPlaybackState,
    ReplayTimingRecord,
    SessionState,
    SubmissionReceipt,
)

if TYPE_CHECKING:
    from src.shared.python.golf_simulator.session import GolfSessionService

logger = logging.getLogger(__name__)


class MonotonicReplayClock:
    """Presentation clock driven by a monotonic time source.

    Supports play, pause, seek, and playback rate adjustments.
    Invariants:
    - Time advancements during play are monotonic.
    - Seeking sets a distinct flag so coordinators can detect scrub actions.
    - Playback rate must be strictly positive.
    """

    def __init__(
        self,
        time_fn: Callable[[], float] | None = None,
        initial_time_s: float = 0.0,
    ) -> None:
        self._time_fn = time_fn or time.monotonic
        if initial_time_s < 0.0:
            raise ValueError("initial_time_s must be non-negative")
        self._current_time_s: float = float(initial_time_s)
        self._playback_rate: float = 1.0
        self._state: ReplayPlaybackState = ReplayPlaybackState.STOPPED
        self._last_real_time: float | None = None
        self._was_seek: bool = False

    @property
    def current_time_s(self) -> float:
        """Return the current replay time in seconds."""
        return self._current_time_s

    @property
    def playback_state(self) -> ReplayPlaybackState:
        """Return the current playback state."""
        return self._state

    @property
    def playback_rate(self) -> float:
        """Return the current playback speed multiplier."""
        return self._playback_rate

    def set_playback_rate(self, rate: float) -> None:
        """Update playback rate multiplier (e.g. 0.5 for slow motion, 1.0 for normal)."""
        if rate <= 0.0:
            raise ValueError(f"playback_rate must be strictly positive, got {rate}")
        self._playback_rate = float(rate)

    def play(self) -> None:
        """Start or resume monotonic playback."""
        self._state = ReplayPlaybackState.PLAYING
        self._last_real_time = self._time_fn()

    def pause(self) -> None:
        """Pause playback at the current replay time."""
        self._state = ReplayPlaybackState.PAUSED
        self._last_real_time = None

    def stop(self) -> None:
        """Stop playback and reset replay time to zero."""
        self._state = ReplayPlaybackState.STOPPED
        self._current_time_s = 0.0
        self._last_real_time = None

    def seek(self, target_time_s: float) -> None:
        """Seek directly to a target replay timestamp."""
        if target_time_s < 0.0:
            raise ValueError("target_time_s must be non-negative")
        self._pre_seek_time_s: float = self._current_time_s
        self._current_time_s = float(target_time_s)
        self._was_seek = True
        self._last_real_time = self._time_fn()

    def tick(self) -> tuple[float, float, bool]:
        """Advance the clock according to elapsed real time and return (t_prev, t_curr, was_seek)."""
        was_seek = self._was_seek
        self._was_seek = False
        if was_seek:
            t_prev = getattr(self, "_pre_seek_time_s", self._current_time_s)
        else:
            t_prev = self._current_time_s

        if self._state != ReplayPlaybackState.PLAYING:
            return t_prev, self._current_time_s, was_seek

        now_real = self._time_fn()
        if self._last_real_time is not None:
            delta_real = max(0.0, now_real - self._last_real_time)
            delta_replay = delta_real * self._playback_rate
            self._current_time_s += delta_replay

        self._last_real_time = now_real
        return t_prev, self._current_time_s, was_seek


class ReplaySubmissionCoordinator:
    """Coordinates presentation replay with single-impact shot submission.

    Guarantees:
    - Exactly one shot submission when replay monotonically crosses impact time.
    - Pausing or seeking across impact automatically disarms the prepared shot.
    - Dropped presentation frames or variable tick rates do not duplicate submissions.
    - Records latency audit metrics (impact-to-send and send-to-response).

    ``impact_to_send_latency_ms`` is wall-clock time from the scheduled impact
    instant on the replay timeline to the send attempt: replay frame
    quantization ``(t_curr - impact_time_s) / playback_rate`` plus software
    path overhead from impact-cross detection to send attempt
    ``t_trigger - t_detect``.
    """

    def __init__(
        self,
        session_service: GolfSessionService,
        clock: MonotonicReplayClock,
        time_fn: Callable[[], float] | None = None,
        realtime_publisher: Callable[[str, Any], None] | None = None,
    ) -> None:
        self._service = session_service
        self._clock = clock
        self._time_fn = time_fn or time.monotonic
        self._realtime_publisher = realtime_publisher

        self._prepared_shot: PreparedShot | None = None
        self._arm_token: str | None = None
        self._impact_time_s: float | None = None
        self._has_triggered: bool = False
        self._last_timing_record: ReplayTimingRecord | None = None

    @property
    def is_armed(self) -> bool:
        """Return True if armed for submission upon impact crossing."""
        return (
            self._prepared_shot is not None
            and self._arm_token is not None
            and not self._has_triggered
        )

    @property
    def has_triggered(self) -> bool:
        """Return True if the impact event was already submitted."""
        return self._has_triggered

    @property
    def last_timing_record(self) -> ReplayTimingRecord | None:
        """Return the most recent latency timing record."""
        return self._last_timing_record

    def arm_submission(
        self,
        prepared_shot: PreparedShot,
        arm_token: str,
    ) -> None:
        """Arm the coordinator with an active prepared shot and single-use arm token."""
        if not isinstance(prepared_shot, PreparedShot):
            raise TypeError("prepared_shot must be a PreparedShot instance")
        if not arm_token or not str(arm_token).strip():
            raise ValueError("arm_token must be a non-empty string")

        shot = prepared_shot.shot
        if shot.impact_time_s is None:
            raise ValueError(
                f"Shot {shot.shot_id} has no impact_time_s; cannot coordinate replay submission"
            )

        self._prepared_shot = prepared_shot
        self._arm_token = arm_token
        self._shot_id: str | None = shot.shot_id
        self._impact_time_s = shot.impact_time_s
        self._has_triggered = False

    def disarm(self) -> None:
        """Disarm the coordinator and notify session service."""
        if self._prepared_shot is not None and self.is_armed:
            try:
                self._service.disarm(self._prepared_shot.prepared_shot_id)
            except Exception as exc:
                logger.warning("Error disarming session service: %s", exc)
        self._arm_token = None
        self._prepared_shot = None
        self._shot_id = None
        self._impact_time_s = None

    async def tick(self) -> SubmissionReceipt | None:
        """Evaluate the replay clock tick and submit shot if impact instant is crossed."""
        t_prev, t_curr, was_seek = self._clock.tick()

        # Telemetry broadcast (best-effort presentation)
        if self._realtime_publisher is not None:
            sess_id = self._service.session_id
            payload = {
                "t_curr": t_curr,
                "is_armed": self.is_armed,
                "has_triggered": self._has_triggered,
            }
            try:
                self._realtime_publisher(
                    f"golf_simulator/replay/{sess_id}/state", payload
                )
            except Exception as exc:
                logger.debug("Realtime publisher error: %s", exc)

        if not self.is_armed or self._impact_time_s is None or self._has_triggered:
            return None

        # If a seek occurred across impact time, disarm immediately to prevent accidental submission
        if was_seek:
            if (t_prev < self._impact_time_s <= t_curr) or (
                t_curr <= self._impact_time_s <= t_prev
            ):
                logger.info(
                    "Seek across impact time %.3fs detected; disarming submission",
                    self._impact_time_s,
                )
                self.disarm()
                return None

        # Normal forward playback crossing impact
        if t_prev < self._impact_time_s <= t_curr:
            t_detect = self._time_fn()
            receipt = await self._execute_impact_submission(
                t_detect_wall_s=t_detect,
                replay_time_at_cross_s=t_curr,
            )
            return receipt

        return None

    async def _execute_impact_submission(
        self,
        *,
        t_detect_wall_s: float,
        replay_time_at_cross_s: float,
    ) -> SubmissionReceipt:
        """Execute single-impact submission and measure latencies."""
        assert self._prepared_shot is not None
        assert self._arm_token is not None
        assert self._impact_time_s is not None
        assert self._shot_id is not None

        prep_id = self._prepared_shot.prepared_shot_id
        token = self._arm_token
        shot_id = self._shot_id
        impact_t = self._impact_time_s

        if not math.isfinite(t_detect_wall_s):
            msg = f"t_detect_wall_s must be finite, got {t_detect_wall_s!r}"
            raise ValueError(msg)
        if not math.isfinite(replay_time_at_cross_s):
            msg = (
                f"replay_time_at_cross_s must be finite, got {replay_time_at_cross_s!r}"
            )
            raise ValueError(msg)

        rate = self._clock.playback_rate
        if rate <= 0.0:
            raise ValueError(f"playback_rate must be strictly positive, got {rate}")

        t_trigger = self._time_fn()
        if not math.isfinite(t_trigger):
            msg = f"t_trigger must be finite, got {t_trigger!r}"
            raise ValueError(msg)
        send_attempt_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Execute submission
        receipt = await self._service.submit_at_impact(
            prepared_shot_id=prep_id,
            arm_token=token,
        )

        t_done = self._time_fn()
        if not math.isfinite(t_done):
            msg = f"t_done must be finite, got {t_done!r}"
            raise ValueError(msg)
        response_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        replay_lag_s = max(0.0, (replay_time_at_cross_s - impact_t) / rate)
        software_lag_s = max(0.0, t_trigger - t_detect_wall_s)
        impact_to_send_ms = round((replay_lag_s + software_lag_s) * 1000.0, 6)
        send_to_response_ms = round(max(0.0, (t_done - t_trigger) * 1000.0), 6)

        if not math.isfinite(impact_to_send_ms):
            raise ValueError("impact_to_send_latency_ms must be finite")
        if not math.isfinite(send_to_response_ms):
            raise ValueError("send_to_response_latency_ms must be finite")

        record = ReplayTimingRecord(
            shot_id=shot_id,
            impact_clock_time_s=impact_t,
            send_attempt_timestamp_utc=send_attempt_utc,
            response_timestamp_utc=response_utc,
            impact_to_send_latency_ms=impact_to_send_ms,
            send_to_response_latency_ms=send_to_response_ms,
        )
        self._last_timing_record = record
        self._has_triggered = True
        self._arm_token = None

        return receipt
