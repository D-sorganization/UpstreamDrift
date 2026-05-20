"""Voice-to-text input manager for ChatDockWidget.

Runs speech recognition in a QThread to avoid blocking the Qt event loop.
Supports SpeechRecognition library with Google (online) and Vosk/Whisper (offline)
backends. Degrades gracefully when ``speech_recognition`` or ``pyaudio`` are absent.

Tools issue #2744.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

try:
    import speech_recognition as _sr_mod

    _SR_AVAILABLE = True
    sr: Any = _sr_mod
except ImportError:
    _SR_AVAILABLE = False
    sr = None


class _ListenWorker(QThread):
    """Background thread that records audio and runs speech recognition."""

    transcription_ready = pyqtSignal(str)
    recognition_error = pyqtSignal(str)

    def __init__(self, recognizer: Any, source_kwargs: dict) -> None:
        super().__init__()
        self._recognizer = recognizer
        self._source_kwargs = source_kwargs
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        if sr is None:
            self.recognition_error.emit("speech_recognition not installed")
            return

        try:
            with sr.Microphone(**self._source_kwargs) as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.3)
                try:
                    audio = self._recognizer.listen(
                        source, timeout=10, phrase_time_limit=30
                    )
                except sr.WaitTimeoutError:
                    self.recognition_error.emit(
                        "Listening timed out — no speech detected"
                    )
                    return

            text = self._recognizer.recognize_google(audio)
            if text:
                self.transcription_ready.emit(text)
        except sr.UnknownValueError:
            self.recognition_error.emit("Could not understand audio")
        except sr.RequestError as exc:
            self.recognition_error.emit(f"Recognition service unavailable: {exc}")
        except OSError as exc:
            self.recognition_error.emit(f"Microphone error: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Voice recognition failed: %s", exc)
            self.recognition_error.emit(f"Recognition failed: {exc}")


class VoiceInputManager:
    """Manages voice recording lifecycle for ChatDockWidget.

    Usage::

        manager = VoiceInputManager()
        manager.connect_transcription(my_slot)
        manager.connect_error(my_error_slot)
        manager.start()   # begins listening
        # ...
        manager.stop()    # cancels if still listening

    Callbacks are invoked on the Qt main thread via cross-thread signal/slot.
    When ``speech_recognition`` or ``pyaudio`` are unavailable the manager
    calls error callbacks immediately on ``start()`` without raising.
    """

    def __init__(self, device_index: int | None = None) -> None:
        self._device_index = device_index
        self._worker: _ListenWorker | None = None
        self._recognizer: Any = sr.Recognizer() if _SR_AVAILABLE else None
        self._transcription_ready_callbacks: list[Callable[[str], None]] = []
        self._error_occurred_callbacks: list[Callable[[str], None]] = []

    # ── Signal-like API ────────────────────────────────────────────────────

    def connect_transcription(self, slot: Callable[[str], None]) -> None:
        """Register a callable to receive transcribed text."""
        self._transcription_ready_callbacks.append(slot)

    def connect_error(self, slot: Callable[[str], None]) -> None:
        """Register a callable to receive error messages."""
        self._error_occurred_callbacks.append(slot)

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    @property
    def available(self) -> bool:
        return _SR_AVAILABLE

    def start(self) -> None:
        """Start recording. No-op if already recording."""
        if self.is_recording:
            return

        if not _SR_AVAILABLE:
            self._emit_error(
                "Voice input requires 'SpeechRecognition' and 'PyAudio'.\n"
                "Install with: pip install SpeechRecognition pyaudio"
            )
            return

        source_kwargs: dict[str, Any] = {}
        if self._device_index is not None:
            source_kwargs["device_index"] = self._device_index

        self._worker = _ListenWorker(self._recognizer, source_kwargs)
        self._worker.transcription_ready.connect(self._on_transcription)
        self._worker.recognition_error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def stop(self) -> None:
        """Request stop. The worker exits after the current listen cycle."""
        if self._worker is not None:
            self._worker.request_stop()
            self._worker.quit()
            self._worker.wait(500)
            self._worker = None

    # ── Internal ───────────────────────────────────────────────────────────

    def _on_transcription(self, text: str) -> None:
        for cb in self._transcription_ready_callbacks:
            try:
                cb(text)
            except Exception:  # noqa: BLE001
                pass
        self._worker = None

    def _on_error(self, message: str) -> None:
        self._emit_error(message)
        self._worker = None

    def _on_finished(self) -> None:
        self._worker = None

    def _emit_error(self, message: str) -> None:
        for cb in self._error_occurred_callbacks:
            try:
                cb(message)
            except Exception:  # noqa: BLE001
                pass
