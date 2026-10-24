"""Voice-to-text input manager for ChatDockWidget.

Runs speech recognition in a QThread to avoid blocking the Qt event loop.
Supports SpeechRecognition library with Google (online) and Vosk/Whisper (offline)
backends. Degrades gracefully when ``speech_recognition`` or ``pyaudio`` are absent.

Tools issue #2744.
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
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


def detect_local_whisper_engine() -> str:
    """Return active local Whisper engine (faster-whisper, whisper.cpp, or none)."""
    if importlib.util.find_spec("faster_whisper") is not None:
        return "faster-whisper"
    if shutil.which("whisper-cli") or shutil.which("whisper-cpp"):
        return "whisper.cpp"
    return "none"


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

    def whisper_status(self) -> dict[str, Any]:
        """Return structured status for local Whisper / voice input."""
        engine = detect_local_whisper_engine()
        is_avail = engine != "none" or _SR_AVAILABLE
        detail = (
            "ready"
            if is_avail
            else "voice_unavailable: install faster-whisper or SpeechRecognition"
        )
        return {
            "available": is_avail,
            "engine": engine
            if engine != "none"
            else ("speech_recognition" if _SR_AVAILABLE else "none"),
            "model": "base.en",
            "detail": detail,
        }

    def tooltip_text(self) -> str:
        """Return composer microphone tooltip reflecting local voice status."""
        status = self.whisper_status()
        if status["available"]:
            return f"Voice input ({status['engine']}, Ctrl+Shift+V)"
        return "Voice input unavailable — install faster-whisper (Ctrl+Shift+V)"

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> str:
        """Transcribe 16kHz PCM WAV bytes locally via faster-whisper or whisper.cpp."""
        if (
            not wav_bytes
            or len(wav_bytes) < 44
            or wav_bytes[:4] != b"RIFF"
            or wav_bytes[8:12] != b"WAVE"
        ):
            raise ValueError("Invalid WAV header: expected RIFF/WAVE magic bytes")
        engine = detect_local_whisper_engine()
        if engine == "none":
            raise RuntimeError("voice_unavailable: local Whisper engine not installed")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(wav_bytes)
        try:
            if engine == "faster-whisper":
                fw_mod = importlib.import_module("faster_whisper")
                model = fw_mod.WhisperModel(
                    "base.en", device="cpu", compute_type="int8"
                )
                segments, _ = model.transcribe(str(tmp_path), beam_size=1)
                return " ".join(
                    s.text.strip() for s in segments if s.text.strip()
                ).strip()
            binary = (
                shutil.which("whisper-cli")
                or shutil.which("whisper-cpp")
                or "whisper-cli"
            )
            proc = subprocess.run(
                [binary, "-f", str(tmp_path), "-nt"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"whisper-cli exited with {proc.returncode}")
            return proc.stdout.strip()
        finally:
            tmp_path.unlink(missing_ok=True)

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
