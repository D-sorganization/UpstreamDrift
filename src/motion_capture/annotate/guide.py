"""The guided cursor: which joint on which frame is asked next (#9799).

A :class:`Guide` walks a frame range on a stride, asking for every joint of
a frame in order before moving on. The user accepts (a click), skips (the
joint is occluded), goes back, jumps to a frame, or moves to the next frame
leaving the rest of the current frame unannotated. Every action is a plain
method so the widget stays a thin mapping from clicks and keys. Qt-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.shared.python.core.contracts import require

from .store import AnnotationSet

KEYMAP: dict[str, str] = {
    "s": "skip",
    "b": "back",
    "n": "next_frame",
    "j": "jump",
    "q": "finish",
}
MAX_HISTORY = 500


@dataclass(frozen=True)
class Prompt:
    frame: int
    joint: str
    text: str


class Guide:
    """Frame-by-frame, joint-by-joint prompting over a store."""

    def __init__(
        self,
        store: AnnotationSet,
        joints: Sequence[str] | None = None,
        frame_range: tuple[int, int] = (0, 0),
        stride: int = 1,
        *,
        only_missing: bool = True,
    ) -> None:
        self.store = store
        self.joints: tuple[str, ...] = tuple(joints or store.joints)
        unknown = [j for j in self.joints if j not in store.joints]
        require(not unknown, "joints must belong to the store", unknown)
        require(len(self.joints) >= 1, "at least one joint to track")
        first, last = frame_range
        require(0 <= first <= last, "frame_range must be 0 <= first <= last")
        require(stride >= 1, "stride must be >= 1", stride)
        self.first, self.last, self.stride = first, last, stride
        self.only_missing = only_missing
        self.frame, self.joint_index = first, 0
        self.finished = False
        self._history: list[tuple[int, int]] = []
        self._settle_forward()

    # -- state ---------------------------------------------------------------
    @property
    def joint(self) -> str:
        return self.joints[self.joint_index]

    def prompt(self) -> Prompt | None:
        """What to ask now, or ``None`` when finished."""
        if self.finished:
            return None
        text = (
            f"Frame {self.frame} · {self.joint} — click the joint, S skip (occluded), "
            f"B back, N next frame, J jump, Q finish"
        )
        return Prompt(self.frame, self.joint, text)

    def progress(self) -> tuple[int, int]:
        """``(prompts answered, prompts in the range)`` for the chosen joints."""
        frames = range(self.first, self.last + 1, self.stride)
        total = len(frames) * len(self.joints)
        done = sum(1 for f in frames for j in self.joints if self.store.has_entry(f, j))
        return done, total

    # -- actions -------------------------------------------------------------
    def accept(self, x_px: float, y_px: float) -> None:
        """Store the click for the current prompt and advance."""
        require(not self.finished, "guide is finished")
        self.store.set_point(self.frame, self.joint, x_px, y_px)
        self._advance()

    def skip(self) -> None:
        require(not self.finished, "guide is finished")
        self.store.skip(self.frame, self.joint)
        self._advance()

    def back(self) -> None:
        """Return to the previous prompt (its entry stays; re-answering replaces)."""
        if self._history:
            self.frame, self.joint_index = self._history.pop()
            self.finished = False

    def next_frame(self) -> None:
        """Leave the rest of this frame and move on by the stride."""
        require(not self.finished, "guide is finished")
        self._push()
        self.frame += self.stride
        self.joint_index = 0
        self._settle_forward()

    def jump(self, frame: int) -> None:
        """Move to ``frame`` (clamped to the range), first joint."""
        self._push()
        self.frame = min(max(int(frame), self.first), self.last)
        self.joint_index = 0
        self.finished = False
        self._settle_forward()

    def finish(self) -> None:
        self.finished = True

    def handle_key(self, key: str) -> bool:
        """Dispatch a single-character key via :data:`KEYMAP`; ``jump`` needs
        a frame and is reported but not performed here. Returns handled."""
        action = KEYMAP.get(key.lower())
        if action is None or action == "jump":
            return action is not None
        getattr(self, action)()
        return True

    # -- internals -----------------------------------------------------------
    def _push(self) -> None:
        self._history.append((self.frame, self.joint_index))
        del self._history[:-MAX_HISTORY]

    def _advance(self) -> None:
        self._push()
        self.joint_index += 1
        if self.joint_index >= len(self.joints):
            self.joint_index = 0
            self.frame += self.stride
        self._settle_forward()

    def _settle_forward(self) -> None:
        """Skip answered prompts (when ``only_missing``) and detect the end."""
        while self.frame <= self.last:
            if not (self.only_missing and self.store.has_entry(self.frame, self.joint)):
                return
            self.joint_index += 1
            if self.joint_index >= len(self.joints):
                self.joint_index = 0
                self.frame += self.stride
        self.finished = True
