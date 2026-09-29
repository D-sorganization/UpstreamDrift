"""Undo/redo stack of :class:`CanonicalPose` snapshots.

Pure-data — no Qt imports — so the unit tests can drive it directly.

Stack semantics:

* :meth:`push` records a snapshot.  Pushing after an undo discards the
  redo branch (standard editor behaviour).
* :meth:`undo` returns the previous snapshot, or ``None`` if the stack
  is at the bottom.
* :meth:`redo` returns the next snapshot, or ``None`` if there is no
  redo branch.

The controller never mutates :class:`CanonicalPose` instances; the
caller is responsible for treating returned poses as the authoritative
new state and re-applying them to the engine.
"""

from __future__ import annotations

from src.shared.python.pose_interchange.canonical import CanonicalPose


class HistoryController:
    """Bounded undo/redo stack of :class:`CanonicalPose` snapshots.

    Parameters
    ----------
    initial_pose
        The starting snapshot.  Cannot be undone past — the bottom of
        the stack always returns ``None`` from :meth:`undo` when the
        cursor would step below it.
    max_depth
        Soft cap on the number of snapshots retained.  Older entries
        are discarded once the stack exceeds this depth.  Must be at
        least ``2`` so a push-undo cycle is always possible.
    """

    def __init__(self, initial_pose: CanonicalPose, max_depth: int = 64) -> None:
        if not isinstance(initial_pose, CanonicalPose):
            raise TypeError(
                "initial_pose must be a CanonicalPose, "
                f"got {type(initial_pose).__name__}"
            )
        if not isinstance(max_depth, int):
            raise TypeError(f"max_depth must be int, got {type(max_depth).__name__}")
        if max_depth < 2:
            raise ValueError(f"max_depth must be >= 2, got {max_depth}")
        self._stack: list[CanonicalPose] = [initial_pose]
        self._cursor: int = 0
        self._max_depth: int = max_depth

    # ---- public surface ------------------------------------------------

    @property
    def can_undo(self) -> bool:
        """``True`` iff :meth:`undo` would return a snapshot."""
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        """``True`` iff :meth:`redo` would return a snapshot."""
        return self._cursor < len(self._stack) - 1

    @property
    def current(self) -> CanonicalPose:
        """The snapshot at the current stack cursor."""
        return self._stack[self._cursor]

    @property
    def depth(self) -> int:
        """Total number of snapshots currently retained."""
        return len(self._stack)

    def push(self, pose: CanonicalPose) -> None:
        """Push *pose* onto the stack.

        Discards any redo branch above the cursor (so a push after an
        undo creates a new branch and the old one is lost).
        """
        if not isinstance(pose, CanonicalPose):
            raise TypeError(f"pose must be a CanonicalPose, got {type(pose).__name__}")
        # Drop any redo branch.
        if self._cursor < len(self._stack) - 1:
            del self._stack[self._cursor + 1 :]
        self._stack.append(pose)
        self._cursor = len(self._stack) - 1
        # Trim the bottom if we exceeded the cap.  Bottom entries are
        # the oldest, so dropping them is safe; the cursor moves with
        # the stack.
        while len(self._stack) > self._max_depth:
            self._stack.pop(0)
            self._cursor -= 1

    def undo(self) -> CanonicalPose | None:
        """Return the snapshot one step earlier, or ``None`` at bottom."""
        if not self.can_undo:
            return None
        self._cursor -= 1
        return self._stack[self._cursor]

    def redo(self) -> CanonicalPose | None:
        """Return the snapshot one step later, or ``None`` at top."""
        if not self.can_redo:
            return None
        self._cursor += 1
        return self._stack[self._cursor]


__all__ = ["HistoryController"]
