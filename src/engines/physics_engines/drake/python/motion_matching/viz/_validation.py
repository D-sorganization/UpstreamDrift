"""Validation helpers for Drake motion-matching visualization."""

from __future__ import annotations

from typing import Any


def validate_shared_timegrid(fit: Any, target: Any) -> None:
    """Validate that fit and target share identical sample counts on the timegrid.

    Raises:
        ValueError: if the ``fit`` and ``target`` timegrids disagree.
    """
    if fit.time.shape[0] != target.time.shape[0]:
        msg = (
            "fit and target must share the canonical timegrid; got "
            f"{fit.time.shape[0]} vs {target.time.shape[0]} samples"
        )
        raise ValueError(msg)
