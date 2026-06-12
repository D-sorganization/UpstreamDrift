"""Qt-free helpers for plot series labels."""

from __future__ import annotations


def joint_name(joint_names: list[str], idx: int) -> str:
    """Return a plain joint label for an index."""
    if 0 <= idx < len(joint_names):
        return joint_names[idx]
    return f"Joint {idx}"


def aligned_joint_label(joint_names: list[str], idx: int, data_dim: int) -> str:
    """Return a joint label aligned to data dimensions where ``nq != nv``."""
    if len(joint_names) == 0:
        return f"DoF {idx}"
    if data_dim == len(joint_names):
        return joint_names[idx] if idx < len(joint_names) else f"DoF {idx}"

    offset = max(0, data_dim - len(joint_names))
    name_idx = idx - offset
    if 0 <= name_idx < len(joint_names):
        return joint_names[name_idx]
    return f"DoF {idx}"
