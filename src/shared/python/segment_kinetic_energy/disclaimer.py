"""User-facing copy for segment kinetic-energy globes (#15389)."""

from __future__ import annotations

#: Shown in UI tooltips and docs; must not claim globe peaks measure energy flow.
SEGMENT_KE_PROXY_DISCLAIMER = (
    "Each globe is total segment kinetic energy (translational + rotational) at "
    "the body centre of mass, in joules. Peak ordering is a kinematic proxy, not "
    "a measurement of inter-segment energy flow or Robertson–Winter interface power. "
    "See docs/research/proximal_distal_energy_transfer (What Energy Transfer Means "
    "Operationally; Kinematic Sequence as a Proxy — And Its Failure Modes)."
)

PROXY_DOC_REFERENCE = (
    "docs/research/proximal_distal_energy_transfer/chapters/_ch02_mechanics.qmd"
)

__all__ = ["PROXY_DOC_REFERENCE", "SEGMENT_KE_PROXY_DISCLAIMER"]
