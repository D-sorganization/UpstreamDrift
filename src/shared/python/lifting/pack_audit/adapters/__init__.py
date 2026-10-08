"""Per-engine adapters; each imports its engine lazily."""

from __future__ import annotations

from ..model import Anthropometry, EngineAdapter
from ..packs import PackLocation


def create_adapter(
    pack: PackLocation, lift: str, anthro: Anthropometry
) -> EngineAdapter:
    """Build the adapter for ``pack.engine``.

    Raises:
        ValueError: If the pack's engine has no adapter.
    """
    if pack.engine == "mujoco":
        from .mujoco_adapter import MujocoAdapter

        return MujocoAdapter(pack, lift, anthro)
    if pack.engine == "opensim":
        from .opensim_adapter import OpensimAdapter

        return OpensimAdapter(pack, lift, anthro)
    if pack.engine == "drake":
        from .drake_adapter import DrakeAdapter

        return DrakeAdapter(pack, lift, anthro)
    if pack.engine == "pinocchio":
        from .pinocchio_adapter import PinocchioAdapter

        return PinocchioAdapter(pack, lift, anthro)
    raise ValueError(f"no adapter for engine {pack.engine!r}")
