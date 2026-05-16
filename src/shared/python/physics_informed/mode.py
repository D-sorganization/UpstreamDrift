"""Factory for creating physics-informed model instances by mode.

Phase 3 of the PINNs epic (#5419). Provides:

- :class:`PhysicsMode`: Enum for the three supported physics modes.
- :func:`create_model`: Factory that returns a :class:`~.rigid_core.RigidCore`,
  :class:`~.mlp_residual.MlpResidual`, or :class:`~.hybrid_model.HybridPINN`
  depending on the requested mode.

Optional dependencies:
- Pinocchio: required for :attr:`PhysicsMode.PURE_RIGID` and
  :attr:`PhysicsMode.PINN_HYBRID`.
- JAX + Equinox: required for :attr:`PhysicsMode.PURE_AI` and
  :attr:`PhysicsMode.PINN_HYBRID`.

If a required dependency is not installed, :func:`create_model` raises
:class:`ImportError` at call time rather than at module-import time.

Part of epic #5419. Closes #5499.
"""

from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class PhysicsMode(Enum):
    """Mode selector for the physics-informed model factory.

    Attributes:
        PURE_RIGID:  Pinocchio inverse dynamics only; no ML.
        PURE_AI:     JAX/Equinox MLP residual only; no rigid-body model.
        PINN_HYBRID: Rigid body + JAX residual torque MLP (full PINN).
    """

    PURE_RIGID = "pure_rigid"
    PURE_AI = "pure_ai"
    PINN_HYBRID = "pinn_hybrid"


def create_model(
    mode: PhysicsMode,
    config: dict,
) -> object:
    """Factory: return a model instance for the given PhysicsMode.

    Supported modes and required ``config`` keys:

    **PURE_RIGID** -> :class:`~.rigid_core.RigidCore`
        - ``urdf_path`` (str): Path to URDF file.

    **PURE_AI** -> :class:`~.mlp_residual.MlpResidual`
        - ``input_dim``  (int): MLP input dimension.
        - ``output_dim`` (int): MLP output dimension (should equal ``nv``).
        - ``hidden_dims`` (list[int], optional): Hidden layer widths.
          Default: ``[64, 64]``.
        - ``key`` (jax.Array, optional): JAX PRNG key. Default: PRNGKey(0).

    **PINN_HYBRID** -> :class:`~.hybrid_model.HybridPINN`
        - All keys from both PURE_RIGID and PURE_AI.

    DbC preconditions:
    - ``mode`` must be a :class:`PhysicsMode` member.
    - Raises :class:`ValueError` for any other value.

    Args:
        mode:   Desired physics mode.
        config: Configuration dictionary; see above for required keys.

    Returns:
        An instance of :class:`~.rigid_core.RigidCore`,
        :class:`~.mlp_residual.MlpResidual`, or
        :class:`~.hybrid_model.HybridPINN`.

    Raises:
        ValueError:   If ``mode`` is not a recognised :class:`PhysicsMode`.
        ImportError:  If a required optional dependency is not installed.
    """
    if not isinstance(mode, PhysicsMode):
        raise ValueError(
            f"Unrecognized PhysicsMode: {mode!r}. "
            f"Expected one of: {[m.value for m in PhysicsMode]}"
        )

    logger.debug(
        "create_model: mode=%s config_keys=%s", mode.value, list(config.keys())
    )

    if mode == PhysicsMode.PURE_RIGID:
        return _create_rigid(config)
    if mode == PhysicsMode.PURE_AI:
        return _create_mlp(config)
    if mode == PhysicsMode.PINN_HYBRID:
        return _create_hybrid(config)

    # Defensive: unreachable if the enum is complete, but satisfies type checkers.
    raise ValueError(f"Unrecognized PhysicsMode: {mode!r}")  # pragma: no cover


# =============================================================================
# Private factory helpers
# =============================================================================


def _create_rigid(config: dict) -> object:
    """Instantiate a :class:`~.rigid_core.RigidCore`.

    Args:
        config: Must contain ``urdf_path`` key.

    Returns:
        A configured :class:`~.rigid_core.RigidCore`.

    Raises:
        ImportError: If Pinocchio is not installed.
        ValueError:  If ``urdf_path`` is missing, empty, or the file does
            not exist.
    """
    from src.shared.python.physics_informed.rigid_core import RigidCore

    urdf_path: str = config.get("urdf_path", "")
    logger.debug("create_model PURE_RIGID: urdf_path=%r", urdf_path)
    return RigidCore(urdf_path)


def _create_mlp(config: dict) -> object:
    """Instantiate an :class:`~.mlp_residual.MlpResidual`.

    Args:
        config: Must contain ``input_dim`` and ``output_dim``.  Optional keys:
            ``hidden_dims`` (default ``[64, 64]``) and ``key`` (default
            ``jax.random.PRNGKey(0)``).

    Returns:
        A configured :class:`~.mlp_residual.MlpResidual`.

    Raises:
        ImportError: If JAX or Equinox are not installed.
        ValueError:  If ``input_dim`` or ``output_dim`` are invalid.
    """
    import jax

    from src.shared.python.physics_informed.mlp_residual import MlpResidual

    input_dim: int = config["input_dim"]
    output_dim: int = config["output_dim"]
    hidden_dims: list[int] = config.get("hidden_dims", [64, 64])
    key = config.get("key", jax.random.PRNGKey(0))

    logger.debug(
        "create_model PURE_AI: input=%d output=%d hidden=%s",
        input_dim,
        output_dim,
        hidden_dims,
    )
    return MlpResidual(
        input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, key=key
    )


def _create_hybrid(config: dict) -> object:
    """Instantiate a :class:`~.hybrid_model.HybridPINN`.

    Args:
        config: Combined keys required by :func:`_create_rigid` and
            :func:`_create_mlp`.

    Returns:
        A configured :class:`~.hybrid_model.HybridPINN`.

    Raises:
        ImportError: If Pinocchio, JAX, or Equinox are not installed.
        ValueError:  If any required config key is invalid.
    """
    from src.shared.python.physics_informed.hybrid_model import HybridPINN

    rigid = _create_rigid(config)
    mlp = _create_mlp(config)

    logger.debug("create_model PINN_HYBRID: combining rigid + mlp")
    return HybridPINN(rigid, mlp)  # type: ignore[arg-type]
