"""JAX/Equinox MLP for predicting residual torques not captured by rigid-body dynamics.

This module provides :class:`MlpResidual`, a lightweight feed-forward neural
network built with `Equinox <https://github.com/patrick-kidger/equinox>`_.  It
is designed to be composed with :class:`~.rigid_core.RigidCore` inside
:class:`~.hybrid_model.HybridPINN` to form a Physics-Informed Neural Network
(PINN) hybrid.

JAX and Equinox are *optional* dependencies:

    pip install upstream-drift[physics_informed]

If they are not installed the module still imports cleanly, but instantiating
:class:`MlpResidual` will raise :class:`ImportError`.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    eqx = None  # type: ignore[assignment]
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]


class MlpResidual:
    """JAX/Equinox MLP predicting residual torque not captured by the rigid-body model.

    The network accepts a concatenation of ``(q, dq, ddq)`` as its input
    vector and returns a torque residual of shape ``(output_dim,)``.

    Parameters
    ----------
    input_dim:
        Dimensionality of the input vector.  Typically ``nq + 2 * nv`` for
        a model with configuration-space size ``nq`` and velocity-space size
        ``nv``.
    output_dim:
        Dimensionality of the output vector.  Must equal ``nv`` to match the
        rigid-body torque vector.
    hidden_dims:
        List of hidden-layer widths.  An empty list produces a single linear
        layer (no hidden layers).
    key:
        A JAX PRNG key used to initialise the network weights.

    Raises
    ------
    ImportError
        If JAX or Equinox are not installed.
    ValueError
        If ``input_dim`` or ``output_dim`` are not positive integers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        *,
        key: Any,
    ) -> None:
        """Build the MLP.

        DbC preconditions:
        - ``input_dim >= 1``
        - ``output_dim >= 1``
        - ``key`` must be a JAX PRNG key.

        Args:
            input_dim:   Number of input features.
            output_dim:  Number of output features (must equal ``nv``).
            hidden_dims: Widths of hidden layers.  Empty list → linear model.
            key:         JAX PRNG key for weight initialisation.

        Raises:
            ImportError: If JAX / Equinox are not installed.
            ValueError:  If dimensions are invalid.
        """
        if not HAS_JAX:
            raise ImportError(
                "jax and equinox are required for MlpResidual; install with: "
                "pip install upstream-drift[physics_informed]"
            )

        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1; got {input_dim}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1; got {output_dim}")

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.hidden_dims: list[int] = list(hidden_dims)

        self._net = self._build_network(input_dim, output_dim, hidden_dims, key)

        logger.debug(
            "MlpResidual built: input=%d hidden=%s output=%d",
            input_dim,
            hidden_dims,
            output_dim,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, x: Any) -> Any:
        """Run a forward pass through the MLP.

        Args:
            x: Input array of shape ``(input_dim,)``.

        Returns:
            Output array of shape ``(output_dim,)``.

        Raises:
            ValueError: If ``x`` has the wrong shape.
        """
        x_arr = jnp.asarray(x, dtype=jnp.float32)
        if x_arr.shape != (self.input_dim,):
            raise ValueError(
                f"Expected input shape ({self.input_dim},); got {x_arr.shape}"
            )
        return self._net(x_arr)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_network(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        key: Any,
    ) -> Any:
        """Construct an ``eqx.nn.Sequential`` MLP.

        Args:
            input_dim:   Input feature count.
            output_dim:  Output feature count.
            hidden_dims: Hidden-layer widths.
            key:         JAX PRNG key.

        Returns:
            An Equinox ``Sequential`` module.
        """
        layer_dims = [input_dim, *hidden_dims, output_dim]
        layers: list[Any] = []

        for idx, (in_d, out_d) in enumerate(
            zip(layer_dims[:-1], layer_dims[1:], strict=True)
        ):
            key, subkey = jax.random.split(key)
            layers.append(eqx.nn.Linear(in_d, out_d, key=subkey))
            # Apply ReLU after every layer except the final output layer
            is_last = idx == len(layer_dims) - 2
            if not is_last:
                layers.append(jax.nn.relu)

        return eqx.nn.Sequential(layers)
