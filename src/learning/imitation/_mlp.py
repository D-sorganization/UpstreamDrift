"""Shared MLP neural network and optimization utilities for imitation learning."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_ADAM_BETA1: float = 0.9
_ADAM_BETA2: float = 0.999
_ADAM_EPS: float = 1e-8


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: list[int] | tuple[int, ...],
    init: Literal["he", "small"] = "he",
    rng: np.random.Generator | None = None,
) -> list[dict[str, NDArray[np.floating]]]:
    """Build a multi-layer perceptron with specified architecture and weight initialisation.

    Args:
        input_dim: Dimension of input features. Must be > 0.
        output_dim: Dimension of output predictions. Must be > 0.
        hidden_sizes: Sequence of hidden layer dimensions.
        init: Weight initialisation scheme. 'he' uses He normal for ReLU hidden
            layers and Xavier normal for the linear output layer. 'small' uses
            legacy 0.01 normal initialisation.
        rng: Optional NumPy random generator.

    Returns:
        List of layer dictionaries containing weight matrix 'W' and bias vector 'b'.

    Raises:
        ValueError: If dimensions are non-positive or init scheme is unknown.
    """
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if output_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {output_dim}")

    layers: list[dict[str, NDArray[np.floating]]] = []
    fan_in = input_dim

    for hidden_dim in hidden_sizes:
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if init == "he":
            std = float(np.sqrt(2.0 / fan_in))
            w = (
                rng.standard_normal((fan_in, hidden_dim)) * std
                if rng is not None
                else np.random.randn(fan_in, hidden_dim) * std
            )
        elif init == "small":
            w = (
                rng.standard_normal((fan_in, hidden_dim)) * 0.01
                if rng is not None
                else np.random.randn(fan_in, hidden_dim) * 0.01
            )
        else:
            raise ValueError(f"Unknown initialization method: {init}")

        layers.append({"W": w.astype(float), "b": np.zeros(hidden_dim, dtype=float)})
        fan_in = hidden_dim

    # Output layer (linear)
    if init == "he":
        std = float(np.sqrt(2.0 / (fan_in + output_dim)))
        w_out = (
            rng.standard_normal((fan_in, output_dim)) * std
            if rng is not None
            else np.random.randn(fan_in, output_dim) * std
        )
    elif init == "small":
        w_out = (
            rng.standard_normal((fan_in, output_dim)) * 0.01
            if rng is not None
            else np.random.randn(fan_in, output_dim) * 0.01
        )
    else:
        raise ValueError(f"Unknown initialization method: {init}")

    layers.append({"W": w_out.astype(float), "b": np.zeros(output_dim, dtype=float)})
    return layers


def _mlp_forward(
    layers: list[dict[str, NDArray[np.floating]]],
    x: NDArray[np.floating],
) -> tuple[NDArray[np.floating], list[NDArray[np.floating]]]:
    """Forward pass through a ReLU MLP, retaining per-layer inputs.

    Args:
        layers: Layer dicts with ``W`` and ``b``.
        x: Batched input, shape ``(N, in_dim)`` or ``(in_dim,)``.

    Returns:
        ``(output, activations)`` where ``activations[i]`` is the input fed to
        ``layers[i]``. Postcondition: ``len(activations) == len(layers)``.

    Raises:
        ValueError: If input is None.
    """
    if x is None:
        raise ValueError("x must be provided")

    activations: list[NDArray[np.floating]] = []
    current = x
    for i, layer in enumerate(layers):
        activations.append(current)
        current = current @ layer["W"] + layer["b"]
        if i < len(layers) - 1:
            current = np.maximum(0.0, current)
    return current, activations


def _mlp_backward(
    layers: list[dict[str, NDArray[np.floating]]],
    activations: list[NDArray[np.floating]],
    grad_output: NDArray[np.floating],
) -> tuple[list[dict[str, NDArray[np.floating]]], NDArray[np.floating]]:
    """Backpropagate ``grad_output`` through a ReLU MLP.

    Args:
        layers: Layer dicts with ``W`` and ``b``.
        activations: Per-layer inputs from :func:`_mlp_forward`.
        grad_output: Gradient of the loss w.r.t. the network output.

    Returns:
        ``(grads, grad_input)`` where ``grads[i]`` holds ``dW``/``db`` for
        ``layers[i]`` and ``grad_input`` is the gradient w.r.t. the network input.

    Raises:
        ValueError: If lengths of activations and layers do not match.
    """
    if len(activations) != len(layers):
        raise ValueError(
            f"activations length ({len(activations)}) must match layers length ({len(layers)})"
        )

    grads: list[dict[str, NDArray[np.floating]]] = [
        {"W": np.zeros(0), "b": np.zeros(0)} for _ in layers
    ]
    grad = grad_output
    for i in range(len(layers) - 1, -1, -1):
        a = activations[i]
        grads[i] = {"W": a.T @ grad, "b": grad.sum(axis=0)}
        grad = grad @ layers[i]["W"].T
        if i > 0:
            # ReLU derivative: activations[i] is the post-ReLU output of i-1.
            grad = grad * (a > 0)
    return grads, grad


def _init_adam_state(
    layers: list[dict[str, NDArray[np.floating]]],
) -> list[dict[str, NDArray[np.floating]]]:
    """Create zeroed Adam moment buffers matching ``layers``."""
    return [
        {
            "m_W": np.zeros_like(layer["W"]),
            "v_W": np.zeros_like(layer["W"]),
            "m_b": np.zeros_like(layer["b"]),
            "v_b": np.zeros_like(layer["b"]),
        }
        for layer in layers
    ]


def _adam_step(
    layers: list[dict[str, NDArray[np.floating]]],
    grads: list[dict[str, NDArray[np.floating]]],
    state: list[dict[str, NDArray[np.floating]]],
    lr: float,
    step: int,
    weight_decay: float = 0.0,
) -> None:
    """Apply one in-place Adam update to ``layers``.

    Args:
        layers: Target layer dicts with 'W' and 'b'.
        grads: Gradients for each layer.
        state: Adam moment buffers for each layer.
        lr: Learning rate.
        step: 1-based global optimizer step.
        weight_decay: L2 regularization coefficient.

    Raises:
        ValueError: If step is non-positive.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    bias1 = 1.0 - _ADAM_BETA1**step
    bias2 = 1.0 - _ADAM_BETA2**step
    for layer, grad, moments in zip(layers, grads, state, strict=True):
        for key in ("W", "b"):
            g = grad[key] + weight_decay * layer[key]
            moments[f"m_{key}"] = (
                _ADAM_BETA1 * moments[f"m_{key}"] + (1.0 - _ADAM_BETA1) * g
            )
            moments[f"v_{key}"] = _ADAM_BETA2 * moments[f"v_{key}"] + (
                1.0 - _ADAM_BETA2
            ) * (g * g)
            m_hat = moments[f"m_{key}"] / bias1
            v_hat = moments[f"v_{key}"] / bias2
            layer[key] -= lr * m_hat / (np.sqrt(v_hat) + _ADAM_EPS)
