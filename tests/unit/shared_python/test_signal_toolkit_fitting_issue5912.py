"""Regression tests for issue #5912 signal_toolkit fitting fallbacks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from src.shared.python.signal_toolkit import fitting as fitting_module
from src.shared.python.signal_toolkit.core import Signal
from src.shared.python.signal_toolkit.fitting import (
    CustomFunctionFitter,
    ExponentialFitter,
    SinusoidFitter,
)


@pytest.fixture
def sample_signal() -> Signal:
    time = np.linspace(0.0, 1.0, 32)
    values = np.sin(2 * np.pi * time)
    return Signal(time=time, values=values, name="sample")


@pytest.mark.parametrize(
    ("runner", "initial_guess"),
    [
        (
            lambda signal, guess: SinusoidFitter().fit(
                signal,
                initial_guess=guess,
            ),
            np.asarray([1.0, 1.0, 0.0, 0.0]),
        ),
        (
            lambda signal, guess: ExponentialFitter().fit_decay(
                signal,
                initial_guess=guess,
            ),
            np.asarray([1.0, 0.5, 0.0]),
        ),
        (
            lambda signal, guess: ExponentialFitter().fit_growth(
                signal,
                initial_guess=guess,
            ),
            np.asarray([1.0, 0.5, 0.0]),
        ),
        (
            lambda signal, guess: CustomFunctionFitter(
                lambda t, a, b: a * t + b,
                ["a", "b"],
            ).fit(
                signal,
                initial_guess=guess,
            ),
            np.asarray([1.0, 0.0]),
        ),
    ],
)
def test_fit_failure_fallback_uses_asarray_not_array(
    monkeypatch: pytest.MonkeyPatch,
    sample_signal: Signal,
    runner: Callable[[Signal, np.ndarray], object],
    initial_guess: np.ndarray,
) -> None:
    """Issue #5912: fallback paths should reuse ndarray guesses without copying."""
    original_array = fitting_module.np.array
    original_asarray = fitting_module.np.asarray
    asarray_calls: list[object] = []

    def fail_curve_fit(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced fit failure")

    def guarded_array(*args: object, **kwargs: object) -> np.ndarray:
        if args and args[0] is initial_guess:
            raise AssertionError("fallback must not call np.array(initial_guess)")
        return original_array(*args, **kwargs)

    def recording_asarray(*args: object, **kwargs: object) -> np.ndarray:
        if args:
            asarray_calls.append(args[0])
        return original_asarray(*args, **kwargs)

    monkeypatch.setattr(fitting_module.optimize, "curve_fit", fail_curve_fit)
    monkeypatch.setattr(fitting_module.np, "array", guarded_array)
    monkeypatch.setattr(fitting_module.np, "asarray", recording_asarray)

    result = runner(sample_signal, initial_guess)

    assert result.success is False
    assert initial_guess in asarray_calls
