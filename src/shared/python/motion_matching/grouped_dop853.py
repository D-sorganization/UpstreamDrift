"""Opt-in DOP853 control using the worst error norm among declared blocks.

This deliberately reuses SciPy's protected DOP853._estimate_error_norm hook,
including its fifth/third-order blend, rather than copying RK coefficients.
Qualified locally with SciPy 1.16.0; requalify after SciPy changes and in each
native runtime. Local adaptive tolerances are not global trajectory guarantees.
Initial-step selection and dense output remain SciPy's unchanged algorithms.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import DOP853


def grouped_dop853(block_sizes: tuple[int, ...], state_size: int) -> type[DOP853]:
    """Validate an exact partition before creating a solver or calling dynamics."""
    if (
        not isinstance(block_sizes, tuple)
        or not block_sizes
        or any(
            isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in block_sizes
        )
        or sum(block_sizes) != state_size
    ):
        raise ValueError(
            "error_block_sizes must be positive integers partitioning the state"
        )
    boundaries = np.cumsum((0,) + block_sizes)
    blocks = tuple(
        slice(int(start), int(end))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    )

    class GroupedDOP853(DOP853):
        def _estimate_error_norm(
            self, stages: NDArray[np.float64], step: float, scale: NDArray[np.float64]
        ) -> float:
            existing_norm = super()._estimate_error_norm
            return max(
                float(existing_norm(stages[:, block], step, scale[block]))
                for block in blocks
            )

    return GroupedDOP853
