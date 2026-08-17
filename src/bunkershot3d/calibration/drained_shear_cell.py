"""Drained shear cell (simplified Jenike-style) calibration experiment.

There is **no** drained-shear-cell simulation implemented for any backend.
Before issue #7999 this module presented a hand-written linear formula
(``phi_peak = 20 + 30 * friction``) as "the shear cell simulation"; it ignored
``self.backend`` entirely and had no ``use_mock`` flag, so a caller asking for
a real calibration silently received fabricated data.

The stand-in is now reachable only through an explicit ``use_mock=True``
(or ``backend="mock"``), exactly like :mod:`angle_of_repose`. Every other
backend raises :class:`~bunkershot3d.exceptions.BackendNotImplementedError`.
"""

from __future__ import annotations

from ..exceptions import BackendNotImplementedError

#: Slope and intercept of the analytical stand-in, in degrees.
_MOCK_PHI_INTERCEPT = 20.0
_MOCK_PHI_SLOPE = 30.0
_MOCK_PEAK_TO_RESIDUAL = 5.0


def _mock_shear_response(friction: float) -> tuple[float, float]:
    """Analytical stand-in used for fast unit tests only.

    This is **not** a simulation: it is a straight line chosen so that the
    target peak angle inverts to the canonical friction of 0.5. It carries no
    information about any backend or about real sand.

    Args:
        friction: Contact friction coefficient.

    Returns:
        ``(phi_peak, phi_res)`` in degrees.
    """
    phi_peak = _MOCK_PHI_INTERCEPT + friction * _MOCK_PHI_SLOPE
    return phi_peak, phi_peak - _MOCK_PEAK_TO_RESIDUAL


class DrainedShearCellExperiment:
    """Applies normal load, shears at constant rate, extracts friction angles.

    Attributes:
        backend: Requested backend name.
        target_phi_peak: Target peak friction angle in degrees.
        target_phi_res: Target residual friction angle in degrees.
        calibrated_parameters: Parameters the response actually depends on.
    """

    #: Only friction enters the response; see #7999.
    calibrated_parameters: tuple[str, ...] = ("friction_coefficient",)

    def __init__(
        self, backend: str = "chrono", *, use_mock: bool | None = None
    ) -> None:
        """Initialise the experiment.

        Args:
            backend: Backend name. Only ``"mock"`` is implemented.
            use_mock: Force the analytical stand-in. Defaults to
                ``backend == "mock"``.

        Raises:
            BackendNotImplementedError: If a non-mock backend is requested.
        """
        self.backend = backend
        self._use_mock = backend == "mock" if use_mock is None else use_mock

        if not self._use_mock:
            raise BackendNotImplementedError(
                backend,
                feature=(
                    "DrainedShearCellExperiment has no DEM implementation for any "
                    "backend. Pass use_mock=True to use the analytical stand-in, "
                    "and do not treat its output as measured data (issue #7999)"
                ),
            )

        self.target_phi_peak = 35.0
        self.target_phi_res = 30.0

    def run_simulation(self, params: dict) -> tuple[float, float]:
        """Return the peak and residual friction angles.

        Args:
            params: Must contain ``friction_coefficient``.

        Returns:
            ``(phi_peak, phi_res)`` in degrees.

        Raises:
            BackendNotImplementedError: If a real backend run was requested.
        """
        if not self._use_mock:
            raise BackendNotImplementedError(
                self.backend,
                feature="DrainedShearCellExperiment physical simulation",
            )
        return _mock_shear_response(float(params.get("friction_coefficient", 0.5)))

    def calibrate(self) -> dict:
        """Grid-search friction to minimise residual vs the target angles.

        Returns:
            The best ``friction_coefficient`` found. ``restitution_coefficient``
            is deliberately absent: no shear-cell response depends on it, so
            reporting one would be a fabricated measurement (issue #7999).
        """
        best_params = {"friction_coefficient": 0.5}
        best_residual = float("inf")
        for step in range(9):
            friction = 0.1 + step * 0.1
            phi_peak, phi_res = self.run_simulation({"friction_coefficient": friction})
            residual = (phi_peak - self.target_phi_peak) ** 2 + (
                phi_res - self.target_phi_res
            ) ** 2
            if residual < best_residual:
                best_residual = residual
                best_params = {"friction_coefficient": float(friction)}
        return best_params
