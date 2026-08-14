"""The F0 solver: Dynamic Resistive Force Theory (issue #8611, ADR-0032).

This is the **default solver of the whole tool**.  Every other tier
exists to calibrate or cross-check it.

The element stress is::

    t = alpha(beta, gamma, psi) * H(-z_tilde) * |z_tilde|
        - n_hat * lambda * rho * v_n^2
    z_tilde = z + delta_h

evaluated only on elements that are **both** leading-edge
(``v_hat . n_hat >= 0``) and below the effective free surface, then
integrated over the discretised surface to a resultant wrench.

Why *dynamic*, not quasi-static
-------------------------------

With ``alpha_z ~ 2.02 N/cm^3``, a 40 mm divot, ``lambda = 1.1`` and
``rho = 1600 kg/m^3`` the two terms cross at **6.8 m/s**.  Greenside
delivery is 20-27 m/s, so the inertial term carries roughly 90% of the
load: it is the leading term, not a correction, and any quasi-static
solver -- baseline RFT, Bekker, Wong-Reece -- is wrong here by an order
of magnitude.  ``lambda`` is consequently the primary calibration target,
ahead of ``alpha``.

Both DRFT corrections are implemented.  See
:mod:`bunkershot3d.solvers.structural` for why the second one,
``delta_h``, is not optional and why its wedge form is unknown.

Three deliberate asymmetries in the implementation
--------------------------------------------------

* **The fit is clamped; the momentum flux is not.**  The 3D-RFT
  polynomial is fitted for ``beta, psi`` in ``[-pi/2, pi/2]``, which
  covers only surface normals pointing at or below the horizontal.  A
  lofted club face points *up*, and is still a leading edge.  Its
  orientation is clamped to the vertical-wall limit for the polynomial
  and the clamped area fraction is reported, while the inertial term --
  which is a momentum flux, not a fit -- uses the true normal.
* **Refusal is a value, not an exception, until a policy is applied.**
  :meth:`DRFTSolver.envelope` always returns a verdict.  ``solve``
  raises only because the default policy is strict.
* **Nothing here uses ``assert``.**  ``python -O`` strips assertions and
  ``DBC_LEVEL=off`` disables contracts; the envelope must survive both.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts import ensure, require

from .coefficients import MaterialResponse, generic_alpha
from .elements import SurfaceElements
from .envelope import (
    GRAVITY_M_S2,
    Caveat,
    RefusalPolicy,
    ValidityVerdict,
    evaluate_envelope,
)
from .exceptions import SolverInputError
from .protocol import FidelityTier, IntrusionState, SolverResult, Wrench
from .structural import (
    DepressionInputs,
    StructuralCorrection,
    default_structural_correction,
)

__all__ = ["DEFAULT_FEATURE_SCALES_M", "DRFTSolver", "ElementResponse"]

_SPEED_FLOOR_M_S = 1e-12
_HORIZONTAL_FLOOR = 1e-12
_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_HORIZONTAL_MASK = np.array([1.0, 1.0, 0.0], dtype=np.float64)
_FALLBACK_RADIAL = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_NO_ELEMENTS = np.zeros(0, dtype=np.int64)

DEFAULT_FEATURE_SCALES_M: Mapping[str, float] = {
    "clubhead": 0.100,
    "sole width": 0.030,
    "leading edge": 0.005,
}
"""The three scales the research addendum judges the envelope on.

Judging on the clubhead alone would report ``I = 0.126``; the 5 mm
leading edge is simultaneously at ``I = 11.3``.  Reporting only the
flattering one is how a solver launders an extrapolation."""


@dataclass(frozen=True)
class ElementResponse:
    """Per-element intermediates, kept for tests and diagnostics.

    All arrays are over the *active* elements only, in the order they
    appear in the body's element arrays.

    Attributes:
        index: Indices of the active elements in the source arrays.
        beta_rad: Surface tilt used by the polynomial (after clamping).
        gamma_rad: Attack angle of the local velocity.
        psi_rad: Twist of the surface normal.
        depth_m: Positive depth below the undisturbed free surface.
        effective_depth_m: ``|z_tilde|`` after the structural correction.
        normal_speed_m_s: ``max(v . n_hat, 0)``.
        depth_traction_pa: Depth-linear traction vectors, ``(k, 3)``.
        inertial_traction_pa: Dynamic traction vectors, ``(k, 3)``.
        was_clamped: Elements whose normal pointed above the horizontal.
    """

    index: NDArray[np.int64]
    beta_rad: NDArray[np.float64]
    gamma_rad: NDArray[np.float64]
    psi_rad: NDArray[np.float64]
    depth_m: NDArray[np.float64]
    effective_depth_m: NDArray[np.float64]
    normal_speed_m_s: NDArray[np.float64]
    depth_traction_pa: NDArray[np.float64]
    inertial_traction_pa: NDArray[np.float64]
    was_clamped: NDArray[np.bool_]


def _cross_sum(
    lever: NDArray[np.float64], force: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Sum of ``lever x force`` over elements, written out.

    ``np.cross`` allocates through ``moveaxis`` and shows up in a profile
    of the shot loop; the explicit form is the same arithmetic without
    the dispatch.
    """
    return np.array(
        [
            float((lever[:, 1] * force[:, 2] - lever[:, 2] * force[:, 1]).sum()),
            float((lever[:, 2] * force[:, 0] - lever[:, 0] * force[:, 2]).sum()),
            float((lever[:, 0] * force[:, 1] - lever[:, 1] * force[:, 0]).sum()),
        ],
        dtype=np.float64,
    )


def _local_frame(
    velocity_direction: NDArray[np.float64], normals: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build ``(r_hat, theta_hat)`` per element.

    ``r_hat`` is the horizontal component of the velocity direction.  It
    is degenerate for purely vertical motion, where the response is
    independent of the choice; the fallbacks are ordered so that the two
    likely degenerate cases stay equivariant under rotation about the
    vertical.

    Args:
        velocity_direction: ``(k, 3)`` unit velocity directions.
        normals: ``(k, 3)`` unit outward normals.

    Returns:
        ``(r_hat, theta_hat)``, each ``(k, 3)``.
    """
    horizontal = velocity_direction * _HORIZONTAL_MASK
    magnitude = np.hypot(horizontal[:, 0], horizontal[:, 1])[:, None]

    if not bool((magnitude > _HORIZONTAL_FLOOR).all()):
        # Fall back to the horizontal part of the inward normal, which
        # rotates with the body, before falling back to a fixed axis.
        inward = -normals * _HORIZONTAL_MASK
        inward_magnitude = np.hypot(inward[:, 0], inward[:, 1])[:, None]
        usable = magnitude > _HORIZONTAL_FLOOR
        horizontal = np.where(usable, horizontal, inward)
        magnitude = np.where(usable, magnitude, inward_magnitude)

        usable = magnitude > _HORIZONTAL_FLOOR
        horizontal = np.where(usable, horizontal, _FALLBACK_RADIAL)
        magnitude = np.where(usable, magnitude, 1.0)

    radial = horizontal / magnitude
    # theta_hat = z_hat x r_hat, written out: the cross product of a unit
    # vertical with a horizontal vector is a quarter turn in the plane.
    tangential = np.empty_like(radial)
    tangential[:, 0] = -radial[:, 1]
    tangential[:, 1] = radial[:, 0]
    tangential[:, 2] = 0.0
    return radial, tangential


def _orientation_angles(
    normals: NDArray[np.float64],
    radial: NDArray[np.float64],
    tangential: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]
]:
    """Extract ``(beta, psi)`` and the clamped normal from element normals.

    The fitted domain covers normals with ``n_z <= 0``.  Normals above
    the horizontal are projected onto it -- the vertical-wall limit --
    and flagged.

    Returns:
        ``(beta_rad, psi_rad, clamped_normals, was_clamped)``.
    """
    was_clamped = normals[:, 2] > 0.0
    if bool(was_clamped.any()):
        clamped = normals.copy()
        clamped[was_clamped, 2] = 0.0
        horizontal_magnitude = np.hypot(clamped[:, 0], clamped[:, 1])
        rescale = np.where(
            was_clamped & (horizontal_magnitude > _HORIZONTAL_FLOOR),
            1.0 / np.where(horizontal_magnitude > 0.0, horizontal_magnitude, 1.0),
            1.0,
        )
        clamped = clamped * rescale[:, None]
    else:
        clamped = normals

    n_radial = np.einsum("ij,ij->i", clamped, radial)
    n_tangential = np.einsum("ij,ij->i", clamped, tangential)
    n_vertical = clamped[:, 2]

    # sign is chosen so that cos(psi) >= 0, which is what confines psi to
    # the fitted [-pi/2, pi/2] band.
    sign = np.where(n_radial >= 0.0, 1.0, -1.0)
    in_plane = np.hypot(n_radial, n_tangential)
    beta_rad = np.arctan2(sign * in_plane, -n_vertical)
    psi_rad = np.arctan2(sign * n_tangential, sign * n_radial)
    return beta_rad, psi_rad, clamped, was_clamped


@dataclass(frozen=True)
class DRFTSolver:
    """Dynamic 3D Resistive Force Theory: the F0 tier.

    Attributes:
        material: Sand response constants and their provenance.
        structural_correction: The ``delta_h`` model. Defaults to the
            documented, uncalibrated
            :class:`~bunkershot3d.solvers.structural.CrossoverSaturatingDepression`.
        dynamic_terms_active: Whether the DRFT inertial term is applied.
            Switching it off yields quasi-static RFT, which the envelope
            then refuses above ``Fr ~ 1``.
        refusal_policy: What happens on a refusal. Strict by default.
        feature_scales_m: The geometric scales the envelope is judged on.
        gravity_m_s2: Gravitational acceleration.
    """

    material: MaterialResponse
    structural_correction: StructuralCorrection = field(
        default_factory=default_structural_correction
    )
    dynamic_terms_active: bool = True
    refusal_policy: RefusalPolicy = RefusalPolicy.STRICT
    feature_scales_m: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FEATURE_SCALES_M)
    )
    gravity_m_s2: float = GRAVITY_M_S2

    def __post_init__(self) -> None:
        if not isinstance(self.material, MaterialResponse):
            raise SolverInputError(
                f"material must be a MaterialResponse, got "
                f"{type(self.material).__name__}"
            )
        if not isinstance(self.structural_correction, StructuralCorrection):
            raise SolverInputError(
                "structural_correction must implement the StructuralCorrection "
                f"protocol, got {type(self.structural_correction).__name__}"
            )
        if not self.feature_scales_m:
            raise SolverInputError("at least one feature scale is required")
        if not math.isfinite(self.gravity_m_s2) or self.gravity_m_s2 <= 0.0:
            raise SolverInputError(
                f"gravity must be positive, got {self.gravity_m_s2!r}"
            )
        object.__setattr__(self, "feature_scales_m", dict(self.feature_scales_m))

    # ------------------------------------------------------------ protocol

    @property
    def fidelity_tier(self) -> FidelityTier:
        """Always :attr:`~bunkershot3d.solvers.protocol.FidelityTier.F0`."""
        return FidelityTier.F0

    def envelope(self, state: IntrusionState) -> ValidityVerdict:
        """Judge a query without integrating any forces.

        Args:
            state: The intrusion query.

        Returns:
            The verdict. Refusal is a value here; it becomes an exception
            only in :meth:`solve`, under the solver's refusal policy.
        """
        self._require_state(state)
        velocities = state.element_velocities_m_s()
        speeds = np.sqrt(np.einsum("ij,ij->i", velocities, velocities))
        depths = -state.element_depths_m()
        submerged = depths > 0.0
        return self._verdict(
            speed_m_s=float(speeds.max()) if speeds.size else 0.0,
            element_size_m=state.elements.characteristic_length_m,
            submerged_depth_m=float(depths[submerged].max())
            if submerged.any()
            else 0.0,
            clamped_area_fraction=0.0,
        )

    def solve(self, state: IntrusionState) -> SolverResult:
        """Integrate the DRFT stress over the submerged leading edge.

        Args:
            state: The intrusion query.

        Returns:
            The resultant wrench about ``state.reference_point_m``, the
            fidelity tier, and the validity verdict -- always all three.

        Raises:
            OutOfEnvelopeError: If the verdict is a refusal and the
                solver's policy is strict.
            SolverInputError: If the query is malformed.
        """
        self._require_state(state)
        response, verdict = self._evaluate(state)
        verdict.require_usable(self.refusal_policy)

        elements = state.elements
        if response.index.size == 0:
            return SolverResult(
                wrench=Wrench.zero(state.reference_point_m),
                fidelity_tier=self.fidelity_tier,
                verdict=verdict,
                depth_force_n=np.zeros(3),
                inertial_force_n=np.zeros(3),
                n_active_elements=0,
                active_area_m2=0.0,
                max_depth_m=0.0,
            )

        active_areas = elements.areas_m2[response.index]
        areas = active_areas[:, None]
        depth_force = (response.depth_traction_pa * areas).sum(axis=0)
        inertial_force = (response.inertial_traction_pa * areas).sum(axis=0)
        element_force = (
            response.depth_traction_pa + response.inertial_traction_pa
        ) * areas
        lever = elements.centroids_m[response.index] - state.reference_point_m
        torque = _cross_sum(lever, element_force)

        wrench = Wrench(depth_force + inertial_force, torque, state.reference_point_m)
        ensure(
            bool(np.isfinite(wrench.force_n).all()),
            "DRFT produced a non-finite resultant force",
            value=wrench.force_n,
        )
        return SolverResult(
            wrench=wrench,
            fidelity_tier=self.fidelity_tier,
            verdict=verdict,
            depth_force_n=depth_force,
            inertial_force_n=inertial_force,
            n_active_elements=int(response.index.size),
            active_area_m2=float(active_areas.sum()),
            max_depth_m=float(response.depth_m.max()),
        )

    # ------------------------------------------------------------ internals

    def _require_state(self, state: IntrusionState) -> None:
        """Precondition: ``state`` is a usable intrusion query."""
        if not isinstance(state, IntrusionState):
            raise SolverInputError(
                f"expected an IntrusionState, got {type(state).__name__}"
            )
        require(
            isinstance(state.elements, SurfaceElements),
            "intrusion state must carry a SurfaceElements structure of arrays",
            value=type(state.elements).__name__,
        )

    def _verdict(
        self,
        *,
        speed_m_s: float,
        element_size_m: float,
        submerged_depth_m: float,
        clamped_area_fraction: float,
    ) -> ValidityVerdict:
        """Assemble the validity verdict for one query."""
        extra: list[Caveat] = []
        return evaluate_envelope(
            speed_m_s=speed_m_s,
            feature_lengths_m=self.feature_scales_m,
            grain_diameter_m=self.material.grain_diameter_m,
            element_size_m=max(element_size_m, self.material.grain_diameter_m),
            dynamic_terms_active=self.dynamic_terms_active,
            submerged_depth_m=submerged_depth_m,
            clamped_area_fraction=clamped_area_fraction,
            structural_correction_calibrated=(
                self.structural_correction.is_calibrated_for_wedge
            ),
            extra_caveats=extra,
            gravity_m_s2=self.gravity_m_s2,
        )

    def _evaluate(
        self, state: IntrusionState
    ) -> tuple[ElementResponse, ValidityVerdict]:
        """Compute every per-element quantity and the verdict, vectorised."""
        elements = state.elements
        depths = -state.element_depths_m()
        submerged = (depths > 0.0) & (elements.areas_m2 > 0.0)
        submerged_depth = max(float(depths.max()) if depths.size else 0.0, 0.0)

        # The leading-edge test only needs the *sign* of v . n, so it is
        # taken on the unnormalised velocity and the direction is formed
        # for the surviving elements alone. On a 600-face head that is
        # one fewer full-body division per timestep.
        if state.has_uniform_velocity:
            max_speed = state.speed_m_s
            moving = max_speed > _SPEED_FLOOR_M_S
            leading = (elements.normals @ state.velocity_m_s) >= 0.0
            index = np.flatnonzero(submerged & leading) if moving else _NO_ELEMENTS
            velocities = None
            unit = state.velocity_m_s / max_speed if moving else None
        else:
            velocities = state.element_velocities_m_s()
            speeds = np.sqrt(np.einsum("ij,ij->i", velocities, velocities))
            max_speed = float(speeds.max()) if speeds.size else 0.0
            leading = np.einsum("ij,ij->i", velocities, elements.normals) >= 0.0
            index = np.flatnonzero(submerged & leading & (speeds > _SPEED_FLOOR_M_S))
            unit = None

        if index.size == 0:
            empty = np.zeros(0, dtype=np.float64)
            response = ElementResponse(
                index=index,
                beta_rad=empty,
                gamma_rad=empty,
                psi_rad=empty,
                depth_m=empty,
                effective_depth_m=empty,
                normal_speed_m_s=empty,
                depth_traction_pa=np.zeros((0, 3)),
                inertial_traction_pa=np.zeros((0, 3)),
                was_clamped=np.zeros(0, dtype=bool),
            )
            verdict = self._verdict(
                speed_m_s=max_speed,
                element_size_m=elements.characteristic_length_m,
                submerged_depth_m=submerged_depth,
                clamped_area_fraction=0.0,
            )
            return response, verdict

        normals = elements.normals[index]
        active_depths = depths[index]
        active_areas = elements.areas_m2[index]
        if unit is not None:
            active_velocities = np.broadcast_to(state.velocity_m_s, normals.shape)
            direction = np.broadcast_to(unit, normals.shape)
        else:
            active_velocities = velocities[index]  # type: ignore[index]
            direction = active_velocities / speeds[index][:, None]

        radial, tangential = _local_frame(direction, normals)
        gamma_rad = np.arcsin(np.clip(-direction[:, 2], -1.0, 1.0))
        beta_rad, psi_rad, fitted_normals, was_clamped = _orientation_angles(
            normals, radial, tangential
        )

        alpha_r, alpha_theta, alpha_z = generic_alpha(beta_rad, gamma_rad, psi_rad)
        alpha_vector = (
            alpha_r[:, None] * radial
            + alpha_theta[:, None] * tangential
            + alpha_z[:, None] * _UP
        )
        capped = self._apply_surface_friction_cutoff(alpha_vector, fitted_normals)

        normal_speed = np.maximum(
            np.einsum("ij,ij->i", active_velocities, normals), 0.0
        )
        depth_stress_scale = self.material.normal_stress_scale_pa_per_m * np.sqrt(
            np.einsum("ij,ij->i", capped, capped)
        )
        depression = self.structural_correction.depression_m(
            DepressionInputs(
                depth_m=active_depths,
                normal_speed_m_s=normal_speed,
                depth_stress_scale_pa_per_m=depth_stress_scale,
                inertial_stress_scale_pa_s2_per_m2=(
                    self.material.inertial_stress_scale_pa_s2_per_m2
                ),
                gravity_m_s2=self.gravity_m_s2,
            )
        )
        if np.any(depression < 0.0):
            raise SolverInputError(
                f"structural correction '{self.structural_correction.name}' "
                "returned a negative depression; delta_h lowers the effective "
                "free surface and cannot be negative"
            )
        effective_depth = np.maximum(active_depths - depression, 0.0)

        depth_traction = (
            self.material.normal_stress_scale_pa_per_m
            * capped
            * effective_depth[:, None]
        )
        if self.dynamic_terms_active:
            inertial_traction = (
                -normals
                * (self.material.inertial_stress_scale_pa_s2_per_m2 * normal_speed**2)[
                    :, None
                ]
            )
        else:
            inertial_traction = np.zeros_like(depth_traction)

        clamped_area = float(active_areas[was_clamped].sum())
        total_area = float(active_areas.sum())
        clamped_fraction = clamped_area / total_area if total_area > 0.0 else 0.0

        response = ElementResponse(
            index=index,
            beta_rad=beta_rad,
            gamma_rad=gamma_rad,
            psi_rad=psi_rad,
            depth_m=active_depths,
            effective_depth_m=effective_depth,
            normal_speed_m_s=normal_speed,
            depth_traction_pa=depth_traction,
            inertial_traction_pa=inertial_traction,
            was_clamped=was_clamped,
        )
        verdict = self._verdict(
            speed_m_s=max_speed,
            element_size_m=elements.characteristic_length_m,
            submerged_depth_m=submerged_depth,
            clamped_area_fraction=clamped_fraction,
        )
        return response, verdict

    def _apply_surface_friction_cutoff(
        self,
        alpha_vector: NDArray[np.float64],
        normals: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Cap tangential traction at ``mu_surf`` times the normal traction.

        ``alpha = |alpha_n| (-n_hat) + min(mu_surf |alpha_n| / |alpha_t|, 1)
        alpha_t``.  The addendum records that the *normal* force is nearly
        independent of ``mu_surf`` over ``mu_int`` in 0.3-0.9, so this is
        deliberately a weak lever: it bounds an unphysical tangential
        traction rather than tuning the answer.

        Args:
            alpha_vector: ``(k, 3)`` generic response in world axes.
            normals: ``(k, 3)`` unit normals used by the fit.

        Returns:
            The capped response, ``(k, 3)``.
        """
        normal_component = np.einsum("ij,ij->i", alpha_vector, normals)
        tangential = alpha_vector - normal_component[:, None] * normals
        magnitude = np.sqrt(np.einsum("ij,ij->i", tangential, tangential))
        engaged = magnitude > 0.0
        limit = self.material.surface_friction_mu * np.abs(normal_component)
        scale = np.where(
            engaged,
            np.minimum(limit / np.where(engaged, magnitude, 1.0), 1.0),
            0.0,
        )
        return (
            np.abs(normal_component)[:, None] * (-normals) + scale[:, None] * tangential
        )

    # ---------------------------------------------------------- diagnostics

    def element_response(self, state: IntrusionState) -> ElementResponse:
        """Return the per-element intermediates for one query.

        Exposed so tests can assert on the depth/inertia split, the
        clamped fraction and the effective depth directly, instead of
        inferring them from a resultant.

        Args:
            state: The intrusion query.

        Returns:
            The per-element response over the active elements.
        """
        self._require_state(state)
        response, _ = self._evaluate(state)
        return response
