"""Vectorized value iteration over the hole MDP.

Two implementations are kept side-by-side:

* ``bellman_backup_scalar`` — readable, pure-Python loop; reference behaviour.
* ``HoleMDP.bellman_backup`` — vectorized; must agree with the scalar version
  on small grids to within float tolerance (property-tested).

The state space is the 2D ball-position grid × discrete lie code. Phase 1
collapses lie ambiguity by tying lie to the raster cell — i.e. every (x, y)
has a unique lie taken from the raster, so V is indexed by (ix, iy). This
keeps memory tractable while still honouring the per-lie shot-model
modifiers via the starting-lie lookup.

(Spec §1.5, pitfall #11.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES, LieRaster
from src.shared.python.sg_optimizer.mdp.action import ActionSet, ShotAction
from src.shared.python.sg_optimizer.mdp.state import State
from src.shared.python.sg_optimizer.mdp.transition import (
    _condition_modifiers,
    sample_transitions,
)

if TYPE_CHECKING:  # pragma: no cover
    from src.shared.python.sg_optimizer.course.conditions import CourseConditions
    from src.shared.python.sg_optimizer.shot_model.baseline import BaselineBag
    from src.shared.python.sg_optimizer.shot_model.player_profile import PlayerProfile


_HOLED = LIE_CODES["holed"]
_WATER = LIE_CODES["water"]
_OB = LIE_CODES["ob"]


# ---------------------------------------------------------------------------
# Scalar reference implementation
# ---------------------------------------------------------------------------


def bellman_backup_scalar(
    V: NDArray[np.float64],
    raster: LieRaster,
    profile: PlayerProfile,
    baseline: BaselineBag,
    conditions: CourseConditions,
    actions: ActionSet,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Single Bellman sweep, pure Python loop.

    Returns ``(V_new, argmax_action_index)``. Holed cells stay at V=0.
    """
    nx, ny = raster.shape
    require(V.shape == (nx, ny), f"V shape {V.shape} != raster {raster.shape}")
    require(n_samples > 0, "n_samples must be > 0")

    V_new = np.zeros_like(V)
    pi = np.zeros((nx, ny), dtype=np.int32)
    action_list = list(actions.iter_actions())

    for ix in range(nx):
        for iy in range(ny):
            lie = int(raster.codes[ix, iy])
            if lie == _HOLED:
                continue
            if lie in (_WATER, _OB):
                # Treat as a 1-stroke penalty; cost is the tee/drop V at origin.
                # Practical heuristic for Phase 1 — water/OB transitions are
                # already handled in sample_transitions so this branch is just
                # a placeholder ensuring V remains finite.
                V_new[ix, iy] = V[ix, iy] + 1.0
                continue

            x = raster.origin[0] + (ix + 0.5) * raster.resolution_yd
            y = raster.origin[1] + (iy + 0.5) * raster.resolution_yd
            state = State(x=x, y=y, lie=lie)

            best_q = np.inf
            best_idx = 0
            for ai, action in enumerate(action_list):
                if action.club == "putter" and lie != LIE_CODES["green"]:
                    continue
                outcomes = sample_transitions(
                    state,
                    action,
                    profile,
                    baseline,
                    conditions,
                    raster,
                    n_samples,
                    rng,
                )
                total = 0.0
                for o in outcomes:
                    nix, niy = raster.world_to_index(o.next_state.x, o.next_state.y)
                    if 0 <= nix < nx and 0 <= niy < ny:
                        v_next = V[nix, niy]
                    else:
                        v_next = V[ix, iy] + 1.0  # off-grid → drop penalty
                    total += 1.0 + o.extra_strokes + v_next
                q = total / len(outcomes)
                if q < best_q:
                    best_q = q
                    best_idx = ai
            V_new[ix, iy] = best_q
            pi[ix, iy] = best_idx

    return V_new, pi


# ---------------------------------------------------------------------------
# Vectorized solver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolveResult:
    """Output of ``HoleMDP.solve``."""

    value: NDArray[np.float64]
    policy: NDArray[np.int32]
    iterations: int
    delta: float


class HoleMDP:
    """Vectorized hole-MDP solver."""

    def __init__(
        self,
        raster: LieRaster,
        profile: PlayerProfile,
        baseline: BaselineBag,
        conditions: CourseConditions,
        actions: ActionSet,
        n_samples: int = 64,
        seed: int = 0,
    ) -> None:
        require(n_samples > 0, "n_samples must be > 0")
        self.raster = raster
        self.profile = profile
        self.baseline = baseline
        self.conditions = conditions
        self.actions = actions
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self._action_list: list[ShotAction] = list(actions.iter_actions())
        self._terminal_mask = raster.codes == _HOLED

    # --- public API ------------------------------------------------------

    def bellman_backup(
        self, V: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """Vectorized single sweep over all cells × actions."""
        nx, ny = self.raster.shape
        nA = len(self._action_list)
        Q = np.full((nx, ny, nA), np.inf, dtype=np.float64)

        for ai, action in enumerate(self._action_list):
            Q[:, :, ai] = self._action_q(V, action)

        V_new = np.min(Q, axis=2)
        V_new[self._terminal_mask] = 0.0
        pi = np.argmin(Q, axis=2).astype(np.int32)
        pi[self._terminal_mask] = 0
        return V_new, pi

    def solve(
        self,
        tol: float = 1e-3,
        max_iter: int = 200,
        warm_start: NDArray[np.float64] | None = None,
    ) -> SolveResult:
        """Iterate Bellman backups to convergence or ``max_iter``."""
        require(tol > 0, "tol must be > 0")
        require(max_iter > 0, "max_iter must be > 0")
        nx, ny = self.raster.shape
        V = warm_start.copy() if warm_start is not None else self._initial_value()
        require(V.shape == (nx, ny), "warm_start shape mismatch")

        last_delta = np.inf
        for it in range(1, max_iter + 1):
            V_new, pi = self.bellman_backup(V)
            last_delta = float(np.max(np.abs(V_new - V)))
            V = V_new
            if last_delta < tol:
                return SolveResult(value=V, policy=pi, iterations=it, delta=last_delta)
        return SolveResult(value=V, policy=pi, iterations=max_iter, delta=last_delta)

    def optimal_action(self, state: State, value: NDArray[np.float64]) -> ShotAction:
        ix, iy = self.raster.world_to_index(state.x, state.y)
        # One-step lookahead from the requested state.
        best, best_q = self._action_list[0], np.inf
        nx, ny = self.raster.shape
        for action in self._action_list:
            if action.club == "putter" and state.lie != LIE_CODES["green"]:
                continue
            q = float(
                self._action_q(value, action)[
                    ix if 0 <= ix < nx else 0, iy if 0 <= iy < ny else 0
                ]
            )
            if q < best_q:
                best, best_q = action, q
        return best

    def expected_strokes(self, state: State, value: NDArray[np.float64]) -> float:
        ix, iy = self.raster.world_to_index(state.x, state.y)
        nx, ny = self.raster.shape
        if not (0 <= ix < nx and 0 <= iy < ny):
            return float("inf")
        return float(value[ix, iy])

    # --- internals -------------------------------------------------------

    def _initial_value(self) -> NDArray[np.float64]:
        """Distance-to-pin / 100-yd heuristic — admissible-ish warm start."""
        nx, ny = self.raster.shape
        ix = np.arange(nx)[:, None]
        iy = np.arange(ny)[None, :]
        x = self.raster.origin[0] + (ix + 0.5) * self.raster.resolution_yd
        y = self.raster.origin[1] + (iy + 0.5) * self.raster.resolution_yd
        d = np.hypot(x - self.raster.pin[0], y - self.raster.pin[1])
        V0 = 1.0 + d / 100.0
        V0[self._terminal_mask] = 0.0
        return V0

    def _action_q(
        self, V: NDArray[np.float64], action: ShotAction
    ) -> NDArray[np.float64]:
        """Vectorized expected one-step cost-to-go for one action across all cells.

        Samples offsets once and reuses them across all cells. Per-cell starting
        lie applies a *condition multiplier* — but condition modifiers depend
        only on the lie class (3 of them in Phase 1: rough, trees, sand vs
        rest), so we precompute the modifier for each cell from its lie code
        and broadcast.
        """
        nx, ny = self.raster.shape
        codes = self.raster.codes

        # Per-cell modifiers (vectorized lookup table).
        mod_table = self._modifier_table()  # shape (n_lie_codes, 3)
        cell_mods = mod_table[codes]  # (nx, ny, 3)

        # Sample once in aim-frame.
        cb = self.baseline.get(action.club)
        skill = self.profile.clubs.get(action.club)
        sl = cb.sigma_long * (skill.skill_mult_long if skill else 1.0)
        slat = cb.sigma_lat * (skill.skill_mult_lat if skill else 1.0)
        rho = cb.rho
        cov = np.array([[sl * sl, rho * sl * slat], [rho * sl * slat, slat * slat]])
        offsets = self.rng.multivariate_normal([0.0, 0.0], cov, size=self.n_samples)
        base_dist = self.baseline.get(action.club).total_mean + (
            skill.distance_offset if skill else 0.0
        )
        # Player + baseline shot bias (chronic miss). Matches
        # PlayerProfile.effective_distribution() / sample_transitions(): sigma
        # gets scaled by condition modifiers, but bias does not.
        bias_long = cb.bias_long + (skill.bias_long if skill else 0.0)
        bias_lat = cb.bias_lat + (skill.bias_lat if skill else 0.0)

        # Apply rotation matrix once per action.
        ca = np.cos(action.aim_angle_rad)
        sa = np.sin(action.aim_angle_rad)

        # Per-cell along/lateral after applying per-cell mods.
        # along = base_dist * dist_mult + dlong * sigma_long_mult
        # lat   = dlat * sigma_lat_mult
        # dx = along * ca - lat * sa
        # dy = along * sa + lat * ca
        # We compute one (nx, ny, n_samples) tensor of next indices in a
        # memory-friendly way: vectorize over samples in an inner loop chunk.
        cell_x = (
            self.raster.origin[0] + (np.arange(nx) + 0.5) * self.raster.resolution_yd
        )
        cell_y = (
            self.raster.origin[1] + (np.arange(ny) + 0.5) * self.raster.resolution_yd
        )
        cx, cy = np.meshgrid(cell_x, cell_y, indexing="ij")  # (nx, ny)

        dist_mult = cell_mods[..., 0]  # (nx, ny)
        sigma_long_mult = cell_mods[..., 1]  # (nx, ny) — longitudinal dispersion

        # Putter restriction: putter only valid on the green; otherwise +inf.
        green_mask = codes == LIE_CODES["green"] if action.club == "putter" else None

        q_sum = np.zeros((nx, ny), dtype=np.float64)
        valid = ~self._terminal_mask

        for k in range(self.n_samples):
            dlong, dlat = offsets[k]
            along = (
                base_dist * dist_mult + bias_long + dlong * sigma_long_mult
            )  # (nx, ny)
            lat = bias_lat + dlat * cell_mods[..., 2]  # σ_lat scaling
            dx = along * ca - lat * sa
            dy = along * sa + lat * ca
            xn = cx + dx
            yn = cy + dy

            # Convert to indices.
            ixn = ((xn - self.raster.origin[0]) / self.raster.resolution_yd).astype(
                np.int32
            )
            iyn = ((yn - self.raster.origin[1]) / self.raster.resolution_yd).astype(
                np.int32
            )

            in_bounds = (ixn >= 0) & (ixn < nx) & (iyn >= 0) & (iyn < ny)
            ixn_c = np.clip(ixn, 0, nx - 1)
            iyn_c = np.clip(iyn, 0, ny - 1)
            land_codes = codes[ixn_c, iyn_c]
            v_next = V[ixn_c, iyn_c]

            # Hazard penalty: water/OB → drop back at current cell + 1 stroke.
            hazard = (land_codes == _WATER) | (land_codes == _OB) | (~in_bounds)
            cost = np.where(hazard, 1.0 + 1.0 + V, 1.0 + v_next)
            q_sum += cost

        q_mean = q_sum / float(self.n_samples)
        q_mean[~valid] = 0.0
        if green_mask is not None:
            invalid_putter = ~green_mask & valid
            q_mean[invalid_putter] = np.inf
        return q_mean

    def _modifier_table(self) -> NDArray[np.float64]:
        """Lookup table (lie_code → (dist_mult, sl_mult, slat_mult))."""
        rows = []
        for code in range(max(LIE_CODES.values()) + 1):
            rows.append(_condition_modifiers(code, self.conditions))
        return np.asarray(rows, dtype=np.float64)
