"""Time interpolation, Butterworth smoothing, and hysteresis (#15788)."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, sosfiltfilt

from src.shared.python.biomechanics.ground_reaction import center_of_pressure
from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.force_overlay import contracts as fc
from src.shared.python.force_overlay import glyphs as fg

Vec3 = tuple[float, float, float]
Arr = NDArray[np.float64]
Fn = Callable[[int], Any]
IdxFn = Callable[[int], int]
GlyphFn = Callable[[int, fg.ForceGlyphStyle], fg.GlyphSet]
SeqF = Sequence[float]
_ENG_R, _REL_R, _MIN_G, _MIN_S, _RAMP_F = 1.1, 0.75, 0.15, 0.5, 0.25


@dataclass(frozen=True)
class FlickerSuppressionConfig:
    """Settings for time-domain wrench filtering, interpolation, and hysteresis."""

    smooth: bool = True
    cutoff_hz: float = 20.0
    filter_order: int = 2
    engage_n: float | None = None
    release_n: float | None = None
    min_dwell_frames: int = 3
    cop_fz_min_n: float = 10.0

    def __post_init__(self) -> None:
        """Validate smoothing, threshold ordering, dwell, and CoP parameters."""
        e, r = self.engage_n, self.release_n
        c_ok = math.isfinite(self.cutoff_hz) and self.cutoff_hz > 0
        b = min(self.cop_fz_min_n, self.filter_order - 1, self.min_dwell_frames - 1)
        if not c_ok or b < 0:
            raise ValueError("Invalid FlickerSuppressionConfig parameter")
        if e is not None and r is not None and (e <= r or r < 0):
            raise ValueError(f"engage ({e}) must exceed release ({r}) >= 0")

    def thresholds(self, floor: float, *, torque: bool = False) -> tuple[float, float]:
        """Return ``(engage, release)`` for ``floor``, resolving overrides."""
        eng, rel = (None, None) if torque else (self.engage_n, self.release_n)
        e = floor * _ENG_R if eng is None else float(eng)
        if rel is None:
            return e, e * (_REL_R / _ENG_R) if eng is not None else floor * _REL_R
        return (max(e, float(rel) * 1.25) if eng is None else e), float(rel)


AL = ArrayLike
Cfg = FlickerSuppressionConfig
FTF = fc.ForceTorqueFrame
OW = fc.OverlayWrench
GS = fg.GlyphSet
Sty = fg.ForceGlyphStyle


@dataclass(frozen=True)
class HysteresisState:
    """Immutable per-glyph threshold hysteresis state."""

    active: bool = False
    dwell_frames: int = 0
    ever_switched: bool = False


StateMap = MutableMapping[str, HysteresisState]
Hyst = tuple[dict[str, HysteresisState], dict[int, GS], list[int]]


def _sc(p: Vec3, o: Vec3, s: float) -> Vec3:
    """Scale ``p`` around ``o`` by ``s``."""
    x, y, z = o
    return x + (p[0] - x) * s, y + (p[1] - y) * s, z + (p[2] - z) * s


def _step(
    sm: StateMap, k: str, mag: float, bounds: tuple[float, float], dwell: int
) -> float | None:
    """Advance hysteresis for ``k`` and return ramp gain or None."""
    eng, rel = bounds
    st = sm.get(k, HysteresisState())
    can = st.dwell_frames >= dwell or not st.ever_switched
    flip = ((mag < rel) if st.active else (mag >= eng)) and can
    on = (not st.active) if flip else st.active
    dn, ev = 1 if flip else st.dwell_frames + 1, st.ever_switched or on
    sm[k] = HysteresisState(on, dn, ev)
    if not on:
        return None
    hi = eng + _RAMP_F * max(eng - rel, 1e-6)
    u = float(np.clip((mag - rel) / max(hi - rel, 1e-9), 0.0, 1.0))
    return 1.0 if mag >= hi else _MIN_G + (1.0 - _MIN_G) * u * u * (3.0 - 2.0 * u)


def apply_hysteresis_ramp(
    glyph_set: GS, config: Cfg, style: Sty, states: StateMap | None = None
) -> GS:
    """Filter and ramp ``glyph_set`` using per-channel hysteresis and dwell."""
    sm: StateMap = {} if states is None else states
    fb = config.thresholds(style.magnitude_floor_n)
    tb = config.thresholds(style.magnitude_floor_nm, torque=True)
    d, arrows, arcs = config.min_dwell_frames, [], []
    for a in glyph_set.arrows:
        if (g := _step(sm, f"f:{a.label}", a.magnitude, fb, d)) is not None:
            if g < 1.0:
                s, r, c = _MIN_S + (1.0 - _MIN_S) * g, a.rgba, a.tail_m
                t, hb = _sc(a.tip_m, c, s), _sc(a.head_base_m, c, s)
                sr, hr = a.shaft_radius_m * s, a.head_radius_m * s
                a = replace(
                    a, tip_m=t, head_base_m=hb, rgba=(r[0], r[1], r[2], r[3] * g)
                )
                a = replace(a, shaft_radius_m=sr, head_radius_m=hr)
            arrows.append(a)
    for arc in glyph_set.torque_arcs:
        if (g := _step(sm, f"t:{arc.label}", arc.magnitude, tb, d)) is not None:
            if g < 1.0:
                s, r, c = _MIN_S + (1.0 - _MIN_S) * g, arc.rgba, arc.center_m
                pts = tuple(_sc(p, c, s) for p in arc.polyline_m)
                ht, hb = _sc(arc.head_tip_m, c, s), _sc(arc.head_base_m, c, s)
                arc = replace(arc, radius_m=arc.radius_m * s, polyline_m=pts)
                arc = replace(
                    arc,
                    head_tip_m=ht,
                    head_base_m=hb,
                    rgba=(r[0], r[1], r[2], r[3] * g),
                )
            arcs.append(arc)
    return replace(glyph_set, arrows=tuple(arrows), torque_arcs=tuple(arcs))


def zero_phase_butterworth_filter(
    values: AL, sample_rate_hz: float, cutoff_hz: float, order: int = 2, axis: int = 0
) -> Arr:
    """Zero-phase dual-pass Butterworth low-pass filter with short-series guard."""
    if sample_rate_hz <= 0 or cutoff_hz <= 0 or isinstance(order, bool) or order < 1:
        raise ValueError("Filter parameters must be positive")
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")
    if arr.size == 0 or arr.shape[axis] < 4 or cutoff_hz >= 0.5 * sample_rate_hz:
        return arr.copy()
    sos = butter(order, cutoff_hz / (0.5 * sample_rate_hz), btype="low", output="sos")
    pad = min(3 * (2 * sos.shape[0] + 1), arr.shape[axis] - 1)
    return np.asarray(sosfiltfilt(sos, arr, axis=axis, padlen=pad), dtype=np.float64)


def _is_bo(frame: FTF) -> bool:
    """Return True when ``frame`` marks an impact blackout window."""
    m, ax = frame.metadata, frame.axial_loads
    no_ax = bool(ax and ax.values_n and all(v is None for v in ax.values_n.values()))
    b = bool(m.get("axial_impact") == "unavailable" or m.get("contact_blackout"))
    return b or no_ax


def _recompute_cop(ws: list[OW], min_fz: float) -> tuple[OW, ...]:
    """Recompute CoP on ground-contact wrenches when ``Fz >= min_fz``."""
    out: list[OW] = []
    for w in ws:
        ok = w.kind == fc.WrenchKind.CONTACT and w.force_n is not None
        if ok and not w.label.startswith("contact:moment_com"):
            p, f = np.asarray(w.point_m, float), np.asarray(w.force_n, float)
            m0 = np.cross(p, f) + np.asarray(w.torque_nm or (0.0, 0.0, 0.0), float)
            cop = center_of_pressure(f, m0, ground_height_m=float(p[2]), min_fz=min_fz)
            if cop is not None:
                cp: Vec3 = (float(cop[0]), float(cop[1]), float(cop[2]))
                mz = float(m0[2] - cp[0] * f[1] + cp[1] * f[0])
                nt = (0.0, 0.0, mz) if w.torque_nm else None
                w = replace(w, point_m=cp, torque_nm=nt)
        out.append(w)
    return tuple(out)


def interpolate_wrenches_to_frame_times(
    source_frames: Sequence[FTF],
    source_times_s: SeqF,
    target_times_s: SeqF,
    *,
    contact_active: Sequence[bool] | None = None,
    config: Cfg | None = None,
) -> tuple[FTF, ...]:
    """Interpolate and Butterworth-filter 6D wrenches onto ``target_times_s``."""
    n_s, ca = len(source_frames), contact_active
    if not n_s or n_s != len(source_times_s) or (ca is not None and len(ca) != n_s):
        raise ValueError("Mismatched input lengths")
    st, tt = np.asarray(source_times_s, float), np.asarray(target_times_s, float)
    bad = st.ndim != 1 or not np.all(np.isfinite(st)) or not np.all(np.isfinite(tt))
    if bad or (n_s > 1 and not np.all(np.diff(st) > 0.0)):
        raise ValueError("Invalid source_times_s or target_times_s")
    if tt.size == 0:
        return ()
    cfg = config or FlickerSuppressionConfig()
    bo = np.asarray([_is_bo(f) for f in source_frames], bool)
    r_i = np.clip(np.searchsorted(st, tt, side="left"), min(1, n_s - 1), n_s - 1)
    l_i = np.maximum(0, r_i - 1)
    near = np.where((tt - st[l_i]) <= (st[r_i] - tt), l_i, r_i)
    fws: list[list[OW]] = [[] for _ in range(tt.size)]
    if cfg.smooth:
        c_ok = np.ones(n_s, bool) if ca is None else np.asarray(ca, bool)
        cross_bo, z0 = bo[l_i] | bo[r_i], (0.0, 0.0, 0.0)
        sp = (("point_m", z0), ("force_n", None), ("torque_nm", None))
        by_lbl = [{w.label: w for w in f.wrenches} for f in source_frames]
        for lbl in sorted({k for m in by_lbl for k in m}):
            ws = [m.get(lbl) for m in by_lbl]
            is_c = next(w for w in ws if w is not None).kind == fc.WrenchKind.CONTACT
            hw = np.array([w is not None for w in ws])
            mask = (~bo) & hw & (c_ok if is_c else True)
            edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
            for a, b in zip(edges[0::2], edges[1::2] - 1, strict=True):
                sub = [w for w in ws[a : b + 1] if w is not None]
                in_s = (a <= l_i) & (r_i <= b) & (~cross_bo)
                if (idx := np.flatnonzero(in_s)).size:
                    st_s, qt = st[a : b + 1], tt[idx]
                    dt = float(np.median(np.diff(st_s))) if st_s.size >= 4 else 0.0
                    do_f, ip = dt > 0.0 and cfg.cutoff_hz < 0.5 / dt, np.interp
                    cols: list[list[Any]] = []
                    for attr, fb in sp:
                        raw = [getattr(w, attr) for w in sub]
                        if any(v is None for v in raw):
                            cols.append([fb] * int(qt.size))
                            continue
                        vals = np.asarray(raw, float)
                        if do_f:
                            zpf = zero_phase_butterworth_filter
                            vals = zpf(vals, 1.0 / dt, cfg.cutoff_hz, cfg.filter_order)
                        c3 = [ip(qt, st_s, v).tolist() for v in vals.T]
                        cols.append(list(zip(*c3, strict=True)))
                    rw, w0, (pc, fc_, tc) = replace, sub[0], cols
                    for pt, fn, tm, j in zip(pc, fc_, tc, idx, strict=True):
                        if fn is not None or tm is not None:
                            w = rw(w0, point_m=pt, force_n=fn, torque_nm=tm)
                            fws[int(j)].append(w)
                for j in np.flatnonzero((~in_s) & (a <= near) & (near <= b)):
                    if (wn := ws[int(near[j])]) is not None:
                        fws[int(j)].append(wn)
    fm, out = cfg.cop_fz_min_n, []
    for fi, t_raw in enumerate(tt):
        t, k = float(t_raw), int(near[fi])
        ref, is_bo = source_frames[k], bool(bo[k])
        ok = cfg.smooth and not is_bo
        ws_out = _recompute_cop(fws[fi], fm) if ok else ref.wrenches
        ra = ref.axial_loads
        vd = (dict.fromkeys(ra.values_n) if is_bo else dict(ra.values_n)) if ra else {}
        ax = None if ra is None else AxialLoadFrame(t, vd, ra.source)
        out.append(replace(ref, time_s=t, wrenches=ws_out, axial_loads=ax))
    return tuple(out)


def resolve_following_fn(
    source: Fn, src_idx: IdxFn, swing: Any, fcfg: Cfg | None, smooth: bool
) -> tuple[Fn, Cfg | None]:
    """Return ``(frame_fn, active_cfg)`` for ``OverlayFeed.following``."""
    st = getattr(swing, "source_times_s", None)
    tt = getattr(swing, "sample_times_s", None)
    if not smooth or st is None or tt is None or len(st) < 2 or len(tt) == 0:
        return (lambda i: source(src_idx(int(i)))), (fcfg if smooth else None)
    cfg = fcfg or FlickerSuppressionConfig(smooth=True)
    iw = interpolate_wrenches_to_frame_times
    cache: list[Any] = []

    def _at(index: int) -> Any:
        """Return interpolated frame at ``index``."""
        if not cache:
            sf = [source(k) for k in range(len(st))]
            if any(x is None or not hasattr(x, "wrenches") for x in sf):
                return source(src_idx(int(index)))
            cache.extend(iw(sf, st, tt, config=cfg))
        return cache[min(max(int(index), 0), len(cache) - 1)]

    return _at, cfg


def step_feed_glyphs(
    raw_fn: GlyphFn, index: int, style: Sty, cfg: Cfg, hyst: Hyst
) -> GS:
    """Step hysteresis sequentially up to ``index`` and return ramped ``GlyphSet``."""
    states, cache, cursor = hyst
    if index not in cache:
        th, mfn, mfm = cfg.thresholds, style.magnitude_floor_n, style.magnitude_floor_nm
        fn = min(mfn, max(th(mfn)[1], 1e-6))
        tn = min(mfm, max(th(mfm, torque=True)[1], 1e-6))
        rs = replace(style, magnitude_floor_n=fn, magnitude_floor_nm=tn)
        for k in range(cursor[0] + 1, index + 1):
            cache[k] = apply_hysteresis_ramp(raw_fn(k, rs), cfg, style, states)
            cursor[0] = k
    return cache[index]
