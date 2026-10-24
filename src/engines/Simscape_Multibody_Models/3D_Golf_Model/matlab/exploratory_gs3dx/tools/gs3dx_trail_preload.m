function h = gs3dx_trail_preload(t, fz, t_on, preload, ki)
%GS3DX_TRAIL_PRELOAD  Trail-foot height preload from contact force (#10979).
%
%   H = GS3DX_TRAIL_PRELOAD(T, FZ, T_ON, PRELOAD, KI) returns 1xN metres,
%   nonnegative.  T and FZ are 1xN.  T_ON, PRELOAD (N, upward) and KI
%   (m per N per s) are scalars.  KI defaults to 1e-4.  T_ON defaults to 1.35.
%   PRELOAD defaults to 0.1*80*9.81.  For t < t_on, h is 0.  For t >= t_on,
%       dh/dt = ki * max(preload - fz, 0)
%   integrated with cumulative trapezoid from the first sample at or after t_on.
%
%   Error id gs3dx:preload.

    if nargin < 5 || isempty(ki)
        ki = 1e-4;
    end
    if nargin < 4 || isempty(preload)
        preload = 0.1 * 80 * 9.81;
    end
    if nargin < 3 || isempty(t_on)
        t_on = 1.35;
    end

    t = t(:).';
    fz = fz(:).';
    n = numel(t);
    assert(n >= 1, 'gs3dx:preload', 'T must be non-empty');
    assert(numel(fz) == n, 'gs3dx:preload', 'FZ must match T length');
    assert(all(isfinite(t)), 'gs3dx:preload', 'T must be finite');
    assert(all(isfinite(fz)), 'gs3dx:preload', 'FZ must be finite');
    assert(all(diff(t) > 0), 'gs3dx:preload', 'T must be strictly increasing');
    assert(isfinite(t_on), 'gs3dx:preload', 'T_ON must be finite');
    assert(preload >= 0, 'gs3dx:preload', 'PRELOAD must be non-negative');
    assert(ki >= 0, 'gs3dx:preload', 'KI must be non-negative');

    h = zeros(1, n);
    idx = find(t >= t_on, 1, 'first');
    if isempty(idx)
        local_assert_postconditions(h, t, t_on);
        return
    end

    tSeg = t(idx:end);
    rate = ki * max(preload - fz(idx:end), 0);
    h(idx:end) = cumtrapz(tSeg, rate);

    local_assert_postconditions(h, t, t_on);
end

function local_assert_postconditions(h, t, t_on)
    assert(all(h >= -1e-12), 'gs3dx:preload', 'Preload height must be non-negative');
    assert(all(h(t < t_on) <= 1e-12), 'gs3dx:preload', ...
        'Preload height must be zero before T_ON');
    if numel(h) >= 2
        assert(all(diff(h) >= -1e-12), 'gs3dx:preload', ...
            'Preload height must be nondecreasing');
    end
end
