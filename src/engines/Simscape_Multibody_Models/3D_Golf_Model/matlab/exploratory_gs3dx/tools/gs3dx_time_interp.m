function [k, k2, w] = gs3dx_time_interp(t, T)
%GS3DX_TIME_INTERP  Bracketing samples of a time in a table (#10979).
%
%   [K, K2, W] = GS3DX_TIME_INTERP(T_NOW, T) finds, in the increasing vector
%   T, the samples K <= K2 around T_NOW (clamped to [T(1), T(end)]) and the
%   weight W in [0, 1], so a table X (one column per time) is
%       X(:, K) + W * (X(:, K2) - X(:, K))
%   at T_NOW.  Bisection; code-generation compatible (called from the
%   MATLAB Function charts of GS3DX_FitTrack and GS3DX_FitBalance).

    n = numel(T);
    tc = min(max(t, T(1)), T(n));
    lo = 1;
    hi = n;
    while hi - lo > 1   % bisection: T(lo) <= tc <= T(hi)
        mid = floor((lo + hi) / 2);
        if T(mid) <= tc
            lo = mid;
        else
            hi = mid;
        end
    end
    k = lo;
    k2 = min(k + 1, n);
    w = 0;
    if k2 > k && T(k2) > T(k)
        w = (tc - T(k)) / (T(k2) - T(k));
    end
end
