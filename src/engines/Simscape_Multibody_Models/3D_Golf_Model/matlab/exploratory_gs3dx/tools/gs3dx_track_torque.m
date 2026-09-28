function tau = gs3dx_track_torque(t, T, A, R, F, Kp, Kd, q, qd)
%GS3DX_TRACK_TORQUE  Feedforward plus PD tracking torque of a joint reference (#10979).
%
%   TAU = GS3DX_TRACK_TORQUE(T_NOW, T, A, R, F, KP, KD, Q, QD) is
%     F(t) + KP .* (A(t) - Q) + KD .* (R(t) - QD)
%   with the reference angle A, rate R and feedforward torque F (one row per
%   joint axis, one column per time in the increasing vector T) linearly
%   interpolated at T_NOW, and held at their first and last columns outside
%   T.  Q and QD are the joint's angles and rates.  Called by the
%   upper-body '<J> Input Function' charts of GS3DX_FitTrack, so it is
%   code-generation compatible.

    n = numel(T);
    tc = min(max(t, T(1)), T(n));
    k = 1;
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
    if n > 1
        k = lo;
    end
    w = 0;
    if n > 1 && T(k + 1) > T(k)
        w = (tc - T(k)) / (T(k + 1) - T(k));
    end
    k2 = min(k + 1, n);
    at = @(X) X(:, k) + w * (X(:, k2) - X(:, k));
    tau = at(F) + Kp(:) .* (at(A) - q(:)) + Kd(:) .* (at(R) - qd(:));
end
