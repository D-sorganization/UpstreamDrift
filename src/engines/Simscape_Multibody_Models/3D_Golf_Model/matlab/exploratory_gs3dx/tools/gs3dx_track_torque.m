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

    [k, k2, w] = gs3dx_time_interp(t, T);
    at = @(X) X(:, k) + w * (X(:, k2) - X(:, k));
    tau = at(F) + Kp(:) .* (at(A) - q(:)) + Kd(:) .* (at(R) - qd(:));
end
