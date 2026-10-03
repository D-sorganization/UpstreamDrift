function [cmd, shift] = gs3dx_balance_command(t, com, com_rate, feet, T, C0, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, on)
%GS3DX_BALANCE_COMMAND  Leg servo command with centre-of-mass and foot feedback (#10979).
%
%   [CMD, SHIFT] = GS3DX_BALANCE_COMMAND(T_NOW, COM, COM_RATE, FEET, T, C0,
%   KP, KD, A, R, CREF, VREF, FREF, G, BKP, BKD, BKF, LIMIT, ON) is the
%   12-axis command of the GS3DX_FitBalance leg servo, which applies
%   CMD - [Kp Kd] [q; qd]:
%
%     CMD = C0(t) + KP .* (A(t) + G(t) SHIFT + FOOT) + KD .* (R(t) + G(t) SHIFT_RATE)
%
%   C0: constant 12-vector (column or row) OR 12 x numel(T) sampled feedforward
%   on reference time grid T (N*m).
%   A, R: leg reference angles and rates (12 x frames, deg and deg/s) on T.
%   COM, COM_RATE: the measured whole-body centre of mass and its rate
%   (World, m and m/s); CREF, VREF: their references (3 x frames).  The
%   error e in the first D World axes (D = SIZE(G, 2): 2, horizontal, or
%   3, also vertical) sets the pelvis shift relative to the feet
%       SHIFT = -BKP e - BKD de/dt,  |SHIFT| <= LIMIT (m),  SHIFT_RATE = -BKP de/dt
%   and G (12 x D x frames, deg/m, GS3DX_BALANCE_GAIN) turns it into leg
%   angle offsets that keep both feet where the reference puts them.
%   FEET = [left; right] ankle positions (6 x 1, World, m) and FREF their
%   reference (6 x frames): each leg also moves its foot back toward the
%   reference, FOOT = G_leg(t) (BKF e_foot) with |BKF e_foot| <= LIMIT.
%   That is the pelvis-shift gain read the other way: a foot moved by -x
%   relative to the pelvis is the pelvis moved by x.  With ON = 0 the
%   command is the plain reference servo of GS3DX_FitLegs.
%   Code-generation compatible (called by 'Lower Body/Leg Torque Commands').

    assert(isnumeric(C0) && isreal(C0), 'gs3dx:balance_command:invalidType', ...
        'Feedforward torque C0 must be a real numeric array.');
    nT = numel(T);
    is_const12 = (numel(C0) == 12) && (size(C0, 1) == 1 || size(C0, 2) == 1);
    is_profile = (size(C0, 1) == 12) && (size(C0, 2) == nT);
    assert(all(isfinite(C0(:))), 'gs3dx:balance_command:nonfinite', ...
        'Feedforward torque C0 must be finite.');
    assert(ismatrix(C0) && (is_const12 || is_profile), 'gs3dx:balance_command:invalidShape', ...
        'Feedforward torque C0 must be a 12-vector or a 12 x numel(T) matrix.');

    [k, k2, w] = gs3dx_time_interp(t, T);
    at = @(X) X(:, k) + w * (X(:, k2) - X(:, k));

    if is_profile
        c0_now = at(C0);
    else
        c0_now = C0(:);
    end

    d = size(G, 2);
    e = com(1:d) - at(Cref(1:d, :));
    de = com_rate(1:d) - at(Vref(1:d, :));
    shift = local_limit(on * (-kp * e - kd * de), limit);
    shift_rate = on * (-kp * de);
    Gt = G(:, :, k) + w * (G(:, :, k2) - G(:, :, k));
    ef = feet(:) - at(Fref);
    foot = zeros(12, 1);
    for s = 1:2
        rows = (s - 1) * 6 + (1:6);
        foot(rows) = Gt(rows, :) * local_limit(on * kf * ef((s - 1) * 3 + (1:d)), limit);
    end
    cmd = c0_now + Kp(:) .* (at(A) + Gt * shift + foot) + Kd(:) .* (at(R) + Gt * shift_rate);
end

function x = local_limit(x, limit)
    n = norm(x);
    if n > limit
        x = x * (limit / n);
    end
end
