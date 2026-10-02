function w = gs3dx_frame_rate(R, dt)
%GS3DX_FRAME_RATE  Body-frame angular velocity from a rotation sequence (#11173).
%
%   W = GS3DX_FRAME_RATE(R, DT) with R (3x3xN) rotation matrices (columns
%   are body axes in the parent frame) and DT > 0 returns W (3xN), angular
%   velocity in the body frame (rad/s), from skew(W) = R.' * dR/dt with
%   dR/dt from gradient along the frame axis.  At N == 1, W is zero.
%
%   Preconditions: each slice of R is a rotation (R.'*R = I, det(R) > 0
%   within 1e-6); DT is finite and positive.
%   Postcondition: constant R gives W = 0; rotation about parent z by
%   theta = w0*t gives body W close to [0;0;w0] on an interior frame.

    arguments
        R (3, 3, :) double
        dt (1, 1) double {mustBePositive, mustBeFinite}
    end

    n = size(R, 3);
    assert(size(R, 1) == 3 && size(R, 2) == 3, 'gs3dx:spin', ...
        'R must be 3x3xN rotation matrices');

    w = zeros(3, n);
    if n == 1
        return;
    end

    eye3 = eye(3);
    for k = 1:n
        Rk = R(:, :, k);
        if max(abs(Rk.' * Rk - eye3), [], 'all') > 1e-6 || det(Rk) <= 1e-6
            error('gs3dx:spin', 'Frame %d is not a valid rotation matrix', k);
        end
    end

    for k = 1:n
        if k == 1
            w(:, k) = body_rate_step(R(:, :, k), R(:, :, k + 1), dt);
        elseif k == n
            w(:, k) = body_rate_step(R(:, :, k), R(:, :, k - 1), -dt);
        else
            wf = body_rate_step(R(:, :, k), R(:, :, k + 1), dt);
            wb = body_rate_step(R(:, :, k), R(:, :, k - 1), -dt);
            w(:, k) = (wf + wb) / 2;
        end
    end
end

function w = body_rate_step(R0, R1, dt)
    S = R0.' * R1;
    c = max(-1, min(1, (trace(S) - 1) / 2));
    s = sqrt(max(0, 1 - c * c));
    if s < 1e-14
        w = zeros(3, 1);
        return;
    end
    axis = [S(3, 2) - S(2, 3); S(1, 3) - S(3, 1); S(2, 1) - S(1, 2)] / (2 * s);
    theta = atan2(s, c);
    w = axis * (theta / dt);
end
