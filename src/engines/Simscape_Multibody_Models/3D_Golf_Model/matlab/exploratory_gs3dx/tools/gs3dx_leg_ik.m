function [q, residual] = gs3dx_leg_ik(geom, pelvis_R, pelvis_p, foot_R, foot_p, q0)
%GS3DX_LEG_IK  Leg joint angles that put a GS3DX foot frame on a target pose.
%
%   [Q, RESIDUAL] = GS3DX_LEG_IK(GEOM, PELVIS_R, PELVIS_P, FOOT_R, FOOT_P, Q0)
%   solves GS3DX_LEG_FK(GEOM, PELVIS_R(:,:,k), PELVIS_P(:,k), Q(:,k)) ==
%   (FOOT_R, FOOT_P) for every pelvis pose k (PELVIS_R 3x3xN, PELVIS_P 3xN)
%   with damped Gauss-Newton, seeding each pose with the previous solution
%   (Q0, 6x1 degrees, for the first), so the branch (knee forward) carries
%   along a smooth trajectory.  Six equations (foot position and
%   orientation), six unknowns [hip X Y Z, knee, ankle X Y] in degrees.
%   RESIDUAL(k) is the final residual norm (m and rad combined); the
%   function errors if any pose does not converge below 1e-9, which is
%   also how an out-of-reach pelvis is reported.

    arguments
        geom (1,1) struct
        pelvis_R (3,3,:) double
        pelvis_p (3,:) double
        foot_R (3,3) double
        foot_p (3,1) double
        q0 (6,1) double
    end
    n = size(pelvis_R, 3);
    assert(size(pelvis_p, 2) == n, 'gs3dx:ik', 'Pelvis rotations and positions differ in length');
    q = zeros(6, n);
    residual = zeros(1, n);
    x = q0;
    for k = 1:n
        f = @(v) local_error(geom, pelvis_R(:, :, k), pelvis_p(:, k), v, foot_R, foot_p);
        [x, residual(k)] = local_solve(f, x);
        assert(residual(k) < 1e-9, 'gs3dx:ik', ...
            'IK did not converge at pose %d of %d (residual %.3g): target out of reach?', k, n, residual(k));
        q(:, k) = x;
    end
end

function e = local_error(geom, pR, pp, v, foot_R, foot_p)
    [R, p] = gs3dx_leg_fk(geom, pR, pp, v);
    E = foot_R.' * R;
    e = [p - foot_p; 0.5 * [E(3, 2) - E(2, 3); E(1, 3) - E(3, 1); E(2, 1) - E(1, 2)]];
end

function [x, r] = local_solve(f, x)
    e = f(x);
    r = norm(e);
    lambda = 1e-6;
    for it = 1:100
        if r < 1e-12
            break;
        end
        J = zeros(6);
        h = 1e-6;
        for c = 1:6
            d = zeros(6, 1);
            d(c) = h;
            J(:, c) = (f(x + d) - f(x - d)) / (2 * h);
        end
        step = -(J.' * J + lambda * eye(6)) \ (J.' * e);
        trial = x + step;
        et = f(trial);
        if norm(et) < r
            x = trial;
            e = et;
            r = norm(e);
            lambda = max(lambda / 10, 1e-12);
        else
            lambda = lambda * 10;
        end
    end
end
