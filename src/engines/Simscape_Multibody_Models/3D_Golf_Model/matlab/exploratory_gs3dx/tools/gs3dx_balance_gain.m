function G = gs3dx_balance_gain(ref, ws, opts)
%GS3DX_BALANCE_GAIN  Leg angle offsets per pelvis shift, feet held (#10979).
%
%   G = GS3DX_BALANCE_GAIN(REF, WS) is, for every frame of the leg reference
%   REF (GS3DX_LEG_REFERENCE), the 12 x D matrix (deg/m) that turns a
%   pelvis shift along the first D World axes (D = opts.axes: 2, x and y;
%   3, also z), at fixed pelvis orientation, into the offsets of the 12 leg
%   angles that keep both feet on their reference pose.  Per leg, with J the 6 x 6 Jacobian of the foot pose (position, m;
%   rotation vector, rad) over [hip X Y Z, knee, ankle X Y] (deg),
%       dq = -J' (J J' + damping^2 I) \ [shift; 0]
%   the damped least-squares inverse, which stays bounded where the lead
%   knee straightens.  WS is the model workspace that holds the leg
%   geometry (<L|R>HipMountRotation, <L|R>HipMountOffset, ThighLength,
%   ShankLength).  G is 12 x D x frames.
%
%   Options: axes (3), damping (0.002 m/deg, about a tenth of J's typical
%   singular values, so it acts only near the straight knee), lever_m (0.5 m, the
%   length that weighs a foot rotation against a translation), step_deg
%   (1e-3, the finite-difference step).

    arguments
        ref (1,1) struct
        ws
        opts.axes (1,1) double {mustBeMember(opts.axes, [2 3])} = 3
        opts.damping (1,1) double {mustBePositive} = 0.002
        opts.lever_m (1,1) double {mustBePositive} = 0.5
        opts.step_deg (1,1) double {mustBePositive} = 1e-3
    end
    n = numel(ref.frames);
    assert(isequal(size(ref.q), [12 n]) && size(ref.pelvis_R, 3) == n, 'gs3dx:balancegain', ...
        'Precondition: REF must hold 12 x frames leg angles and a pelvis pose per frame');
    d = opts.axes;
    G = zeros(12, d, n);
    sides = 'LR';
    for s = 1:2
        P = sides(s);
        geom = struct('mount_R', ws.getVariable([P 'HipMountRotation']), ...
            'mount_p', ws.getVariable([P 'HipMountOffset']), ...
            'thigh', ws.getVariable('ThighLength'), 'shank', ws.getVariable('ShankLength'));
        rows = (s - 1) * 6 + (1:6);
        for i = 1:n
            J = local_jacobian(geom, ref.pelvis_R(:, :, i), ref.pelvis_p(:, i), ref.q(rows, i), opts);
            A = J * J.' + opts.damping ^ 2 * eye(6);
            G(rows, :, i) = -J.' * (A \ [eye(d); zeros(6 - d, d)]);
        end
    end
end

function J = local_jacobian(geom, Rp, pp, q, opts)
% Central differences of the foot pose; rotation rows scaled to metres.
    h = opts.step_deg;
    J = zeros(6, 6);
    [R0, ~] = gs3dx_leg_fk(geom, Rp, pp, q);
    for a = 1:6
        dq = zeros(6, 1);
        dq(a) = h;
        [Rh, ph] = gs3dx_leg_fk(geom, Rp, pp, q + dq);
        [Rl, pl] = gs3dx_leg_fk(geom, Rp, pp, q - dq);
        rot = local_rotvec(R0.' * Rh) - local_rotvec(R0.' * Rl);
        J(:, a) = [ph - pl; opts.lever_m * (R0 * rot)] / (2 * h);
    end
end

function v = local_rotvec(R)
% Small-angle rotation vector of R (the finite-difference steps are tiny).
    v = 0.5 * [R(3, 2) - R(2, 3); R(1, 3) - R(3, 1); R(2, 1) - R(1, 2)];
end
