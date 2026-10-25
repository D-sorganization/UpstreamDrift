function [h, hdot] = gs3dx_point_momentum(mass, position, dt)
%GS3DX_POINT_MOMENTUM  Angular momentum of point masses about the COM (#11173).
%
%   [H, HDOT] = GS3DX_POINT_MOMENTUM(MASS, POSITION, DT) returns H and HDOT
%   (3xN, kg*m^2/s and kg*m^2/s^2).  MASS is Px1 (kg).  POSITION is 3xPxN (m).
%   DT is a positive scalar (s).  Point masses only; segment spin is omitted.
%
%   H(:, k) = sum_i mass_i * cross(r_i - com, v_i - v_com) with com and
%   velocities from POSITION and gradient(..., DT) along the frame axis.
%   HDOT is gradient(H, DT).  A single particle has H identically zero.
%
%   Error id gs3dx:momentum.

    assert(isvector(mass) && numel(mass) >= 1, 'gs3dx:momentum', 'MASS must be Px1');
    mass = mass(:);
    P = numel(mass);
    assert(all(mass > 0) && all(isfinite(mass)), 'gs3dx:momentum', 'MASS must be positive and finite');
    assert(size(position, 1) == 3 && size(position, 2) == P, 'gs3dx:momentum', ...
        'POSITION must be 3xPxN matching MASS');
    n = size(position, 3);
    assert(n >= 1, 'gs3dx:momentum', 'POSITION must have at least one frame');
    assert(isscalar(dt) && isfinite(dt) && dt > 0, 'gs3dx:momentum', 'DT must be a positive finite scalar');
    assert(all(isfinite(position(:))), 'gs3dx:momentum', 'POSITION must be finite');

    Mtot = sum(mass);
    com = zeros(3, n);
    for i = 1:P
        ri = squeeze(position(:, i, :));
        if size(ri, 2) ~= n
            ri = reshape(position(:, i, :), 3, n);
        end
        com = com + mass(i) * ri;
    end
    com = com / Mtot;

    vel = zeros(3, P, n);
    for i = 1:P
        ri = reshape(position(:, i, :), 3, n);
        vel(:, i, :) = gradient(ri, dt, 2);
    end
    vcom = gradient(com, dt, 2);

    h = zeros(3, n);
    for k = 1:n
        hk = zeros(3, 1);
        for i = 1:P
            ri = position(:, i, k) - com(:, k);
            vi = squeeze(vel(:, i, k)) - vcom(:, k);
            hk = hk + mass(i) * cross(ri, vi);
        end
        h(:, k) = hk;
    end
    hdot = gradient(h, dt, 2);

    if P == 1
        assert(all(abs(h(:)) <= 1e-12), 'gs3dx:momentum', ...
            'Single-particle angular momentum must be zero');
    end
end
