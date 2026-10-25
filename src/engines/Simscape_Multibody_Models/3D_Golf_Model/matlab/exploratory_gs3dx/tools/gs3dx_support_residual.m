function s = gs3dx_support_residual(f_required, p_com, points, mu, m_required)
%GS3DX_SUPPORT_RESIDUAL  Shear centre-of-pressure tip residual (#11173).
%
%   S = GS3DX_SUPPORT_RESIDUAL(F_REQUIRED, P_COM, POINTS, MU, M_REQUIRED)
%   returns a struct with fields moment (1xN), force (1xN), and cop (2xN).
%   F_REQUIRED and P_COM are 3xN.  POINTS is 3xPxN contact positions (P >= 1).
%   MU defaults to 0.8 (Coulomb friction limit on horizontal GRF).  M_REQUIRED
%   is 3xN (N*m): the ground may supply this moment about the COM; default
%   is zero (legacy zero-moment COP).  The yaw component M_REQUIRED(3,:) is
%   unused.
%
%   Required COP (horizontal):
%
%     cop_xy = com_xy + (z_ground - com_z) * f_applied_xy / Fz + [-My; Mx] / Fz
%
%   where z_ground is the mean contact z that frame, Fz = f_required(3), and
%   f_applied_xy is f_required(1:2) scaled to norm at most mu*max(Fz, 0).
%   Tip residual moment uses GS3DX_HULL_MOMENT on cop projected into the hull.
%
%   Error id gs3dx:support.

    if nargin < 4 || isempty(mu)
        mu = 0.8;
    end

    assert(size(f_required, 1) == 3, 'gs3dx:support', 'F_REQUIRED must be 3xN');
    assert(size(p_com, 1) == 3, 'gs3dx:support', 'P_COM must be 3xN');
    assert(size(points, 1) == 3, 'gs3dx:support', 'POINTS must be 3xPxN');
    n = size(f_required, 2);
    assert(size(p_com, 2) == n, 'gs3dx:support', 'P_COM column count must match F_REQUIRED');
    p = size(points, 2);
    assert(p >= 1, 'gs3dx:support', 'POINTS must have at least one contact per frame');
    if ndims(points) < 3
        assert(n == 1, 'gs3dx:support', 'POINTS must be 3xPxN');
        points = reshape(points, 3, p, 1);
    end
    nFrames = size(points, 3);
    assert(nFrames == n, 'gs3dx:support', ...
        'F_REQUIRED column count must match POINTS third dimension');
    assert(isscalar(mu) && isfinite(mu) && mu >= 0, 'gs3dx:support', ...
        'MU must be a non-negative finite scalar');
    if nargin < 5 || isempty(m_required)
        m_required = zeros(3, n);
    end
    assert(isequal(size(m_required), [3, n]), 'gs3dx:support', ...
        'M_REQUIRED must be 3xN matching F_REQUIRED');
    assert(all(isfinite(f_required(:))) && all(isfinite(p_com(:))) && ...
        all(isfinite(points(:))) && all(isfinite(m_required(:))), 'gs3dx:support', ...
        'F_REQUIRED, P_COM, POINTS and M_REQUIRED must be finite');

    s.moment = zeros(1, n);
    s.force = zeros(1, n);
    s.cop = nan(2, n);

    for k = 1:n
        fz = f_required(3, k);
        fxy = f_required(1:2, k);
        mk = m_required(:, k);
        zGround = mean(points(3, :, k));
        comZ = p_com(3, k);
        comXy = p_com(1:2, k);

        if fz <= 0
            s.force(k) = norm(f_required(:, k));
            if norm(mk) <= 0
                s.moment(k) = 0;
            else
                s.moment(k) = norm(mk);
            end
            continue
        end

        fMax = mu * max(fz, 0);
        nxy = norm(fxy);
        if nxy <= fMax || nxy <= 0
            fApplied = fxy;
        else
            fApplied = fxy * (fMax / nxy);
        end

        mTip = [-mk(2); mk(1)] / fz;
        s.cop(:, k) = comXy + (zGround - comZ) * fApplied / fz + mTip;
        s.force(k) = norm(fxy - fApplied);

        cop3 = [s.cop(:, k); 0];
        ptsK = points(:, :, k);
        s.moment(k) = gs3dx_hull_moment(cop3, ptsK, fz);
    end

    assert(all(isfinite(s.moment)) && all(isfinite(s.force)), ...
        'gs3dx:support', 'Moment and force must be finite');
    assert(all(s.moment >= -1e-9) && all(s.force >= -1e-9), ...
        'gs3dx:support', 'Moment and force must be non-negative');
    s.moment = max(s.moment, 0);
    s.force = max(s.force, 0);
end
