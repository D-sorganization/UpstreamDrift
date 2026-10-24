function w = gs3dx_contact_wrench(f_required, p_com, points, mu)
%GS3DX_CONTACT_WRENCH  Per-contact forces for foot-width solve (#11173).
%
%   W = GS3DX_CONTACT_WRENCH(F_REQUIRED, P_COM, POINTS, MU) returns struct
%   fields f (3xPxN contact forces), moment (1xN), and force (1xN).  Moment
%   and force match GS3DX_SUPPORT_RESIDUAL without M_REQUIRED.  MU defaults
%   to 0.8.
%
%   Error id gs3dx:wrench.

    if nargin < 4 || isempty(mu)
        mu = 0.8;
    end

    assert(size(f_required, 1) == 3, 'gs3dx:wrench', 'F_REQUIRED must be 3xN');
    assert(size(p_com, 1) == 3, 'gs3dx:wrench', 'P_COM must be 3xN');
    assert(size(points, 1) == 3, 'gs3dx:wrench', 'POINTS must be 3xPxN');
    n = size(f_required, 2);
    assert(size(p_com, 2) == n, 'gs3dx:wrench', 'P_COM column count must match F_REQUIRED');
    p = size(points, 2);
    assert(p >= 1, 'gs3dx:wrench', 'POINTS must have at least one contact per frame');
    if ndims(points) < 3
        assert(n == 1, 'gs3dx:wrench', 'POINTS must be 3xPxN');
        points = reshape(points, 3, p, 1);
    end
    nFrames = size(points, 3);
    assert(nFrames == n, 'gs3dx:wrench', ...
        'F_REQUIRED column count must match POINTS third dimension');
    assert(isscalar(mu) && isfinite(mu) && mu >= 0, 'gs3dx:wrench', ...
        'MU must be a non-negative finite scalar');
    assert(all(isfinite(f_required(:))) && all(isfinite(p_com(:))) && ...
        all(isfinite(points(:))), 'gs3dx:wrench', ...
        'F_REQUIRED, P_COM and POINTS must be finite');

    s = gs3dx_support_residual(f_required, p_com, points, mu);

    w.moment = s.moment;
    w.force = s.force;
    w.f = zeros(3, p, n);

    totalVert = max(f_required(3, :), 0);
    cop = s.cop;
    for k = 1:n
        if totalVert(k) <= 0
            cop(:, k) = 0;
        end
    end
    fzShare = gs3dx_vertical_share(cop, points, totalVert);

    for k = 1:n
        fz = f_required(3, k);
        if fz <= 0
            continue
        end

        fxy = f_required(1:2, k);
        nxy = norm(fxy);
        fMax = mu * fz;
        if nxy <= 0
            fApplied = fxy;
        else
            fApplied = fxy * min(1, fMax / nxy);
        end

        fzK = fzShare(:, k);
        for i = 1:p
            w.f(1:2, i, k) = fApplied * (fzK(i) / fz);
            w.f(3, i, k) = fzK(i);
        end

        fSum = sum(w.f(:, :, k), 2);
        fTarget = [fApplied; max(fz, 0)];
        assert(norm(fSum - fTarget) <= 1e-6, 'gs3dx:wrench', ...
            'Contact forces must sum to applied wrench');

        comK = p_com(:, k);
        zGround = mean(points(3, :, k));
        mVec = zeros(3, 1);
        for i = 1:p
            ri = points(:, i, k) - comK;
            rGround = [points(1:2, i, k); zGround] - comK;
            fi = w.f(:, i, k);
            mVec = mVec + cross(rGround, [fi(1:2); 0]) + cross(ri, [0; 0; fi(3)]);
        end
        mTip = hypot(mVec(1), mVec(2));
        assert(abs(mTip - w.moment(k)) <= 1e-4, 'gs3dx:wrench', ...
            'Contact tip moment must match support residual moment');
    end

    assert(all(w.f(3, :, :) >= -1e-9, 'all'), 'gs3dx:wrench', ...
        'Contact vertical forces must be non-negative');
end
