function m = gs3dx_hull_moment(p_com, points, fz)
%GS3DX_HULL_MOMENT  Minimum vertical GRF moment from COM outside support hull (#11173).
%
%   M = GS3DX_HULL_MOMENT(P_COM, POINTS, FZ) returns a 1xN vector.  P_COM is
%   3xN (centre of mass).  POINTS is 3xPxN contact positions (P >= 1).  FZ is
%   1xN total upward force (N).  For each frame,
%
%     m(k) = fz(k) * d_k
%
%   where d_k is the horizontal distance from p_com(1:2,k) to the convex hull of
%   points(1:2,:,k).  d_k is zero when the COM lies inside or on the hull
%   boundary (including on a vertex or edge).
%
%   Uses GS3DX_NEAREST_HULL_POINT for the 2D hull distance.  Error id gs3dx:hull.

    assert(size(p_com, 1) == 3, 'gs3dx:hull', 'P_COM must be 3xN');
    assert(size(points, 1) == 3, 'gs3dx:hull', 'POINTS must be 3xPxN');
    n = size(p_com, 2);
    p = size(points, 2);
    nFrames = size(points, 3);
    assert(p >= 1, 'gs3dx:hull', 'POINTS must have at least one contact per frame');
    assert(nFrames == n, 'gs3dx:hull', ...
        'P_COM column count must match POINTS third dimension');
    fz = fz(:).';
    assert(numel(fz) == n, 'gs3dx:hull', 'FZ must be 1xN matching P_COM');
    assert(all(isfinite(p_com(:))) && all(isfinite(points(:))) && all(isfinite(fz)), ...
        'gs3dx:hull', 'P_COM, POINTS and FZ must be finite');
    assert(all(fz >= 0), 'gs3dx:hull', 'FZ must be non-negative');

    m = zeros(1, n);
    for k = 1:n
        [~, d] = gs3dx_nearest_hull_point(p_com(1:2, k), points(1:2, :, k));
        m(k) = fz(k) * d;
    end

    assert(all(m >= -1e-9), 'gs3dx:hull', 'Hull moment must be non-negative');
    m = max(m, 0);

    for k = 1:n
        q = p_com(1:2, k);
        pts = points(1:2, :, k);
        for j = 1:p
            if norm(q - pts(:, j)) <= 1e-12
                assert(abs(m(k)) <= 1e-9, 'gs3dx:hull', ...
                    'Moment must be zero when COM equals a contact point');
            end
        end
    end
end
