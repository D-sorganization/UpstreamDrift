function fz = gs3dx_vertical_share(cop, points, total)
%GS3DX_VERTICAL_SHARE  Split vertical GRF across foot contacts (#11173).
%
%   FZ = GS3DX_VERTICAL_SHARE(COP, POINTS, TOTAL) returns PxN nonnegative
%   vertical forces (N) that sum to TOTAL on each frame.  COP is 2xN.
%   POINTS is 2xPxN or 3xPxN (z ignored).  TOTAL is 1xN upward force, >= 0.
%   The force-weighted xy centroid of POINTS equals GS3DX_NEAREST_HULL_POINT(COP)
%   when TOTAL > 0; when TOTAL == 0, FZ is zero.
%
%   Error id gs3dx:share.

    assert(size(cop, 1) == 2, 'gs3dx:share', 'COP must be 2xN');
    n = size(cop, 2);
    assert(n >= 1, 'gs3dx:share', 'COP must have at least one frame');
    assert(size(points, 1) == 2 || size(points, 1) == 3, 'gs3dx:share', ...
        'POINTS must be 2xPxN or 3xPxN');
    p = size(points, 2);
    assert(p >= 1, 'gs3dx:share', 'POINTS must have at least one contact');
    if ndims(points) < 3
        points = reshape(points, size(points, 1), p, 1);
    end
    nFrames = size(points, 3);
    assert(nFrames == n, 'gs3dx:share', 'COP column count must match POINTS third dimension');
    total = total(:).';
    assert(numel(total) == n, 'gs3dx:share', 'TOTAL must be 1xN matching COP');
    assert(all(isfinite(cop(:))) && all(isfinite(points(:))) && all(isfinite(total)), ...
        'gs3dx:share', 'COP, POINTS and TOTAL must be finite');
    assert(all(total >= 0), 'gs3dx:share', 'TOTAL must be non-negative');

    fz = zeros(p, n);
    for k = 1:n
        pts2 = points(1:2, :, k);
        tot = total(k);
        if tot <= 0
            continue
        end
        [qNear, ~] = gs3dx_nearest_hull_point(cop(:, k), pts2);
        w = local_hull_weights(cop(:, k), qNear, pts2);
        fz(:, k) = tot * w;
    end

    assert(all(fz(:) >= -1e-12), 'gs3dx:share', 'Vertical share must be non-negative');
    assert(max(abs(sum(fz, 1) - total), [], 'all') <= 1e-8, 'gs3dx:share', ...
        'Vertical share must sum to TOTAL');

    for k = 1:n
        if total(k) <= 0
            continue
        end
        pts2 = points(1:2, :, k);
        [qNear, ~] = gs3dx_nearest_hull_point(cop(:, k), pts2);
        c = pts2 * fz(:, k);
        if sum(fz(:, k)) > 0
            c = c / sum(fz(:, k));
        end
        assert(norm(c - qNear) <= 1e-6, 'gs3dx:share', ...
            'Force centroid must match nearest hull point');
    end
end

function w = local_hull_weights(cop, qNear, pts)
    p = size(pts, 2);
    w = zeros(p, 1);

    if p == 1
        w(1) = 1;
        return
    end
    if p == 2
        w = local_segment_weights(qNear, pts(:, 1), pts(:, 2));
        return
    end

    k = convhull(pts(1, :), pts(2, :));
    hx = pts(1, k);
    hy = pts(2, k);
    if inpolygon(cop(1), cop(2), hx, hy)
        q = cop;
        for i = 2:numel(k) - 2
            ia = k(1);
            ib = k(i);
            ic = k(i + 1);
            wb = local_triangle_bary(q, pts(:, ia), pts(:, ib), pts(:, ic));
            if min(wb) >= -1e-12
                w(ia) = max(wb(1), 0);
                w(ib) = max(wb(2), 0);
                w(ic) = max(wb(3), 0);
                s = sum(w);
                if s > 0
                    w = w / s;
                end
                return
            end
        end
        error('gs3dx:share', 'COP inside hull but no containing triangle found');
    end

    for j = 1:p
        if norm(qNear - pts(:, j)) <= 1e-12
            w(j) = 1;
            return
        end
    end

    d = inf;
    bestIa = k(1);
    bestIb = k(2);
    bestT = 0;
    for i = 1:numel(k) - 1
        ia = k(i);
        ib = k(i + 1);
        a = pts(:, ia);
        b = pts(:, ib);
        [~, di, t] = local_segment_nearest(qNear, a, b);
        if di < d
            d = di;
            bestIa = ia;
            bestIb = ib;
            bestT = t;
        end
    end

    w(bestIa) = 1 - bestT;
    w(bestIb) = bestT;
    s = sum(w);
    if s > 0
        w = w / s;
    end
end

function w = local_segment_weights(q, a, b)
    ab = b - a;
    len2 = ab.' * ab;
    if len2 <= 1e-24
        w = [1; 0];
        return
    end
    t = ((q - a).' * ab) / len2;
    t = min(max(t, 0), 1);
    w = [1 - t; t];
end

function wb = local_triangle_bary(q, a, b, c)
    v0 = c - a;
    v1 = b - a;
    v2 = q - a;
    d00 = v0.' * v0;
    d01 = v0.' * v1;
    d11 = v1.' * v1;
    d20 = v0.' * v2;
    d21 = v1.' * v2;
    den = d00 * d11 - d01 * d01;
    if abs(den) <= 1e-24
        wb = [1; 0; 0];
        return
    end
    v = (d11 * d20 - d01 * d21) / den;
    w = (d00 * d21 - d01 * d20) / den;
    u = 1 - v - w;
    wb = [u; w; v];
end

function [proj, d, t] = local_segment_nearest(q, a, b)
    ab = b - a;
    len2 = ab.' * ab;
    if len2 <= 1e-24
        proj = a;
        d = norm(q - a);
        t = 0;
        return
    end
    t = ((q - a).' * ab) / len2;
    t = min(max(t, 0), 1);
    proj = a + t * ab;
    d = norm(q - proj);
end
