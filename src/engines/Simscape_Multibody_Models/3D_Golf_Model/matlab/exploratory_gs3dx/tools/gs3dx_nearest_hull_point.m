function [qNear, d] = gs3dx_nearest_hull_point(q, pts)
%GS3DX_NEAREST_HULL_POINT  Closest point on the 2D support hull (#11173).
%
%   [Q_NEAR, D] = GS3DX_NEAREST_HULL_POINT(Q, PTS) returns the closest point
%   Q_NEAR (2x1) on the convex hull of PTS (2xP) to query Q (2x1).  D is the
%   horizontal distance norm(Q - Q_NEAR).  When Q lies inside or on the hull,
%   Q_NEAR = Q and D = 0.
%
%   Error id gs3dx:hull.

    assert(size(q, 1) == 2 && size(q, 2) == 1, 'gs3dx:hull', 'Q must be 2x1');
    assert(size(pts, 1) == 2, 'gs3dx:hull', 'PTS must be 2xP');
    assert(all(isfinite(q)) && all(isfinite(pts(:))), 'gs3dx:hull', ...
        'Q and PTS must be finite');

    pts = pts(:, all(isfinite(pts), 1));
    if isempty(pts)
        error('gs3dx:hull', 'Contact points must be finite');
    end
    if size(pts, 2) == 1
        qNear = pts(:, 1);
        d = norm(q - qNear);
        return
    end
    if size(pts, 2) == 2
        [qNear, d] = local_segment_nearest(q, pts(:, 1), pts(:, 2));
        return
    end

    k = convhull(pts(1, :), pts(2, :));
    hx = pts(1, k);
    hy = pts(2, k);
    if inpolygon(q(1), q(2), hx, hy)
        qNear = q;
        d = 0;
        return
    end

    d = inf;
    qNear = q;
    for i = 1:numel(k) - 1
        a = [hx(i); hy(i)];
        b = [hx(i + 1); hy(i + 1)];
        [proj, di] = local_segment_nearest(q, a, b);
        if di < d
            d = di;
            qNear = proj;
        end
    end
end

function [proj, d] = local_segment_nearest(q, a, b)
    ab = b - a;
    len2 = ab.' * ab;
    if len2 <= 1e-24
        proj = a;
        d = norm(q - a);
        return
    end
    t = ((q - a).' * ab) / len2;
    t = min(max(t, 0), 1);
    proj = a + t * ab;
    d = norm(q - proj);
end
