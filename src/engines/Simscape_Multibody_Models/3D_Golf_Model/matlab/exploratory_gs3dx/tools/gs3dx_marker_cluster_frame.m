function [F, valid] = gs3dx_marker_cluster_frame(origin, axis_marker, plane_marker)
%GS3DX_MARKER_CLUSTER_FRAME  Pure rigid 3-marker cluster coordinate frame (#10979).
%
%   [F, VALID] = GS3DX_MARKER_CLUSTER_FRAME(ORIGIN, AXIS_MARKER, PLANE_MARKER)
%   constructs an orthonormal coordinate frame F in SO(3) for each sample k in
%   1..N from three non-collinear tracking markers (e.g. HeadTop, HeadFront,
%   HeadSide).
%
%   Axes Definition (Pure Marker-Cluster Frame, NOT Anatomical):
%     v1 = axis_marker - origin
%     v2 = plane_marker - origin
%     x = normalize(v1)
%     y = normalize(v2 - x * dot(x, v2))
%     z = cross(x, y)
%     F = [x, y, z]   (3x3 orthonormal rotation matrix in SO(3))
%
%   Biomechanical Scope & Boundary:
%     The axes [x, y, z] represent the raw rigid triad of the tracking marker
%     cluster, NOT measured anatomical head/body axes. Physical anatomical
%     orientation requires capture-specific body-frame calibration.
%
%   Fail-Closed Nonfinite & Relative Degeneracy Guards:
%     - Nonfinite samples (NaN, Inf) are marked missing: valid=false, F=NaN(3,3).
%     - Coincident or collinear/near-collinear samples are flagged invalid
%       (valid=false, F=NaN(3,3)) rather than fabricating an orientation.
%     - Guards use scale-relative thresholds (TOL_REL = 1e-6) without absolute
%       eps cutoffs; differences are scaled by their common max component to
%       avoid norm overflow/underflow across finite representable magnitudes.
%     - If direct differences overflow despite finite coordinates, points are
%       normalized by their common max-abs prior to differencing.
%     - Strong SO(3) postconditions enforce det(F)=1 and orthogonality.
%
%   Floating-Point Representability Limits:
%     Scale invariance holds across finite representable floating-point magnitudes
%     (~1e-300 to ~1e300). Exact invariance does not hold for arbitrary large
%     translations of tiny clusters, as IEEE-754 mantissa cancellation loses
%     relative precision when cluster dimensions fall below eps(translation).
%
%   Inputs:
%     origin        (3, N) real numeric coordinates (m)
%     axis_marker   (3, N) real numeric coordinates (m)
%     plane_marker  (3, N) real numeric coordinates (m)
%
%   Outputs:
%     F             (3, 3, N) double SO(3) frame matrices (NaN when invalid)
%     valid         (1, N) logical validity mask

    TOL_REL = 1e-6;

    if ~isnumeric(origin) || ~isnumeric(axis_marker) || ~isnumeric(plane_marker) || ...
       ~isreal(origin) || ~isreal(axis_marker) || ~isreal(plane_marker)
        error('gs3dx:marker_cluster_frame:invalid_input', ...
            'origin, axis_marker, and plane_marker must be real numeric arrays');
    end

    if ndims(origin) ~= 2 || size(origin, 1) ~= 3 || ...
       ndims(axis_marker) ~= 2 || size(axis_marker, 1) ~= 3 || ...
       ndims(plane_marker) ~= 2 || size(plane_marker, 1) ~= 3
        error('gs3dx:marker_cluster_frame:invalid_dimensions', ...
            'origin, axis_marker, and plane_marker must have size (3, N)');
    end

    N = size(origin, 2);
    if size(axis_marker, 2) ~= N || size(plane_marker, 2) ~= N
        error('gs3dx:marker_cluster_frame:mismatched_frames', ...
            'origin (3x%d), axis_marker (3x%d), and plane_marker (3x%d) frame counts must match', ...
            N, size(axis_marker, 2), size(plane_marker, 2));
    end

    if N < 1
        error('gs3dx:marker_cluster_frame:empty_input', 'Sample count N must be >= 1');
    end

    F = nan(3, 3, N);
    valid = false(1, N);

    for k = 1:N
        p0 = double(origin(:, k));
        p1 = double(axis_marker(:, k));
        p2 = double(plane_marker(:, k));

        if any(~isfinite(p0)) || any(~isfinite(p1)) || any(~isfinite(p2))
            continue;
        end

        v1 = p1 - p0;
        v2 = p2 - p0;
        v12 = p1 - p2;

        if any(~isfinite(v1)) || any(~isfinite(v2)) || any(~isfinite(v12))
            pts_max = max(abs([p0; p1; p2]));
            if ~isfinite(pts_max) || pts_max == 0
                continue;
            end
            p0 = p0 / pts_max; p1 = p1 / pts_max; p2 = p2 / pts_max;
            v1 = p1 - p0; v2 = p2 - p0; v12 = p1 - p2;
            if any(~isfinite(v1)) || any(~isfinite(v2)) || any(~isfinite(v12))
                continue;
            end
        end

        mv = max(abs([v1; v2; v12]));
        if ~isfinite(mv) || mv <= 0
            continue;
        end

        u1 = v1 / mv;
        u2 = v2 / mv;
        u12 = v12 / mv;

        n1 = norm(u1);
        n2 = norm(u2);
        n12 = norm(u12);
        L = max(n1, n2);

        if L <= 0 || (min([n1, n2, n12]) / L) < TOL_REL
            continue;
        end

        x = u1 / n1;
        v2_orth = u2 - (x' * u2) * x;
        n2_orth = norm(v2_orth);

        if (n2_orth / n2) < TOL_REL
            continue;
        end

        y = v2_orth / n2_orth;
        z = cross(x, y);
        nz = norm(z);
        if nz <= 0 || ~isfinite(nz)
            continue;
        end
        z = z / nz;
        y = cross(z, x);
        Rk = [x, y, z];

        if any(~isfinite(Rk), 'all') || ...
           abs(det(Rk) - 1.0) > 1e-11 || ...
           norm(Rk' * Rk - eye(3), 'fro') > 1e-11
            continue;
        end

        F(:, :, k) = Rk;
        valid(k) = true;
    end
end
