function shift = gs3dx_com_shift(f_required, p_com, points, mu)
%GS3DX_COM_SHIFT  Smallest horizontal COM shift to land COP on support hull (#11173).
%
%   SHIFT = GS3DX_COM_SHIFT(F_REQUIRED, P_COM, POINTS, MU) returns 3xN.
%   Rows 1 and 2 are the horizontal correction (m); row 3 is zero.  MU defaults
%   to 0.8.  Required COP comes from GS3DX_SUPPORT_RESIDUAL.  When COP is NaN
%   or already inside the hull, that column of SHIFT is zero; otherwise the
%   column is [nearest_xy - cop_xy; 0].
%
%   Error id gs3dx:shift.  Input preconditions match GS3DX_SUPPORT_RESIDUAL.

    if nargin < 4 || isempty(mu)
        mu = 0.8;
    end

    s = gs3dx_support_residual(f_required, p_com, points, mu);
    n = size(f_required, 2);
    shift = zeros(3, n);

    assert(size(points, 1) == 3, 'gs3dx:shift', 'POINTS must be 3xPxN');
    p = size(points, 2);
    if ndims(points) < 3
        points = reshape(points, 3, p, n);
    end

    for k = 1:n
        if f_required(3, k) <= 0
            continue
        end
        cop = s.cop(:, k);
        if any(isnan(cop))
            continue
        end
        qNear = gs3dx_nearest_hull_point(cop, points(1:2, :, k));
        delta = qNear - cop;
        if norm(delta) <= 1e-12
            continue
        end
        shift(1:2, k) = delta;
    end

    s2 = gs3dx_support_residual(f_required, p_com + shift, points, mu);
    for k = 1:n
        fz = f_required(3, k);
        if fz > 0
            assert(s2.moment(k) <= 1e-6, 'gs3dx:shift', ...
                'Corrected tip moment must be negligible');
            if s.moment(k) > 1e-12
                want = s.moment(k) / fz;
                assert(abs(norm(shift(1:2, k)) - want) <= 1e-6, 'gs3dx:shift', ...
                    'Shift magnitude must match moment/Fz');
            else
                assert(norm(shift(1:2, k)) <= 1e-9, 'gs3dx:shift', ...
                    'Shift must be zero when uncorrected moment is zero');
            end
        end
    end
end
