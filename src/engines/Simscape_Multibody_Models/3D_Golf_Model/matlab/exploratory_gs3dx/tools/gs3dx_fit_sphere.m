function [center, radius, rms_err, diagnostics] = gs3dx_fit_sphere(P)
%GS3DX_FIT_SPHERE  Pure functional algebraic least-squares sphere fit.
%
%   [CENTER, RADIUS, RMS_ERR, DIAG] = GS3DX_FIT_SPHERE(P) fits a 3D sphere
%   to the columns of P (3 x N) using a centered, scaled algebraic least-squares
%   formulation with SVD rank and condition validation.
%
%   Inputs:
%     P         3 x N real finite matrix with N >= 4 non-coplanar points.
%
%   Outputs:
%     CENTER    3 x 1 sphere center [x; y; z] in same units as P.
%     RADIUS    Scalar sphere radius (> 0) in same units as P.
%     RMS_ERR   Scalar root-mean-square radial error:
%               sqrt(mean((vecnorm(P - center) - radius).^2)).
%     DIAG      Diagnostic struct with fields:
%                 .rank             numerical rank of the design matrix (must be 4)
%                 .condition        condition number of the scaled design matrix
%                 .singular_values  4 x 1 singular values of the scaled design matrix
%                 .rank_tolerance   transparent numerical tolerance used for rank
%                 .scale            mean point-to-centroid distance used for scaling
%                 .center_mean      3 x 1 centroid of input points
%                 .n_points         number of measured points (N)
%                 .rms              identical to RMS_ERR
%                 .geometric_residual 1 x N vector of signed radial errors
%                 .algebraic_residual N x 1 algebraic equation residual
%
%   Preconditions and Numerical Safeguards:
%     - Requires N >= 4 points; fails closed if N < 4.
%     - Rejects non-finite, NaN, Inf, or complex inputs.
%     - Centers coordinates at centroid and scales by mean radial distance to
%       eliminate numerical cancellation and stabilize condition number.
%     - Validates full column rank (rank == 4) using standard MATLAB tolerance
%       max(size(A))*eps(s(1)); coplanar, colinear, or repeated points fail closed.
%     - Validates positive real radius; rejects non-positive/complex radius.
%     - Never falls back to zeros, NaNs, or unvalidated algebraic approximations.

    arguments
        P (3,:) double
    end

    % Preconditions: Verify input type, dimensions, count, and finiteness
    assert(isnumeric(P) && isreal(P), 'gs3dx:fit_sphere:invalid_type', ...
        'P must be a real numeric matrix');
    assert(size(P, 1) == 3 && ndims(P) == 2, 'gs3dx:fit_sphere:invalid_dimensions', ...
        'P must be a 3xN matrix (got %dx%d)', size(P, 1), size(P, 2));
    N = size(P, 2);
    assert(N >= 4, 'gs3dx:fit_sphere:too_few_points', ...
        'At least 4 non-coplanar points required for sphere fit (got %d)', N);
    assert(all(isfinite(P), 'all'), 'gs3dx:fit_sphere:non_finite', ...
        'P must contain only finite numbers (no NaN or Inf)');

    % Centering: subtract centroid to prevent catastrophic precision loss
    mu = mean(P, 2);
    P_centered = P - mu;

    % Scaling: normalize by mean Euclidean distance from centroid
    dists = sqrt(sum(P_centered .^ 2, 1));
    scale = mean(dists);
    assert(isfinite(scale) && scale > 100 * eps(class(P)), 'gs3dx:fit_sphere:degenerate_points', ...
        'Points are identical or degenerate (scale = %e)', scale);

    P_norm = P_centered / scale;

    % Algebraic LS formulation in normalized space:
    % (x - c_norm)^2 = r_norm^2
    % 2*x*c_norm + w_norm = ||x||^2, where w_norm = r_norm^2 - ||c_norm||^2
    % Design matrix: A_norm = [2 * P_norm.', ones(N, 1)]
    % b_norm = sum(P_norm .^ 2, 1).'
    A_norm = [2 * P_norm.', ones(N, 1)];
    b_norm = sum(P_norm .^ 2, 1).';

    % SVD for rank, conditioning, and robust least-squares solution
    [U, S_mat, V] = svd(A_norm, 'econ');
    sv = diag(S_mat);

    % Transparent numerical tolerance (standard MATLAB definition max(size(A))*eps(s(1)))
    rank_tol = max(size(A_norm)) * eps(sv(1));
    A_rank = sum(sv > rank_tol);

    assert(A_rank == 4, 'gs3dx:fit_sphere:rank_deficient', ...
        'Design matrix is rank deficient (%d < 4); points are coplanar, colinear, or repeated', A_rank);

    cond_num = sv(1) / sv(4);
    assert(cond_num < 1 / eps(class(P)), 'gs3dx:fit_sphere:ill_conditioned', ...
        'Design matrix is numerically singular (condition number %e)', cond_num);

    % Solve algebraic system: x_norm = [c_norm; w_norm]
    x_norm = V * (diag(1 ./ sv) * (U.' * b_norm));
    c_norm = x_norm(1:3);
    w_norm = x_norm(4);

    r_norm_sq = w_norm + sum(c_norm .^ 2);
    assert(isreal(r_norm_sq) && isfinite(r_norm_sq) && r_norm_sq > 0, ...
        'gs3dx:fit_sphere:invalid_radius', ...
        'Fitted radius squared is non-positive or non-finite (r_norm_sq = %f)', r_norm_sq);

    r_norm = sqrt(r_norm_sq);

    % Uncenter and unscale back to original coordinate frame
    center = mu + c_norm * scale;
    radius = r_norm * scale;

    assert(all(isfinite(center)) && isreal(center), 'gs3dx:fit_sphere:invalid_center', ...
        'Fitted center is non-finite or complex');
    assert(isfinite(radius) && isreal(radius) && radius > 0, 'gs3dx:fit_sphere:invalid_radius', ...
        'Fitted radius is non-finite or non-positive');

    % Geometric residual and RMS error
    pt_dists = sqrt(sum((P - center) .^ 2, 1));
    geom_res = pt_dists - radius;
    rms_err = sqrt(mean(geom_res .^ 2));

    % Diagnostic struct
    if nargout >= 4
        diagnostics = struct();
        diagnostics.rank = A_rank;
        diagnostics.condition = cond_num;
        diagnostics.singular_values = sv;
        diagnostics.rank_tolerance = rank_tol;
        diagnostics.scale = scale;
        diagnostics.center_mean = mu;
        diagnostics.n_points = N;
        diagnostics.rms = rms_err;
        diagnostics.geometric_residual = geom_res;
        diagnostics.algebraic_residual = A_norm * x_norm - b_norm;
    end
end
