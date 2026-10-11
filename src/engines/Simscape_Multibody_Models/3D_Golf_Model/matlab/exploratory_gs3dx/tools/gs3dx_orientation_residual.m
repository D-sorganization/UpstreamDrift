function [r, info] = gs3dx_orientation_residual(R_pred, R_target, gaps, weight)
%GS3DX_ORIENTATION_RESIDUAL  Generic SO(3) chordal residual helper for kinematic tracking.
%
%   [R, INFO] = GS3DX_ORIENTATION_RESIDUAL(R_PRED, R_TARGET, GAPS, WEIGHT)
%   computes the 9*K column residual vector comparing predicted SO(3) rotations
%   against target orientations for K >= 1 frames.
%
%   Mathematical Formulation:
%     E_k = (R_pred(:, :, k) - R_target(:, :, k)) / sqrt(2)
%     r((k-1)*9 + (1:9)) = weight * E_k(:)
%     err_chordal(k) = norm(E_k, 'fro') = 2 * sin(theta_k / 2)
%     theta_k = acos(clamp((trace(R_target(:,:,k)' * R_pred(:,:,k)) - 1) / 2, -1, 1))
%
%   Inputs:
%     R_pred    (3, 3, K) double predicted rotation matrices (real, finite, proper SO(3))
%     R_target  (3, 3, K) double target rotation matrices (proper SO(3) unless gap)
%     gaps      (1, K) logical gap flags (default false(1, K))
%     weight    (1, 1) double residual weight (default 0)
%
%   Outputs:
%     r         (9*K, 1) double column residual vector
%     info      struct with .valid (1xK), .err_chordal (1xK), .err_rad (1xK), .err_deg (1xK)

    arguments
        R_pred double {mustBeReal, mustBeFinite}
        R_target double
        gaps = []
        weight (1,1) double {mustBeReal, mustBeFinite, mustBeNonnegative} = 0
    end

    if size(R_pred, 1) ~= 3 || size(R_pred, 2) ~= 3 || ndims(R_pred) > 3 || size(R_pred, 3) < 1
        error('gs3dx:orientation_residual:invalid_dimensions', ...
            'R_pred must be a 3x3xK tensor with K >= 1.');
    end
    K = size(R_pred, 3);

    if size(R_target, 1) ~= 3 || size(R_target, 2) ~= 3 || size(R_target, 3) ~= K || ndims(R_target) > 3
        error('gs3dx:orientation_residual:invalid_dimensions', ...
            'R_target must be 3x3xK matching R_pred dimensions.');
    end

    if isempty(gaps)
        gaps = false(1, K);
    elseif ~islogical(gaps) || ~isequal(size(gaps), [1, K])
        error('gs3dx:orientation_residual:invalid_dimensions', ...
            'gaps must be a 1xK logical row vector.');
    end

    r = zeros(9 * K, 1);
    err_chordal = nan(1, K);
    err_rad = nan(1, K);
    err_deg = nan(1, K);
    valid = false(1, K);

    tol_so3 = 1e-8;

    for k = 1:K
        % Validate predicted rotation (must be proper SO(3) EVEN IF gap is true)
        Rp = R_pred(:, :, k);
        if norm(Rp' * Rp - eye(3), 'fro') >= tol_so3 || abs(det(Rp) - 1.0) >= tol_so3
            error('gs3dx:orientation_residual:invalid_rotation', ...
                'R_pred(:,:,%d) is not a valid SO(3) rotation matrix', k);
        end

        if gaps(k)
            % Explicit gap: mask measurement, report zero residual and NaN metrics
            valid(k) = false;
            err_chordal(k) = NaN;
            err_rad(k) = NaN;
            err_deg(k) = NaN;
            r((k - 1) * 9 + (1:9)) = zeros(9, 1);
        else
            % Gap is false: target MUST be real, finite, and strictly SO(3)
            Rt = R_target(:, :, k);
            if ~isreal(Rt) || ~all(isfinite(Rt), 'all') || ...
               norm(Rt' * Rt - eye(3), 'fro') >= tol_so3 || abs(det(Rt) - 1.0) >= tol_so3
                error('gs3dx:orientation_residual:invalid_rotation', ...
                    'R_target(:,:,%d) is not a valid SO(3) rotation matrix with gap=false', k);
            end

            % Normalized chordal error matrix
            E = (Rp - Rt) / sqrt(2);
            err_chordal(k) = norm(E, 'fro');

            % Angular error from relative rotation trace with roundoff clip
            R_rel = Rt' * Rp;
            cos_theta = max(-1.0, min(1.0, (trace(R_rel) - 1.0) / 2.0));
            theta_rad = acos(cos_theta);
            err_rad(k) = theta_rad;
            err_deg(k) = theta_rad * 180 / pi;
            valid(k) = true;

            if weight > 0
                r((k - 1) * 9 + (1:9)) = weight * E(:);
            end
        end
    end

    if nargout > 1
        info = struct( ...
            'valid', valid, ...
            'err_chordal', err_chordal, ...
            'err_rad', err_rad, ...
            'err_deg', err_deg ...
        );
    end
end
