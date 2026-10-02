function [r, info] = gs3dx_foot_orientation_residual(R_pred, R_target, gaps, weight)
%GS3DX_FOOT_ORIENTATION_RESIDUAL  Pure foot-orientation residual helper for whole-body IK (#10979, #11161).
%
%   [R, INFO] = GS3DX_FOOT_ORIENTATION_RESIDUAL(R_PRED, R_TARGET, GAPS, WEIGHT)
%   computes the 18x1 orientation residual vector comparing the model Foot solid
%   rotations against calibrated target foot orientations for Left (k=1) and Right (k=2) feet.
%
%   Scientific Rationale:
%     A 1D direction vector alone cannot constrain the foot's roll degree of freedom
%     (a shoe sole rolled upside-down shares the same forward axis).  This helper
%     evaluates the normalized chordal distance on SO(3), returning 9 residual entries
%     per foot (18 total):
%
%       E_k = (R_pred(:, :, k) - R_target(:, :, k)) / sqrt(2)
%       r((k-1)*9 + (1:9)) = weight * E_k(:)
%
%     The Frobenius norm ||E_k||_F is the normalized chordal distance on SO(3):
%       ||E_k||_F = 2 * sin(theta_k / 2)
%       - At theta = 0 deg:   ||E_k||_F = 0
%       - At theta = 90 deg:  ||E_k||_F = sqrt(2) ~= 1.414
%       - At theta = 180 deg: ||E_k||_F = 2.0  (||r_k|| = 2 * weight)
%     This identically penalizes 180-deg reversals in yaw (backward foot),
%     roll (sole upside-down), and pitch.
%
%   Validation & Fail-Closed Guardrails:
%     1. Real finite SO(3) rotations: R_pred must be real, finite, and strictly
%        SO(3) (orthonormal with det = +1 within 1e-8 tolerance)
%        (gs3dx:foot_residual:invalid_rotation).
%     2. Unless explicit gap=true, R_target must also be real, finite, and strictly
%        SO(3). No fake axes or fabricated measurements.
%     3. Explicit gap masking: Only explicit gaps(k)=true masks missing measurements,
%        returning zero residual, valid=false, and NaN error metrics.
%     4. Weight validation: weight must be real, finite, and non-negative.
%     5. Angular error: computed directly from trace with roundoff clipping to [-1, 1].
%
%   Inputs:
%     R_pred    (3, 3, 2) double predicted Foot solid rotation matrices
%     R_target  (3, 3, 2) double calibrated measured target rotation matrices
%     gaps      (1, 2) logical gap flags [gap_L, gap_R] (default [false, false])
%     weight    (1, 1) double residual weight (default 0)
%
%   Outputs:
%     r         (18, 1) double column residual vector
%     info      struct with .valid (1x2), .err_chordal (1x2), .err_rad (1x2), .err_deg (1x2)

    arguments
        R_pred (3,3,2) double {mustBeReal, mustBeFinite}
        R_target (3,3,2) double
        gaps (1,2) logical = [false, false]
        weight (1,1) double {mustBeReal, mustBeFinite, mustBeNonnegative} = 0
    end

    r = zeros(18, 1);
    err_chordal = nan(1, 2);
    err_rad = nan(1, 2);
    err_deg = nan(1, 2);
    valid = false(1, 2);

    tol_so3 = 1e-8;

    for k = 1:2
        % Validate predicted model rotation
        Rp = R_pred(:, :, k);
        if norm(Rp' * Rp - eye(3), 'fro') >= tol_so3 || abs(det(Rp) - 1.0) >= tol_so3
            error('gs3dx:foot_residual:invalid_rotation', ...
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
                error('gs3dx:foot_residual:invalid_rotation', ...
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
