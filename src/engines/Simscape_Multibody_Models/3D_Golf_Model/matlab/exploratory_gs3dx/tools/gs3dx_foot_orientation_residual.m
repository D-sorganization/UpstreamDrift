function [r, info] = gs3dx_foot_orientation_residual(R_pred, R_target, gaps, weight)
%GS3DX_FOOT_ORIENTATION_RESIDUAL  Pure foot-orientation residual helper for whole-body IK (#10979, #11161).
%
%   [R, INFO] = GS3DX_FOOT_ORIENTATION_RESIDUAL(R_PRED, R_TARGET, GAPS, WEIGHT)
%   computes the 18x1 orientation residual vector comparing the model Foot solid
%   rotations against calibrated target foot orientations for Left (k=1) and Right (k=2) feet.
%   Delegates to generic gs3dx_orientation_residual for K=2 frames.
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

    try
        if nargout > 1
            [r, info] = gs3dx_orientation_residual(R_pred, R_target, gaps, weight);
        else
            r = gs3dx_orientation_residual(R_pred, R_target, gaps, weight);
        end
    catch ME
        if strcmp(ME.identifier, 'gs3dx:orientation_residual:invalid_rotation')
            error('gs3dx:foot_residual:invalid_rotation', '%s', ME.message);
        end
        rethrow(ME);
    end
end
