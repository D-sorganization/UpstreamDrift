function [r, info] = gs3dx_head_orientation_residual(R_pred, R_target, gaps, weight)
%GS3DX_HEAD_ORIENTATION_RESIDUAL  Pure head-orientation residual helper for whole-body IK (#10979, #11161).
%
%   [R, INFO] = GS3DX_HEAD_ORIENTATION_RESIDUAL(R_PRED, R_TARGET, GAPS, WEIGHT)
%   computes the 9x1 orientation residual vector comparing the model Head solid
%   rotation against calibrated target head orientation for single-frame (K=1) tracking.
%   Delegates to generic gs3dx_orientation_residual.
%
%   Inputs:
%     R_pred    (3, 3) double predicted Head solid rotation matrix
%     R_target  (3, 3) double calibrated measured target rotation matrix
%     gaps      (1, 1) logical gap flag (default false)
%     weight    (1, 1) double residual weight (default 0)
%
%   Outputs:
%     r         (9, 1) double column residual vector
%     info      struct with .valid (1x1), .err_chordal (1x1), .err_rad (1x1), .err_deg (1x1)

    arguments
        R_pred (3,3) double {mustBeReal, mustBeFinite}
        R_target (3,3) double
        gaps (1,1) logical = false
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
            error('gs3dx:head_residual:invalid_rotation', '%s', ME.message);
        end
        rethrow(ME);
    end
end
