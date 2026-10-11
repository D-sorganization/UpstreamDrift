function [R_target, gaps_out, meta] = gs3dx_head_target_orientation(F, address_frame, R_model_head_address, cluster_valid)
%GS3DX_HEAD_TARGET_ORIENTATION  Calibrated head target orientation from cluster frames (#10979, #11161).
%
%   [R_TARGET, GAPS_OUT, META] = GS3DX_HEAD_TARGET_ORIENTATION(F, ADDRESS_FRAME, R_MODEL_HEAD_ADDRESS, CLUSTER_VALID)
%   computes calibrated target rotation matrices for the head solid in SO(3):
%
%       R_target(f) = F(f) * F(address)' * R_model_head_address
%
%   Biomechanical Scope & Boundary:
%     The marker cluster frame F is a pure rigid triad of the tracking markers,
%     NOT an anatomical head frame. Alignment to the model address pose is
%     an explicit kinematic assumption; anatomical orientation requires an
%     independent body-frame calibration. At address, the target retains
%     the assumed R_model_head_address orientation.
%
%   Fail-Closed Nonfinite & Calibration Guards:
%     - address_frame must be an integer index within 1..N.
%     - F(:,:,address_frame) and R_model_head_address must be valid, finite,
%       strictly proper SO(3) rotations (tol 1e-8) ('gs3dx:head_orientation:invalid_calibration').
%     - Missing or degenerate frames are marked gap=true with R_target=NaN(3,3).
%
%   Inputs:
%     F                     (3, 3, N) double marker cluster frames in SO(3)
%     address_frame         (1, 1) double integer frame index
%     R_model_head_address  (3, 3) double model head rotation matrix at address
%     cluster_valid         (1, N) logical validity mask (optional)
%
%   Outputs:
%     R_target              (3, 3, N) double target rotation matrices in SO(3)
%     gaps_out              (1, N) logical gap mask
%     meta                  struct with calibration metadata

    arguments
        F (3, 3, :) double
        address_frame (1, 1) double
        R_model_head_address (3, 3) double
        cluster_valid logical = []
    end

    tol_so3 = 1e-8;
    N = size(F, 3);

    if address_frame < 1 || address_frame > N || address_frame ~= round(address_frame) || ~isfinite(address_frame)
        error('gs3dx:head_orientation:invalid_address', ...
            'address_frame must be a finite integer index within 1..N (1..%d)', N);
    end

    if isempty(cluster_valid)
        cluster_valid = true(1, N);
    else
        cluster_valid = cluster_valid(:).';
        if numel(cluster_valid) ~= N
            error('gs3dx:head_orientation:invalid_dimension', ...
                'cluster_valid length (%d) must match frame count (%d)', numel(cluster_valid), N);
        end
    end

    % Validate R_model_head_address
    if ~isreal(R_model_head_address) || any(~isfinite(R_model_head_address), 'all') || ...
       norm(R_model_head_address' * R_model_head_address - eye(3), 'fro') >= tol_so3 || ...
       abs(det(R_model_head_address) - 1.0) >= tol_so3
        error('gs3dx:head_orientation:invalid_calibration', ...
            'R_model_head_address is not a valid SO(3) rotation matrix');
    end

    % Validate F at address_frame
    F_addr = F(:, :, address_frame);
    if ~cluster_valid(address_frame) || ~isreal(F_addr) || any(~isfinite(F_addr), 'all') || ...
       norm(F_addr' * F_addr - eye(3), 'fro') >= tol_so3 || abs(det(F_addr) - 1.0) >= tol_so3
        error('gs3dx:head_orientation:invalid_calibration', ...
            'F(:,:,%d) at address is not a valid SO(3) rotation matrix', address_frame);
    end

    R_target = nan(3, 3, N);
    gaps_out = true(1, N);

    for f = 1:N
        if ~cluster_valid(f)
            continue;
        end
        F_f = F(:, :, f);
        if ~isreal(F_f) || any(~isfinite(F_f), 'all') || ...
           norm(F_f' * F_f - eye(3), 'fro') >= tol_so3 || abs(det(F_f) - 1.0) >= tol_so3
            continue;
        end

        R_rel = F_f * F_addr';
        R_f = R_rel * R_model_head_address;

        % Guard numerical roundoff in SO(3)
        [U, ~, V] = svd(R_f);
        R_target(:, :, f) = U * diag([1, 1, det(U * V')]) * V';
        gaps_out(f) = false;
    end

    meta = struct( ...
        'F_address', F_addr, ...
        'R_model_head_address', R_model_head_address, ...
        'address_frame', address_frame, ...
        'notes', "Pure cluster-relative rotation from address compounded onto model address pose." ...
    );
end
