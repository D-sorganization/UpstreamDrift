function data = gs3dx_head_input_data(jc, frames, calibration_frames, weight)
%GS3DX_HEAD_INPUT_DATA  Validate and extract head orientation input data for IK.
%
%   DATA = GS3DX_HEAD_INPUT_DATA(JC, FRAMES, CALIBRATION_FRAMES, WEIGHT)
%   validates head tracking inputs and returns a structured record for IK.
%
%   Biomechanical Scope:
%     The rotation matrices in jc.head_R represent relative cluster orientation,
%     NOT anatomical skull qualification.
%
%   Error ID:
%     Uniform error ID 'gs3dx:ik' for all validation failures.

    if nargin < 4
        error('gs3dx:ik', 'weight must be explicitly provided');
    end

    if ~isnumeric(weight) || ~isreal(weight) || ~isscalar(weight) || ~isfinite(weight) || weight < 0
        error('gs3dx:ik', 'weight must be a finite real non-negative scalar');
    end

    has_head_R = isstruct(jc) && isfield(jc, 'head_R');
    is_active = has_head_R || (weight > 0);

    if ~is_active
        data = struct('active', false, 'R', [], 'gaps', []);
        return;
    end

    if ~isstruct(jc)
        error('gs3dx:ik', 'jc must be a struct');
    end

    if ~isfield(jc, 'pelvis') || ~isnumeric(jc.pelvis) || ~isreal(jc.pelvis) || ...
       ndims(jc.pelvis) ~= 2 || size(jc.pelvis, 1) ~= 3 || size(jc.pelvis, 2) < 1
        error('gs3dx:ik', 'jc.pelvis must be real numeric 3xN with N >= 1');
    end
    N = size(jc.pelvis, 2);

    if ~has_head_R || ~isnumeric(jc.head_R) || ~isreal(jc.head_R) || ...
       ndims(jc.head_R) > 3 || size(jc.head_R, 1) ~= 3 || size(jc.head_R, 2) ~= 3 || size(jc.head_R, 3) ~= N
        error('gs3dx:ik', 'jc.head_R must be real numeric 3x3xN matching pelvis N');
    end

    if ~isfield(jc, 'gap') || ~isstruct(jc.gap) || ~isfield(jc.gap, 'head_R') || ...
       ~islogical(jc.gap.head_R) || ~isvector(jc.gap.head_R) || numel(jc.gap.head_R) ~= N
        error('gs3dx:ik', 'jc.gap.head_R must be a logical vector with numel equal to N');
    end

    if ~isnumeric(frames) || ~isreal(frames) || ~isvector(frames) || ...
       any(~isfinite(frames)) || isempty(frames) || any(floor(frames) ~= frames) || ...
       any(frames < 1) || any(frames > N)
        error('gs3dx:ik', 'frames must be a non-empty finite positive integer vector in 1..N');
    end

    if ~isempty(calibration_frames)
        if ~isnumeric(calibration_frames) || ~isreal(calibration_frames) || ~isvector(calibration_frames) || ...
           any(~isfinite(calibration_frames)) || any(floor(calibration_frames) ~= calibration_frames) || ...
           any(calibration_frames < 1) || any(calibration_frames > N)
            error('gs3dx:ik', 'calibration_frames must be empty or a finite positive integer vector in 1..N');
        end
    end

    gaps = reshape(jc.gap.head_R, 1, N);
    head_R = double(jc.head_R);

    tol = 1e-8;
    for k = 1:N
        if gaps(k)
            continue;
        end
        Rk = head_R(:, :, k);
        if any(~isfinite(Rk), 'all') || abs(det(Rk) - 1.0) > tol || norm(Rk' * Rk - eye(3), 'fro') > tol
            error('gs3dx:ik', 'Measured head_R frame %d is not a finite proper SO(3) matrix', k);
        end
    end

    data = struct('active', true, 'R', head_R, 'gaps', gaps);
end
