function diag = gs3dx_ik_diagnostics(ik, opts)
%GS3DX_IK_DIAGNOSTICS  Kinematic tracking diagnostics for whole-body IK.
%
%   DIAG = GS3DX_IK_DIAGNOSTICS(IK) computes kinematic tracking diagnostics
%   from a GS3DX_WHOLE_BODY_IK result struct IK.
%
%   DIAG = GS3DX_IK_DIAGNOSTICS(IK, phase_ranges=PR) computes overall
%   diagnostics as well as partitioned diagnostics for each named range
%   specified in OPTS.phase_ranges.
%
%   Inputs:
%     ik: struct from GS3DX_WHOLE_BODY_IK containing:
%       .names       target names (cell array of char or string array)
%       .frames      tracked capture frames (strictly increasing positive integers)
%       .t           timestamps in seconds (strictly increasing, real)
%       .residual    measured distance residuals (targets x frames, m, NaN where missing)
%       .points      predicted target positions (3 x targets x frames, m, real, finite)
%       .status      solver status per frame (1 = loop closed)
%       .joint       (optional) 2D joint coordinates (ids x frames, real, finite)
%       .model       (optional) model variant identifier
%
%     opts.phase_ranges: (optional) struct where each field is a 1x2 vector
%       [capture_start, capture_end] defining inclusive capture frame boundaries.
%       Empty phase ranges (no samples falling within the range) fail closed
%       with an error.
%
%   Outputs:
%     diag: struct with fields:
%       .names       target names (1 x targets cell array of char)
%       .frames      capture frames evaluated (1 x frames double)
%       .t           timestamps evaluated (1 x frames double, s)
%       .targets     struct with one sub-struct per target name:
%                    .residual_rms   RMS of measured residuals (m, NaN if 0 samples)
%                    .residual_max   maximum measured residual (m, NaN if 0 samples)
%                    .sample_count   count of measured samples (>= 0)
%                    .worst_frame    capture frame of maximum residual (NaN if 0 samples)
%                    .max_step_speed maximum predicted step speed (m/s, NaN if < 2 frames)
%       .aggregate   struct of aggregate kinematic tracking metrics:
%                    .rms_mean       mean of sampled frame RMS residuals (m, NaN if 0 samples)
%                    .rms_max        maximum of sampled frame RMS residuals (m, NaN if 0 samples)
%                    .worst_frame    capture frame of maximum frame RMS residual
%                    .sample_count   total measured residual samples across all targets/frames
%                    .frame_count    count of frames with >= 1 measured sample
%                    .max_step_speed maximum predicted step speed across all targets (m/s)
%       .phases      struct with one sub-struct per phase in opts.phase_ranges
%
%   Domain Rule & Scope:
%     This diagnostic tool evaluates kinematic tracking residuals and
%     predicted kinematic step speeds. It does not infer contact from peak
%     speed and makes no claim of physical qualification or player quality.
%
%   See also GS3DX_WHOLE_BODY_IK.

    arguments
        ik (1,1) struct
        opts.phase_ranges (1,1) struct = struct()
    end

    % Validate input IK struct and extract normalized names
    names = local_validate_ik(ik);

    frames = double(reshape(ik.frames, 1, []));
    t = double(reshape(ik.t, 1, []));
    residual = double(ik.residual);
    points = double(ik.points);

    % Compute overall diagnostics
    core = local_compute_core(names, frames, t, residual, points);

    diag = struct();
    if isfield(ik, 'model')
        diag.model = ik.model;
    end
    diag.names = names;
    diag.frames = frames;
    diag.t = t;
    diag.targets = core.targets;
    diag.aggregate = core.aggregate;
    diag.phases = struct();

    % Compute per-phase diagnostics if phase_ranges provided
    if ~isempty(fieldnames(opts.phase_ranges))
        p_names = fieldnames(opts.phase_ranges);
        for p = 1:numel(p_names)
            p_name = p_names{p};
            rng = opts.phase_ranges.(p_name);

            % Validate range specifications
            assert(isnumeric(rng) && isreal(rng) && isvector(rng) && numel(rng) == 2 && all(isfinite(rng)), ...
                'gs3dx:ik_diagnostics:InvalidPhaseRange', ...
                'Phase range "%s" must be a 2-element real numeric vector', p_name);
            assert(all(rng > 0) && all(floor(rng) == rng), ...
                'gs3dx:ik_diagnostics:InvalidPhaseRange', ...
                'Phase range "%s" must contain positive integers', p_name);
            assert(rng(1) <= rng(2), ...
                'gs3dx:ik_diagnostics:InvalidPhaseRange', ...
                'Phase range "%s" start (%d) exceeds end (%d)', p_name, rng(1), rng(2));

            % Find frames in inclusive range
            p_mask = (frames >= rng(1)) & (frames <= rng(2));
            p_idx = find(p_mask);

            % Fail-closed on empty phase
            assert(~isempty(p_idx), ...
                'gs3dx:ik_diagnostics:EmptyPhase', ...
                'Phase "%s" [%d, %d] contains no samples in IK frames', p_name, rng(1), rng(2));

            sub_frames = frames(p_idx);
            sub_t = t(p_idx);
            sub_res = residual(:, p_idx);
            sub_pts = points(:, :, p_idx);

            sub_core = local_compute_core(names, sub_frames, sub_t, sub_res, sub_pts);

            phase_entry = struct();
            phase_entry.frames = sub_frames;
            phase_entry.t = sub_t;
            phase_entry.targets = sub_core.targets;
            phase_entry.aggregate = sub_core.aggregate;

            diag.phases.(p_name) = phase_entry;
        end
    end
end

function norm_names = local_validate_ik(ik)
% Strict validation of whole_body_ik contract and preconditions.
    assert(isstruct(ik), 'gs3dx:ik_diagnostics:InvalidInput', 'IK must be a struct');

    need = {'names', 'frames', 't', 'residual', 'points', 'status'};
    for k = 1:numel(need)
        assert(isfield(ik, need{k}), 'gs3dx:ik_diagnostics:MissingField', ...
            'IK missing required field "%s"', need{k});
    end

    % 1. Names: cell array or string array of non-empty scalar text valid as struct field name
    raw_names = ik.names;
    assert(iscell(raw_names) || isstring(raw_names), ...
        'gs3dx:ik_diagnostics:InvalidNames', 'IK.names must be a cell array or string array');
    nt = numel(raw_names);
    assert(nt >= 1, 'gs3dx:ik_diagnostics:InvalidNames', 'IK.names cannot be empty');

    norm_names = cell(1, nt);
    for k = 1:nt
        if iscell(raw_names)
            item = raw_names{k};
        else
            item = raw_names(k);
        end
        assert((ischar(item) && (isrow(item) || isequal(item, '')) && ~isempty(item)) || ...
               (isstring(item) && isscalar(item) && strlength(item) > 0), ...
               'gs3dx:ik_diagnostics:InvalidNames', ...
               'Each target name must be non-empty scalar text');
        nm = char(item);
        assert(isvarname(nm), 'gs3dx:ik_diagnostics:InvalidNames', ...
            'Target name "%s" is not a valid MATLAB struct field name', nm);
        norm_names{k} = nm;
    end
    assert(numel(unique(norm_names)) == nt, 'gs3dx:ik_diagnostics:DuplicateNames', ...
        'Target names must be unique after normalization');

    % 2. Frames: positive integers, real, strictly increasing
    frames = ik.frames;
    assert(isnumeric(frames) && isreal(frames) && isvector(frames) && ~isempty(frames), ...
        'gs3dx:ik_diagnostics:InvalidFrames', 'IK.frames must be a non-empty real numeric vector');
    nf = numel(frames);
    assert(all(isfinite(frames)) && all(frames > 0) && all(floor(frames) == frames), ...
        'gs3dx:ik_diagnostics:InvalidFrames', 'IK.frames must be positive integers');
    if nf > 1
        assert(all(diff(frames) > 0), 'gs3dx:ik_diagnostics:InvalidFrames', ...
            'IK.frames must be strictly increasing');
    end

    % 3. Time: strictly increasing real seconds, length matching frames
    t = ik.t;
    assert(isnumeric(t) && isreal(t) && isvector(t) && numel(t) == nf, ...
        'gs3dx:ik_diagnostics:InvalidTime', 'IK.t must be a real numeric vector with length matching frames');
    assert(all(isfinite(t)), 'gs3dx:ik_diagnostics:InvalidTime', 'IK.t must be finite');
    if nf > 1
        assert(all(diff(t) > 0), 'gs3dx:ik_diagnostics:InvalidTime', ...
            'IK.t must be strictly increasing in seconds');
    end

    % 4. Residual: [targets x frames], real, allow NaN but reject negative/infinite
    res = ik.residual;
    assert(isnumeric(res) && isreal(res) && ismatrix(res) && isequal(size(res), [nt, nf]), ...
        'gs3dx:ik_diagnostics:InvalidShape', 'IK.residual must be a real 2D matrix of size [targets x frames]');
    valid_res = res(~isnan(res));
    assert(all(isfinite(valid_res)), 'gs3dx:ik_diagnostics:InvalidResidual', ...
        'IK.residual cannot contain infinite values');
    assert(all(valid_res >= 0), 'gs3dx:ik_diagnostics:InvalidResidual', ...
        'IK.residual cannot contain negative values');

    % 5. Points: [3 x targets x frames], real, finite
    pts = ik.points;
    assert(isnumeric(pts) && isreal(pts) && ndims(pts) <= 3 && size(pts, 1) == 3 && ...
        size(pts, 2) == nt && size(pts, 3) == nf, ...
        'gs3dx:ik_diagnostics:InvalidShape', 'IK.points must be real [3 x targets x frames]');
    assert(all(isfinite(pts(:))), 'gs3dx:ik_diagnostics:NonFinitePoints', ...
        'IK.points must be finite');

    % 6. Status: loop closure status == 1 for all frames
    st = ik.status;
    assert(((isnumeric(st) && isreal(st)) || islogical(st)) && isvector(st) && numel(st) == nf, ...
        'gs3dx:ik_diagnostics:InvalidStatus', 'IK.status must be a real vector matching frames');
    assert(all(st == 1), 'gs3dx:ik_diagnostics:LoopClosure', ...
        'IK.status must indicate loop closure (status == 1) for all frames');

    % 7. Joint: if present, real numeric 2D finite matrix with columns matching frames
    if isfield(ik, 'joint') && ~isempty(ik.joint)
        assert(isnumeric(ik.joint) && isreal(ik.joint) && ismatrix(ik.joint) && size(ik.joint, 2) == nf, ...
            'gs3dx:ik_diagnostics:InvalidShape', 'IK.joint must be a real 2D matrix with columns matching frames');
        assert(all(isfinite(ik.joint(:))), 'gs3dx:ik_diagnostics:NonFiniteJoints', ...
            'IK.joint must be finite');
    end
end

function core = local_compute_core(names, frames, t, residual, points)
% Compute per-target and aggregate metrics on a validated slice of frames.
    nt = numel(names);
    nf = numel(frames);

    % Step speeds: computed from predicted points and time, even if measurements are missing
    if nf >= 2
        dt = reshape(t(2:end) - t(1:end-1), 1, nf - 1);
        dp = points(:, :, 2:end) - points(:, :, 1:end-1); % 3 x nt x (nf - 1)
        step_dist = reshape(sqrt(sum(dp .^ 2, 1)), nt, nf - 1); % nt x (nf - 1)
        step_speeds = step_dist ./ repmat(dt, nt, 1); % nt x (nf - 1)
        target_max_speeds = max(step_speeds, [], 2); % nt x 1
        agg_max_speed = max(target_max_speeds);
    else
        target_max_speeds = nan(nt, 1);
        agg_max_speed = NaN;
    end

    % Per-target residual metrics
    targets_struct = struct();
    for k = 1:nt
        nm = names{k};
        r_k = residual(k, :);
        valid_idx = find(~isnan(r_k));
        n_meas = numel(valid_idx);

        stat = struct();
        if n_meas > 0
            meas_r = r_k(valid_idx);
            stat.residual_rms = sqrt(mean(meas_r .^ 2));
            [max_val, rel_idx] = max(meas_r);
            stat.residual_max = max_val;
            stat.sample_count = n_meas;
            stat.worst_frame = frames(valid_idx(rel_idx));
        else
            % No measured samples: NaN summary plus explicit zero count
            stat.residual_rms = NaN;
            stat.residual_max = NaN;
            stat.sample_count = 0;
            stat.worst_frame = NaN;
        end
        stat.max_step_speed = target_max_speeds(k);

        targets_struct.(nm) = stat;
    end

    % Aggregate sampled RMS: frame-wise RMS of measured targets
    frame_rms = nan(1, nf);
    for j = 1:nf
        col = residual(:, j);
        valid_col = col(~isnan(col));
        if ~isempty(valid_col)
            frame_rms(j) = sqrt(mean(valid_col .^ 2));
        end
    end

    sampled_frames = find(~isnan(frame_rms));
    agg = struct();
    if ~isempty(sampled_frames)
        sampled_rms = frame_rms(sampled_frames);
        agg.rms_mean = mean(sampled_rms);
        [agg_max, best_rel] = max(sampled_rms);
        agg.rms_max = agg_max;
        agg.worst_frame = frames(sampled_frames(best_rel));
        agg.sample_count = nnz(~isnan(residual));
        agg.frame_count = numel(sampled_frames);
    else
        % No measured samples across any target/frame
        agg.rms_mean = NaN;
        agg.rms_max = NaN;
        agg.worst_frame = NaN;
        agg.sample_count = 0;
        agg.frame_count = 0;
    end
    agg.max_step_speed = agg_max_speed;

    core.targets = targets_struct;
    core.aggregate = agg;
end
