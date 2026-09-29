function overrides = solve_starting_pose(target, base_input_mat, opts)
%SOLVE_STARTING_POSE  Find the *StartPosition* / *StartVelocity* overrides
%   that put the model's grip at the measured grip pose at the address frame.
%
%   OVERRIDES = SOLVE_STARTING_POSE(TARGET, BASE_INPUT_MAT) runs the
%   Stage-1 fit described in GRIP_FIT_PLAYBOOK.md.  It picks a small set of
%   model-workspace start-position variables (default: 8 DOF spanning hip
%   translation, hip rotation, shoulder Y, and elbow scalars), wraps each
%   candidate in a 5 ms FastRestart Simulink call, and returns the
%   perturbations that minimise
%
%     J(x) = w_pos * ||grip_model(0) - target.grip(addr_idx, :)||^2
%          + w_ori * geodesic_angle(R_model_grip, R_meas_grip)^2
%
%   where ``addr_idx`` is the aligned-grid index of the address frame. The
%   raw-sheet ``target.events.A_sample`` is NOT a valid index into the
%   resampled/aligned ``target.grip`` — see "ADDRESS-FRAME INDEXING" below.
%
%   The output OVERRIDES is a struct keyed by model-workspace variable
%   name → numeric value, suitable for `prepare_fast_sim_input`'s
%   `opts.input_overrides` (and therefore for `simulate_with_coefficients`).
%
%   OVERRIDES = SOLVE_STARTING_POSE(TARGET, BASE_INPUT_MAT, OPTS) accepts:
%     .vars         cell array of model-workspace variable names
%                   (default 8 DOF list — see DEFAULT_VARS below)
%     .max_iters    fminsearch MaxFunEvals + MaxIter (default 200)
%     .tol          position tolerance in metres (default 1e-4)
%     .verbose      print per-iter residuals (default true)
%     .rng_seed     determinism seed (default 42).  Stage-1 is deterministic
%                   given identical RNG state; the seed is restored locally.
%     .w_pos        position weight (default 1.0, m^-2)
%     .w_ori        orientation weight (default 0.1, rad^-2)
%     .stop_time    inner-sim stop time (default 0.005 s — just enough for
%                   one logged frame)
%     .model_name   default 'GolfSwing3D_Kinetic'
%     .x0           initial perturbation vector (default zeros(numel(vars),1))
%     .lb / .ub     symmetric bounds applied via a barrier in the cost
%                   (defaults: ±2 m for translations, ±pi for angles).
%                   Bounds are *also* asserted as a postcondition.
%
%   Determinism: the only stochastic ingredient is fminsearch's initial
%   simplex, which is deterministic once we control the RNG.  The function
%   saves+restores the global RNG state, so callers see no side effect.
%
%   See also: PREPARE_FAST_SIM_INPUT, LOAD_CLUB_TARGET_EXCEL,
%             SIMULATE_WITH_COEFFICIENTS, GRIP_FIT_PLAYBOOK.

    arguments
        target          (1,1) struct
        base_input_mat  (1,1) string {mustBeFile}
        opts            (1,1) struct = struct()
    end

    % ---- 1. Fill defaults -------------------------------------------------
    DEFAULT_VARS = { ...
        'TranslationStartPositionX', ...
        'TranslationStartPositionY', ...
        'TranslationStartPositionZ', ...
        'HipStartPositionZ',         ...
        'LSStartPositionY',          ...
        'RSStartPositionY',          ...
        'LEStartPosition',           ...
        'REStartPosition'};

    if ~isfield(opts, 'vars'),       opts.vars       = DEFAULT_VARS;          end
    if ~isfield(opts, 'max_iters'),  opts.max_iters  = 200;                   end
    if ~isfield(opts, 'tol'),        opts.tol        = 1e-4;                  end
    if ~isfield(opts, 'verbose'),    opts.verbose    = true;                  end
    if ~isfield(opts, 'rng_seed'),   opts.rng_seed   = 42;                    end
    if ~isfield(opts, 'w_pos'),      opts.w_pos      = 1.0;                   end
    if ~isfield(opts, 'w_ori'),      opts.w_ori      = 0.1;                   end
    if ~isfield(opts, 'stop_time'),  opts.stop_time  = 0.005;                 end
    if ~isfield(opts, 'model_name'), opts.model_name = 'GolfSwing3D_Kinetic'; end

    vars = cellstr(opts.vars);
    n    = numel(vars);

    % Plausibility bounds: ±2 m for the three translational DOFs in our
    % default list, ±pi for everything else (joint-angle starts).
    is_trans = startsWith(vars, 'TranslationStartPosition');
    if ~isfield(opts, 'lb')
        opts.lb = -pi * ones(n, 1);
        opts.lb(is_trans) = -2.0;
    end
    if ~isfield(opts, 'ub')
        opts.ub =  pi * ones(n, 1);
        opts.ub(is_trans) =  2.0;
    end
    opts.lb = opts.lb(:);  opts.ub = opts.ub(:);
    assert(numel(opts.lb) == n && numel(opts.ub) == n, ...
        "solve_starting_pose:badBounds", ...
        "Precondition: opts.lb / opts.ub must each have numel(opts.vars) entries");
    assert(all(opts.ub > opts.lb), ...
        "solve_starting_pose:badBounds", ...
        "Precondition: opts.ub must be strictly greater than opts.lb element-wise");

    if ~isfield(opts, 'x0')
        opts.x0 = zeros(n, 1);
    end
    x0 = double(opts.x0(:));
    assert(numel(x0) == n, ...
        "solve_starting_pose:badX0", ...
        "Precondition: opts.x0 must have numel(opts.vars) entries");

    % ---- 2. Validate target ----------------------------------------------
    assert(isfield(target, 'grip') && isfield(target, 'grip_quat') && ...
           isfield(target, 'events'), ...
        "solve_starting_pose:badTarget", ...
        "Precondition: target must have .grip, .grip_quat, .events");
    assert(isfield(target.events, 'A_sample') && ...
           isfinite(target.events.A_sample), ...
        "solve_starting_pose:noAddressFrame", ...
        "Precondition: target.events.A_sample must be finite");

    % Map A_sample (1-indexed sample number from the Wiffle row-1 header)
    % onto an index into the aligned target arrays.
    %
    % A_sample is a raw sheet sample number (e.g., 240 Hz frame count). The
    % aligned target arrays (target.grip, target.grip_quat, etc.) are
    % resampled onto the simulation grid via align_to_simulation_grid, which
    % also trims pre-address data. Therefore, A_sample cannot be used
    % directly as an index.
    %
    % DOMINANT CASE: The aligned series starts at or after the address frame,
    % so address is at row 1 (target.time(1) == 0). We pin to index 1.
    %
    % FALLBACK: If the aligned series somehow starts before address (rare),
    % clamp to the first row.
    %
    % TODO(#4091 resolution): If callers need arbitrary-frame addressing
    % (e.g., top-of-backswing at a later time), the loader must capture a
    % mapping from raw sample numbers to aligned indices, or A_sample must
    % be documented as "the time of address in raw timegrid", not "sample #".
    n_rows = size(target.grip, 1);
    if isfield(target, 'time') && ~isempty(target.time)
        % Aligned series with resampling: address is always row 1.
        addr_idx = 1;
    else
        % No time field: raw (unaligned) arrays. Pin to first row (fallback).
        addr_idx = 1;
    end
    addr_idx = max(1, min(n_rows, addr_idx));  % Safety clamp.

    grip_target_xyz = target.grip(addr_idx, :);
    R_meas_grip     = local_quat_to_rotmat(target.grip_quat(addr_idx, :));

    assert(all(isfinite(grip_target_xyz)), ...
        "solve_starting_pose:nonFiniteTarget", ...
        "Precondition: target.grip(addr_idx,:) must be finite");
    assert(all(isfinite(R_meas_grip(:))), ...
        "solve_starting_pose:nonFiniteTargetQuat", ...
        "Precondition: target.grip_quat(addr_idx,:) must be finite");

    % ---- 3. Determinism --------------------------------------------------
    rng_state_before = rng();
    cleanup_rng = onCleanup(@() rng(rng_state_before));
    rng(opts.rng_seed, 'twister');

    % ---- 4. Build the inner cost closure ---------------------------------
    cost_fn = @(x) local_pose_cost(x, vars, base_input_mat, ...
                                   grip_target_xyz, R_meas_grip, opts);

    % ---- 5. Run fminsearch -----------------------------------------------
    fmin_opts = optimset( ...
        'MaxFunEvals', opts.max_iters, ...
        'MaxIter',     opts.max_iters, ...
        'TolX',        opts.tol, ...
        'TolFun',      opts.tol^2, ...
        'Display',     ternary(opts.verbose, 'iter', 'off'));

    if opts.verbose
        fprintf('[solve_starting_pose] starting fminsearch (n=%d, max_iters=%d, A_sample=%d)\n', ...
                n, opts.max_iters, addr_idx);
        fprintf('[solve_starting_pose] target.grip(addr) = [% .4f % .4f % .4f]\n', ...
                grip_target_xyz);
    end

    x_opt = fminsearch(cost_fn, x0, fmin_opts);

    % Bound clamp (postcondition guarantee).  fminsearch is unconstrained,
    % but the local_pose_cost barrier already pushes it into the box.  We
    % still clamp here so the returned overrides cannot violate the spec.
    x_opt = max(opts.lb, min(opts.ub, x_opt(:)));

    % ---- 6. Pack overrides ----------------------------------------------
    overrides = local_vec_to_struct(x_opt, vars);

    % ---- 7. Postconditions ----------------------------------------------
    f = fieldnames(overrides);
    assert(numel(f) == n, ...
        "solve_starting_pose:postBadCount", ...
        "Postcondition: overrides struct must have one field per opts.vars");
    for k = 1:n
        v = overrides.(f{k});
        assert(isscalar(v) && isfinite(v) && isreal(v), ...
            "solve_starting_pose:postBadValue", ...
            "Postcondition: overrides.%s must be a finite real scalar", f{k});
        assert(v >= opts.lb(k) - 10*eps && v <= opts.ub(k) + 10*eps, ...
            "solve_starting_pose:postOutOfBounds", ...
            "Postcondition: overrides.%s = %g out of [% g, %g]", ...
            f{k}, v, opts.lb(k), opts.ub(k));
    end

    if opts.verbose
        fprintf('[solve_starting_pose] done. ||residual_grip|| ~ %.4f mm\n', ...
                1000 * sqrt(local_pos_residual_only(x_opt, vars, base_input_mat, ...
                                                    grip_target_xyz, opts)));
    end
end


%% =====================================================================
function J = local_pose_cost(x, vars, base_input_mat, grip_target, R_meas, opts)
%LOCAL_POSE_COST  Scalar cost for one candidate perturbation vector x.
    x = x(:);

    % Box-constraint barrier (smooth quadratic outside the box, zero inside).
    pen = 0.0;
    over_ub = max(0, x - opts.ub);
    over_lb = max(0, opts.lb - x);
    if any(over_ub > 0) || any(over_lb > 0)
        pen = 1e3 * (sum(over_ub.^2) + sum(over_lb.^2));
    end

    [r_grip_pos, r_grip_R] = local_run_inner_sim(x, vars, base_input_mat, opts);

    if any(isnan(r_grip_pos)) || any(isnan(r_grip_R(:)))
        % Failed sim → very large but finite cost so fminsearch can still
        % steer away from this region.
        J = 1e6 + pen;
        return;
    end

    e_pos = r_grip_pos - grip_target;
    pos_term = sum(e_pos .^ 2);

    % Geodesic angle between two rotation matrices: acos((trace(R'*R_meas)-1)/2).
    M = r_grip_R' * R_meas;
    cos_theta = max(-1, min(1, (trace(M) - 1) / 2));
    ang = acos(cos_theta);
    ori_term = ang^2;

    J = opts.w_pos * pos_term + opts.w_ori * ori_term + pen;
end


%% =====================================================================
function J = local_pos_residual_only(x, vars, base_input_mat, grip_target, opts)
%LOCAL_POS_RESIDUAL_ONLY  Position-squared residual at x (for reporting).
    [r_grip_pos, ~] = local_run_inner_sim(x, vars, base_input_mat, opts);
    if any(isnan(r_grip_pos))
        J = NaN;  return;
    end
    e_pos = r_grip_pos - grip_target;
    J = sum(e_pos .^ 2);
end


%% =====================================================================
function [r_grip_pos, r_grip_R] = local_run_inner_sim(x, vars, base_input_mat, opts)
%LOCAL_RUN_INNER_SIM  Build the SimulationInput for one candidate, run, return
%   grip position (1x3) and grip rotation matrix (3x3) at frame 1.
    r_grip_pos = nan(1, 3);
    r_grip_R   = nan(3, 3);

    % Layer the base input MAT first, then perturbations on top.
    base = load(base_input_mat);
    base_overrides = struct();
    f = fieldnames(base);
    for k = 1:numel(f)
        base_overrides.(f{k}) = base.(f{k});
    end
    for k = 1:numel(vars)
        % Apply the perturbation as an *absolute* override; the value the
        % optimizer sees is therefore the final workspace value.  The
        % default x0=0 means we try the unperturbed Impact MAT first, but
        % the optimizer can move freely from there.  We therefore ADD x to
        % the base value so x=0 means "no change".
        if isfield(base_overrides, vars{k})
            cur = base_overrides.(vars{k});
            % Fields loaded from the MAT are Simulink.Parameter objects;
            % extract the numeric .Value before arithmetic.
            if isa(cur, 'Simulink.Parameter')
                cur = cur.Value;
            end
            base_overrides.(vars{k}) = cur + x(k);
        else
            % Variable not present in MAT: set as scalar perturbation.
            base_overrides.(vars{k}) = x(k);
        end
    end

    sim_opts = struct( ...
        'model_name',      opts.model_name, ...
        'stop_time',       opts.stop_time, ...
        'simscape_log',    'all', ...
        'input_overrides', base_overrides);

    try
        in = prepare_fast_sim_input([], sim_opts);
        simOut = sim(in);
    catch ME
        if opts.verbose
            fprintf('[solve_starting_pose] inner sim failed: %s\n', ME.message);
        end
        return;
    end

    % Pull grip position from MidpointCalcsLogs.MPGlobalPosition (frame 1).
    try
        csb = simOut.CombinedSignalBus;
        d = double(csb.MidpointCalcsLogs.MPGlobalPosition.Data);
        d = reshape(d(1, :), 1, []);
        r_grip_pos = d(1:3);
    catch
    end

    % Pull grip rotation from MomentandCoupleLogs.RotationTransformMP frame 1.
    try
        csb = simOut.CombinedSignalBus;
        d = double(csb.MomentandCoupleLogs.RotationTransformMP.Data);
        if ndims(d) == 3 && size(d, 1) == 3 && size(d, 2) == 3
            r_grip_R = squeeze(d(:, :, 1));
        end
    catch
    end

    if any(isnan(r_grip_R(:)))
        % Fallback: some builds publish the duplicate channel under the
        % MidpointCalcsLogs branch.
        try
            csb = simOut.CombinedSignalBus;
            d = double(csb.MidpointCalcsLogs.RotationTransformMP.Data);
            if ndims(d) == 3 && size(d, 1) == 3 && size(d, 2) == 3
                r_grip_R = squeeze(d(:, :, 1));
            end
        catch
        end
    end
end


%% =====================================================================
function s = local_vec_to_struct(x, vars)
%LOCAL_VEC_TO_STRUCT  Pack the optimizer's solution into a name->value struct.
    s = struct();
    for k = 1:numel(vars)
        s.(vars{k}) = x(k);
    end
end


%% =====================================================================
function R = local_quat_to_rotmat(q)
%LOCAL_QUAT_TO_ROTMAT  [w x y z] unit quaternion → 3x3 rotation matrix.
    q = q(:)';
    q = q / max(norm(q), eps);
    w = q(1); x = q(2); y = q(3); z = q(4);
    R = [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w); ...
         2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w); ...
         2*(x*z - y*w),       2*(y*z + x*w),   1 - 2*(x*x + y*y)];
end


%% =====================================================================
function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
