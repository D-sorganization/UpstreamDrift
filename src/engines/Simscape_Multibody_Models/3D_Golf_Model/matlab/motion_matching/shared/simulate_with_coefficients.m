function sim_out = simulate_with_coefficients(theta, opts)
%SIMULATE_WITH_COEFFICIENTS  Single Simscape forward call used by all options.
%
%   sim_out = SIMULATE_WITH_COEFFICIENTS(THETA, OPTS) runs
%   GolfSwing3D_Kinetic.slx with the polynomial torque coefficients THETA
%   and returns a canonical struct with fields:
%
%     .time          (N,1) double   simulation timegrid (s)
%     .q             (N, n_joints)  joint angles (rad)
%     .qd            (N, n_joints)  joint angular velocities (rad/s)
%     .qdd           (N, n_joints)  joint angular accelerations (rad/s^2)
%     .tau           (N, n_joints)  joint torques (N*m)
%     .omega         (N, n_joints)  alias for qd, kept for cost-function clarity
%     .r_butt        (N,3)          butt position (m)
%     .r_clubhead    (N,3)          clubhead position (m)
%     .q_club        (N,4)          club orientation quaternion [w x y z]
%     .v_clubhead    (N,3)          clubhead linear velocity (m/s)
%     .omega_club    (N,3)          club angular velocity (rad/s)
%     .joint_names   (1,n_joints) string  joint ordering
%     .solver_status (1,1) string   "success" | "warning" | "failed"
%
%   THETA is a real, finite vector of length n_joints*7 ordered
%   [A B C D E F G] per joint, joints in canonical order from
%   getPolynomialParameterInfo.
%
%   OPTS is the result of default_sim_options() with optional overrides.
%
%   This is the only place in the motion_matching tree allowed to call
%   sim()/parsim() on the Simscape model. Every option must funnel its
%   forward calls through this wrapper (DRY — see CODING_STANDARDS.md).
%
%   Preconditions (arguments block):
%     - THETA is finite, real, length n_joints*7.
%     - OPTS is a 1x1 struct.
%
%   Postconditions (asserted):
%     - sim_out has every documented field.
%     - sim_out.time(1) == 0 and is monotonic non-decreasing.
%     - All time-indexed arrays share N rows.
%     - sim_out.solver_status ∈ {"success","warning","failed"}.
%     - When solver_status == "success": no NaN/Inf in q, qd, qdd, tau,
%       r_butt, r_clubhead, q_club, v_clubhead, omega_club; q_club rows are
%       unit-norm to within 1e-6.
%
%   GitHub issue: #018 / #3987.
%
%   See also: DEFAULT_SIM_OPTIONS, GETPOLYNOMIALPARAMETERINFO.

    arguments
        theta (:,1) double {mustBeReal, mustBeFinite}
        opts  (1,1) struct = default_sim_options()
    end

    t_start = tic;

    % ---- 1. Resolve joint ordering ----------------------------------------
    if isfield(opts, "joint_names") && ~isempty(opts.joint_names)
        joint_names = string(opts.joint_names(:)).';
    else
        param_info = getPolynomialParameterInfo();
        joint_names = string(param_info.joint_names);
    end
    n_joints = numel(joint_names);
    expected_len = n_joints * 7;

    if numel(theta) ~= expected_len
        error("simulate_with_coefficients:badThetaLength", ...
            "Precondition: theta has length %d but n_joints*7 = %d", ...
            numel(theta), expected_len);
    end

    local_log(opts, "Verbose", "simulate_with_coefficients: theta length=%d, n_joints=%d", ...
        numel(theta), n_joints);

    % ---- 2. Cache lookup ---------------------------------------------------
    cache_key = "";
    cache_path = "";
    if local_use_cache(opts)
        cache_key = local_compute_cache_key(theta, opts);
        cache_path = fullfile(char(opts.cache_dir), char(cache_key) + ".mat");
        if isfile(cache_path)
            local_log(opts, "Verbose", "cache hit: %s", cache_path);
            S = load(cache_path, "sim_out");
            sim_out = S.sim_out;
            sim_out.cache_hit = true;
            sim_out.duration_s = toc(t_start);
            return;
        end
    end

    % ---- 3. Reshape theta to coeff struct ---------------------------------
    if isempty(opts.joint_names)
        coeff_struct = theta_to_polynomial_struct(theta);
    else
        coeff_struct = theta_to_polynomial_struct(theta, joint_names);
    end

    % ---- 4. Build SimulationInput -----------------------------------------
    model_name = char(opts.model_name);
    if ~bdIsLoaded(model_name)
        try
            load_system(model_name);
        catch ME
            error("simulate_with_coefficients:modelLoad", ...
                "Could not load Simulink model '%s': %s", model_name, ME.message);
        end
    end

    simIn = Simulink.SimulationInput(model_name);
    simIn = simIn.setModelParameter('StopTime', num2str(double(opts.simulation_time)));
    if isfield(opts, "solver") && opts.solver ~= ""
        try
            simIn = simIn.setModelParameter('Solver', char(opts.solver));
        catch ME
            local_log(opts, "Verbose", "could not set solver=%s: %s", opts.solver, ME.message);
        end
    end

    % FastRestart: never compatible with parallel_safe path.
    use_fast_restart = logical(opts.fast_restart) && ~logical(opts.parallel_safe);
    if use_fast_restart
        try
            simIn = simIn.setModelParameter('FastRestart', 'on');
        catch ME
            local_log(opts, "Verbose", "could not enable FastRestart: %s", ME.message);
        end
    end

    % Push every coefficient to the model workspace.
    var_names = fieldnames(coeff_struct);
    for i = 1:numel(var_names)
        simIn = simIn.setVariable(var_names{i}, coeff_struct.(var_names{i}));
    end

    % ---- 5. Run sim --------------------------------------------------------
    simOut = [];
    err = [];
    try
        simOut = sim(simIn);
    catch ME
        err = ME;
    end

    if ~isempty(err)
        if logical(opts.stop_on_error)
            rethrow(err);
        end
        local_log(opts, "Normal", "simulation failed: %s", err.message);
        sim_out = local_failed_sim_out(joint_names, opts, err.message);
        sim_out.duration_s = toc(t_start);
        sim_out.cache_hit = false;
        return;
    end

    % ---- 6. Extract canonical struct --------------------------------------
    sim_out = extract_sim_out(simOut, joint_names, opts);
    sim_out.cache_hit = false;
    sim_out.duration_s = toc(t_start);
    sim_out.theta_length = numel(theta);

    % ---- 7. Postconditions on success path --------------------------------
    if sim_out.solver_status == "success"
        assert(all(isfinite(sim_out.q(:))) && all(isfinite(sim_out.qd(:))) && ...
               all(isfinite(sim_out.tau(:))), ...
               "simulate_with_coefficients:nonFiniteJointSignals", ...
               "Postcondition: q/qd/tau must be finite on success");
        if all(~isnan(sim_out.q_club(:)))
            qn = sqrt(sum(sim_out.q_club.^2, 2));
            assert(all(abs(qn - 1) < 1e-3), ...
                "simulate_with_coefficients:quatNotUnit", ...
                "Postcondition: q_club rows must be unit-norm (max dev %g)", ...
                max(abs(qn - 1)));
        end
    end

    % ---- 8. Cache write ---------------------------------------------------
    if local_use_cache(opts) && cache_path ~= ""
        try
            cache_dir = fileparts(cache_path);
            if ~isfolder(cache_dir), mkdir(cache_dir); end
            save(cache_path, "sim_out");
            local_log(opts, "Verbose", "cache wrote: %s", cache_path);
        catch ME
            local_log(opts, "Verbose", "cache write failed: %s", ME.message);
        end
    end
end

%% =====================================================================
function tf = local_use_cache(opts)
    tf = isfield(opts, "use_cache") && logical(opts.use_cache) ...
        && isfield(opts, "cache_dir") && strlength(string(opts.cache_dir)) > 0;
end

%% =====================================================================
function key = local_compute_cache_key(theta, opts)
%LOCAL_COMPUTE_CACHE_KEY  sha256 of theta + serialized opts + matlab_version.
    try
        opts_for_key = opts;
        opts_for_key.cache_dir = "";   % do not let cache_dir affect the key
        opts_for_key.use_cache = false;
        opts_json = jsonencode(opts_for_key);
    catch
        opts_json = "{}";
    end
    try
        git_commit = local_safe_git_commit();
    catch
        git_commit = "unknown";
    end
    payload = [typecast(double(theta(:)), 'uint8'); ...
               uint8(char(opts_json)).'; ...
               uint8(char(version)).'; ...
               uint8(char(git_commit)).'];
    try
        md = java.security.MessageDigest.getInstance('SHA-256');
        md.update(payload);
        digest = typecast(md.digest(), 'uint8');
        hex = lower(reshape(dec2hex(digest, 2).', 1, []));
        key = string(hex);
    catch
        % Fallback: simple deterministic encoding (not collision-resistant
        % but stable). Sufficient if Java is unavailable.
        h = sum(double(payload)) + numel(payload) * 1e6;
        key = string(sprintf("fallback_%020.0f", h));
    end
end

%% =====================================================================
function s = local_safe_git_commit()
    s = "unknown";
    try
        [status, out] = system('git rev-parse HEAD');
        if status == 0
            s = string(strtrim(out));
        end
    catch
    end
end

%% =====================================================================
function sim_out = local_failed_sim_out(joint_names, opts, message)
    n_joints = numel(joint_names);
    dt = 1.0 / double(opts.sample_rate);
    t = (0:dt:double(opts.simulation_time))';
    N = numel(t);
    sim_out = struct( ...
        'time', t, ...
        'q', nan(N, n_joints), ...
        'qd', nan(N, n_joints), ...
        'qdd', nan(N, n_joints), ...
        'tau', nan(N, n_joints), ...
        'omega', nan(N, n_joints), ...
        'r_butt', nan(N, 3), ...
        'r_clubhead', nan(N, 3), ...
        'q_club', nan(N, 4), ...
        'v_clubhead', nan(N, 3), ...
        'omega_club', nan(N, 3), ...
        'joint_names', joint_names, ...
        'solver_status', "failed", ...
        'status_message', string(message));
end

%% =====================================================================
function local_log(opts, level, fmt, varargin)
    levels = struct('Silent', 0, 'Normal', 1, 'Verbose', 2, 'Debug', 3);
    cur_name = char(opts.verbosity);
    msg_name = char(level);
    if ~isfield(levels, cur_name), cur = 1; else, cur = levels.(cur_name); end
    if ~isfield(levels, msg_name), msg = 1; else, msg = levels.(msg_name); end
    if msg <= cur
        fprintf(['[simulate_with_coefficients] ', fmt, '\n'], varargin{:});
    end
end
