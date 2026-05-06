function result = fit_swing_fmincon(target, opts)
%FIT_SWING_FMINCON  Single-start fmincon-sqp swing fit.
%
%   RESULT = FIT_SWING_FMINCON(TARGET, OPTS) optimises polynomial torque
%   coefficients THETA so that running the Simscape forward model with
%   THETA reproduces TARGET in the sense of
%   shared/COST_FUNCTION_SPEC.md.
%
%   The function is a thin orchestrator:
%     1. Builds [LB, UB] via build_coefficient_bounds(n_joints).
%     2. Builds an initial guess THETA0 from OPTS.initial_theta when given,
%        otherwise uniform-random within bounds (seeded by OPTS.rng_seed).
%     3. Wraps compute_cost in a closure that captures TARGET and the
%        forward-sim adapter.
%     4. Runs fmincon('algorithm','sqp').
%     5. Re-simulates the optimum and assembles the canonical RESULT
%        struct via build_result_struct.
%
%   TARGET is the canonical struct from shared/CLUB_IK_SPEC.md; at minimum
%   it must contain {time, butt, clubhead, club_quat, impact_idx}.
%
%   OPTS is the result of default_option1_options() with optional
%   overrides.
%
%   Preconditions (DbC, validated by `arguments`):
%     - TARGET has the required fields.
%     - OPTS is a 1x1 struct.
%
%   Postconditions:
%     - result.solver == "fmincon".
%     - 0 <= result.final_rmse_m < Inf.
%     - result.coefficients lies inside [lb, ub].
%
%   GitHub issue: #024 / #3993.
%
%   See also: COMPUTE_COST, SIMULATE_WITH_COEFFICIENTS,
%             DEFAULT_OPTION1_OPTIONS.

    arguments
        target (1,1) struct {validators.mustHaveFields(target, ...
            ["time","butt","clubhead","club_quat","impact_idx"])}
        opts (1,1) struct = default_option1_options()
    end

    t_start = tic;

    % ---- 1. Determine n_joints and bounds ----------------------------------
    n_joints = local_n_joints(opts);
    [lb, ub] = build_coefficient_bounds(n_joints);
    d = numel(lb);

    % ---- 2. Build initial guess --------------------------------------------
    theta0 = local_initial_theta(opts, lb, ub, d);

    % ---- 3. Build sim and cost closures ------------------------------------
    sim_fn  = @(theta) local_sim_adapter(theta, opts);
    cost_fn = @(theta) local_safe_cost(theta, target, sim_fn, opts);

    % ---- 4. fmincon options ------------------------------------------------
    iter_history = local_new_history();

    % Iteration capture uses a nested function so it can mutate
    % iter_history in the parent workspace.
    function stop_inner = capture(x, optimValues, state)
        stop_inner = false;
        if strcmp(state, "iter")
            row = table( ...
                double(optimValues.iteration), ...
                double(optimValues.fval), ...
                double(local_get(optimValues, 'firstorderopt', NaN)), ...
                double(local_get(optimValues, 'stepsize', NaN)), ...
                'VariableNames', ...
                {'iteration','fval','firstorderopt','stepsize'});
            iter_history = [iter_history; row]; %#ok<AGROW>
        end
        if isfield(opts, "output_fcn") && ~isempty(opts.output_fcn) && ...
                isa(opts.output_fcn, 'function_handle')
            try
                stop_inner = stop_inner || opts.output_fcn(x, optimValues, state);
            catch
                % Never let a buggy user callback abort the solve.
            end
        end
    end

    fmincon_opts = optimoptions('fmincon', ...
        'Algorithm',                'sqp', ...
        'Display',                  char(opts.display), ...
        'MaxIterations',            double(opts.max_iter), ...
        'MaxFunctionEvaluations',   double(opts.max_function_evals), ...
        'OptimalityTolerance',      double(opts.tol_fun), ...
        'StepTolerance',            double(opts.tol_x), ...
        'OutputFcn',                @capture);

    if opts.fd_central
        fmincon_opts = optimoptions(fmincon_opts, 'FiniteDifferenceType', 'central');
    end

    % ---- 5. Solve -----------------------------------------------------------
    [theta_star, fval, exitflag, output] = fmincon( ...
        cost_fn, theta0, [], [], [], [], lb, ub, [], fmincon_opts); %#ok<ASGLU>

    % ---- 6. Re-evaluate at the optimum to capture trajectory + terms -------
    final_sim_out = sim_fn(theta_star);
    [J_final, terms_final] = compute_cost(theta_star, target, sim_fn, opts.cost);

    % ---- 7. Assemble result -------------------------------------------------
    args = struct();
    args.coefficients     = theta_star;
    args.final_cost_terms = terms_final;
    args.final_sim_out    = final_sim_out;
    args.solver           = "fmincon";
    args.solver_options   = local_struct_from_optimoptions(fmincon_opts);
    args.options          = opts;
    args.target           = target;
    args.exitflag         = exitflag;
    args.output           = output;
    args.iter_history     = iter_history;
    args.duration_s       = toc(t_start);
    args.start_points     = theta0;
    args.start_costs      = local_safe_initial_cost(cost_fn, theta0);
    args.cache_hit        = false;

    result = build_result_struct(args);
    result.fval_final = J_final;

    % ---- 8. Postconditions --------------------------------------------------
    assert(result.solver == "fmincon", ...
        "fit_swing_fmincon:postSolver", ...
        "Postcondition: result.solver must be ""fmincon""");
    assert(isfinite(result.final_rmse_m) && result.final_rmse_m >= 0, ...
        "fit_swing_fmincon:postRmse", ...
        "Postcondition: final_rmse_m must be finite and non-negative");
    assert(numel(result.coefficients) == d, ...
        "fit_swing_fmincon:postSize", ...
        "Postcondition: coefficients length mismatch");
    assert(all(result.coefficients >= lb - 1e-9) && ...
           all(result.coefficients <= ub + 1e-9), ...
        "fit_swing_fmincon:postBounds", ...
        "Postcondition: coefficients must lie within [lb, ub]");
end

% =====================================================================
function n = local_n_joints(opts)
%LOCAL_N_JOINTS  Resolve joint count without loading the Simulink model.
    if isfield(opts, "sim") && isfield(opts.sim, "joint_names") && ...
            ~isempty(opts.sim.joint_names)
        n = numel(string(opts.sim.joint_names));
        return;
    end
    % Fall back to getPolynomialParameterInfo when available.
    try
        info = getPolynomialParameterInfo();
        n = numel(info.joint_names);
    catch
        n = 7;  % conservative default; tests should override via opts.sim.joint_names
    end
end

% =====================================================================
function theta0 = local_initial_theta(opts, lb, ub, d)
    if isfield(opts, "initial_theta") && ~isempty(opts.initial_theta)
        theta0 = double(opts.initial_theta(:));
        if numel(theta0) ~= d
            error("fit_swing_fmincon:badInitialTheta", ...
                "initial_theta has length %d but expected %d", ...
                numel(theta0), d);
        end
        if any(theta0 < lb) || any(theta0 > ub)
            error("fit_swing_fmincon:initialThetaOutOfBounds", ...
                "initial_theta has %d entries outside [lb, ub]", ...
                nnz(theta0 < lb) + nnz(theta0 > ub));
        end
        return;
    end
    rng(double(opts.rng_seed));
    u = rand(d, 1);
    theta0 = lb + u .* (ub - lb);
end

% =====================================================================
function sim_out = local_sim_adapter(theta, opts)
%LOCAL_SIM_ADAPTER  Call simulate_with_coefficients and re-key fields to the
%names compute_cost expects (butt/clubhead/club_quat).
    raw = simulate_with_coefficients(theta, opts.sim);
    sim_out = raw;
    if isfield(raw, "r_butt") && ~isfield(raw, "butt")
        sim_out.butt = raw.r_butt;
    end
    if isfield(raw, "r_clubhead") && ~isfield(raw, "clubhead")
        sim_out.clubhead = raw.r_clubhead;
    end
    if isfield(raw, "q_club") && ~isfield(raw, "club_quat")
        sim_out.club_quat = raw.q_club;
    end
end

% =====================================================================
function J = local_safe_cost(theta, target, sim_fn, opts)
%LOCAL_SAFE_COST  Evaluate compute_cost; on simulation failure return the
%configured penalty so fmincon can recover.
    try
        J = compute_cost(theta, target, sim_fn, opts.cost);
    catch ME
        if isfield(opts, "penalty_on_sim_failure")
            J = double(opts.penalty_on_sim_failure);
        else
            rethrow(ME);
        end
    end
end

% =====================================================================
function J0 = local_safe_initial_cost(cost_fn, theta0)
    try
        J0 = cost_fn(theta0);
    catch
        J0 = NaN;
    end
end

% =====================================================================
function tbl = local_new_history()
    tbl = table('Size', [0, 4], ...
        'VariableTypes', {'double','double','double','double'}, ...
        'VariableNames', {'iteration','fval','firstorderopt','stepsize'});
end

% =====================================================================
function v = local_get(s, name, default)
    if isfield(s, name)
        v = s.(name);
    else
        v = default;
    end
end

% =====================================================================
function s = local_struct_from_optimoptions(o)
%LOCAL_STRUCT_FROM_OPTIMOPTIONS  Best-effort conversion to a plain struct
%so the result file does not embed a MATLAB optim object.
    try
        s = struct();
        keys = ["Algorithm","Display","MaxIterations", ...
                "MaxFunctionEvaluations","OptimalityTolerance", ...
                "StepTolerance","FiniteDifferenceType"];
        for i = 1:numel(keys)
            try
                s.(keys(i)) = o.(keys(i));
            catch
            end
        end
    catch
        s = struct('summary', "optimoptions");
    end
end
