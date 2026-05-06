function result = fit_swing_surrogateopt(target, opts)
%FIT_SWING_SURROGATEOPT  Hybrid surrogateopt + fmincon-sqp polish swing fit.
%
%   RESULT = FIT_SWING_SURROGATEOPT(TARGET, OPTS) runs MATLAB's
%   surrogateopt (Global Optimization Toolbox) over the bounded coefficient
%   space for OPTS.surrogate_max_evals function evaluations, then
%   warm-starts fit_swing_fmincon from the surrogate's best incumbent for
%   final SQP polish. The returned RESULT is the polished result and
%   includes the surrogate phase's iteration history under
%   RESULT.surrogateopt_history.
%
%   The combined wall-clock budget is bounded by OPTS.max_wall_seconds; if
%   the surrogate phase exceeds this, the polish is skipped and the
%   surrogate-best is returned.
%
%   Preconditions (DbC, validated by `arguments`):
%     - TARGET has fields {time, butt, clubhead, club_quat, impact_idx}.
%     - OPTS is a 1x1 struct.
%
%   Postconditions:
%     - result.solver == "surrogateopt+fmincon" (or "surrogateopt" if the
%       polish was skipped).
%     - result.final_rmse_m <= surrogate-phase final_rmse_m, or equal
%       (polish never strictly worsens the answer).
%     - result.surrogateopt_history is a table.
%
%   GitHub issue: #026 / #3995.
%
%   See also: FIT_SWING_FMINCON, BUILD_SURROGATEOPT_OPTIONS,
%             BUILD_COEFFICIENT_BOUNDS, COMPUTE_COST.
    arguments
        target (1,1) struct {validators.mustHaveFields(target, ...
            ["time","butt","clubhead","club_quat","impact_idx"])}
        opts   (1,1) struct = local_default_opts()
    end

    opts = local_fill_defaults(opts);
    t_start = tic;

    % ---- 1. Bounds & dimension --------------------------------------------
    n_joints = local_n_joints(opts);
    [lb, ub] = build_coefficient_bounds(n_joints);
    d = numel(lb);

    % ---- 2. Sim and cost closures -----------------------------------------
    sim_fn  = local_sim_fn(opts);
    cost_fn = @(theta) local_safe_cost(theta, target, sim_fn, opts);

    % ---- 3. surrogateopt OutputFcn captures iterates ----------------------
    history = local_new_history();
    function [stop, optnew, optchanged] = capture_so(optimValues, optnew_in, state)
        stop = false;
        optnew = optnew_in;
        optchanged = false;
        if strcmp(state, "iter")
            try
                row = table( ...
                    double(local_get(optimValues, 'funccount', NaN)), ...
                    double(local_get(optimValues, 'fval', NaN)), ...
                    double(local_get(optimValues, 'currentFlag', NaN)), ...
                    'VariableNames', {'funccount','fval','currentFlag'});
                history = [history; row]; %#ok<AGROW>
            catch
                % Never abort the solve because of a logging hiccup.
            end
        end
        if toc(t_start) > opts.max_wall_seconds
            stop = true;
        end
    end

    % ---- 4. Run surrogateopt ----------------------------------------------
    so_opts = build_surrogateopt_options(opts, @capture_so);
    problem = struct();
    problem.objective = @(x) cost_fn(x(:));
    problem.lb        = lb(:)';
    problem.ub        = ub(:)';
    problem.solver    = 'surrogateopt';
    problem.options   = so_opts;

    [theta_global, fval_global, exitflag_so, output_so] = surrogateopt(problem);
    theta_global = theta_global(:);
    surrogate_elapsed = toc(t_start);

    % Capture the surrogate-phase result struct (so we can compare).
    surrogate_sim_out = local_safe_sim(theta_global, sim_fn);
    [~, surrogate_terms] = local_safe_compute_cost(theta_global, target, sim_fn, opts);

    surrogate_args = struct();
    surrogate_args.coefficients     = theta_global;
    surrogate_args.final_cost_terms = surrogate_terms;
    surrogate_args.final_sim_out    = surrogate_sim_out;
    surrogate_args.solver           = "surrogateopt";
    surrogate_args.solver_options   = local_struct_from_optimoptions(so_opts);
    surrogate_args.options          = opts;
    surrogate_args.target           = target;
    surrogate_args.exitflag         = exitflag_so;
    surrogate_args.output           = output_so;
    surrogate_args.iter_history     = history;
    surrogate_args.duration_s       = surrogate_elapsed;
    surrogate_args.start_points     = [];
    surrogate_args.start_costs      = [];
    surrogate_args.cache_hit        = false;
    surrogate_result = build_result_struct(surrogate_args);
    surrogate_result.fval_final = fval_global;

    % ---- 5. Polish phase via fit_swing_fmincon ----------------------------
    skip_polish = local_get_field(opts, "skip_polish", false) || ...
                  surrogate_elapsed >= opts.max_wall_seconds;

    if skip_polish
        result = surrogate_result;
        result.solver = "surrogateopt";
        result.surrogateopt_history = history;
        result.surrogateopt_phase   = surrogate_result;
        result.duration_s = toc(t_start);
        return;
    end

    polish_opts = local_polish_opts(opts, theta_global, lb, ub);
    try
        polished = fit_swing_fmincon(target, polish_opts);
    catch ME
        warning("fit_swing_surrogateopt:polishFailed", ...
            "Polish phase failed (%s); returning surrogate-best.", ME.message);
        result = surrogate_result;
        result.solver = "surrogateopt";
        result.surrogateopt_history = history;
        result.surrogateopt_phase   = surrogate_result;
        result.duration_s = toc(t_start);
        return;
    end

    % ---- 6. Pick best of the two (polish should be <= surrogate) ----------
    if isfinite(polished.final_rmse_m) && ...
            polished.final_rmse_m <= surrogate_result.final_rmse_m + 1e-12
        result = polished;
    else
        % Polish made things worse (numerical edge); fall back to surrogate.
        result = surrogate_result;
    end

    result.solver               = "surrogateopt+fmincon";
    result.surrogateopt_history = history;
    result.surrogateopt_phase   = surrogate_result;
    result.duration_s           = toc(t_start);

    % ---- 7. Postconditions ------------------------------------------------
    assert(result.solver == "surrogateopt+fmincon" || ...
           result.solver == "surrogateopt", ...
        "fit_swing_surrogateopt:postSolver", ...
        "Postcondition: solver tag must be surrogateopt or surrogateopt+fmincon");
    assert(result.final_rmse_m <= surrogate_result.final_rmse_m + 1e-12, ...
        "fit_swing_surrogateopt:postPolishImproves", ...
        "Postcondition: polish must not strictly worsen surrogate-best RMSE");
    assert(istable(result.surrogateopt_history), ...
        "fit_swing_surrogateopt:postHistory", ...
        "Postcondition: surrogateopt_history must be a table");
    assert(numel(result.coefficients) == d, ...
        "fit_swing_surrogateopt:postSize", ...
        "Postcondition: coefficient length mismatch");
end

% =====================================================================
function opts = local_default_opts()
    opts = default_option1_options();
    opts = local_fill_defaults(opts);
end

% =====================================================================
function opts = local_fill_defaults(opts)
%LOCAL_FILL_DEFAULTS  Apply surrogateopt-specific defaults if missing.
    if ~isfield(opts, "surrogate_max_evals") || isempty(opts.surrogate_max_evals)
        opts.surrogate_max_evals = 1500;
    end
    if ~isfield(opts, "polish_max_iter") || isempty(opts.polish_max_iter)
        opts.polish_max_iter = 200;
    end
    if ~isfield(opts, "max_wall_seconds") || isempty(opts.max_wall_seconds)
        opts.max_wall_seconds = 600;
    end
    if ~isfield(opts, "min_surrogate_points")
        opts.min_surrogate_points = 50;
    end
    if ~isfield(opts, "min_sample_distance")
        opts.min_sample_distance = 1e-3;
    end
    if ~isfield(opts, "use_parallel")
        opts.use_parallel = false;
    end
    if ~isfield(opts, "skip_polish")
        opts.skip_polish = false;
    end
    if ~isfield(opts, "sim")
        opts.sim = default_sim_options();
    end
    if ~isfield(opts, "cost")
        opts.cost = default_cost_options();
    end
    if ~isfield(opts, "rng_seed")
        opts.rng_seed = uint32(42);
    end
    if ~isfield(opts, "penalty_on_sim_failure")
        opts.penalty_on_sim_failure = 1e9;
    end
end

% =====================================================================
function n = local_n_joints(opts)
    if isfield(opts, "sim") && isfield(opts.sim, "joint_names") && ...
            ~isempty(opts.sim.joint_names)
        n = numel(string(opts.sim.joint_names));
        return;
    end
    try
        info = getPolynomialParameterInfo();
        n = numel(info.joint_names);
    catch
        n = 7;
    end
end

% =====================================================================
function fn = local_sim_fn(opts)
    if isfield(opts, "sim_fn") && ~isempty(opts.sim_fn) && ...
            isa(opts.sim_fn, 'function_handle')
        fn = opts.sim_fn;
    else
        fn = @(theta) local_default_sim_adapter(theta, opts);
    end
end

% =====================================================================
function sim_out = local_default_sim_adapter(theta, opts)
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
%LOCAL_SAFE_COST  Cost function that converts simulation failures into a
%finite (or Inf) penalty so surrogateopt continues exploring.
    try
        J = compute_cost(theta, target, sim_fn, opts.cost);
        if ~isfinite(J)
            J = local_failure_penalty(opts);
        end
    catch
        J = local_failure_penalty(opts);
    end
end

function pen = local_failure_penalty(opts)
    if isfield(opts, "penalty_on_sim_failure") && ...
            isfinite(opts.penalty_on_sim_failure)
        pen = double(opts.penalty_on_sim_failure);
    else
        pen = Inf;
    end
end

% =====================================================================
function sim_out = local_safe_sim(theta, sim_fn)
    try
        sim_out = sim_fn(theta);
    catch
        sim_out = struct();
    end
end

% =====================================================================
function [J, terms] = local_safe_compute_cost(theta, target, sim_fn, opts)
    try
        [J, terms] = compute_cost(theta, target, sim_fn, opts.cost);
    catch
        J = local_failure_penalty(opts);
        terms = struct('position', NaN);
    end
end

% =====================================================================
function tbl = local_new_history()
    tbl = table('Size', [0, 3], ...
        'VariableTypes', {'double','double','double'}, ...
        'VariableNames', {'funccount','fval','currentFlag'});
end

% =====================================================================
function v = local_get(s, name, default)
    if isstruct(s) && isfield(s, name)
        v = s.(name);
    else
        v = default;
    end
end

% =====================================================================
function v = local_get_field(s, name, default)
    if isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = default;
    end
end

% =====================================================================
function polish_opts = local_polish_opts(opts, theta_warm, lb, ub)
%LOCAL_POLISH_OPTS  Build options for fit_swing_fmincon polish phase.
    polish_opts = opts;
    % Clamp warm start strictly inside bounds for fmincon's bounds check.
    theta_warm = max(min(theta_warm, ub), lb);
    polish_opts.initial_theta      = theta_warm;
    polish_opts.max_iter           = uint32(opts.polish_max_iter);
    polish_opts.max_iterations     = uint32(opts.polish_max_iter);
    polish_opts.algorithm          = "sqp";
    polish_opts.display            = "off";
    if ~isfield(polish_opts, "max_function_evals") || ...
            isempty(polish_opts.max_function_evals)
        polish_opts.max_function_evals = uint32(opts.polish_max_iter * 50);
    end
end

% =====================================================================
function s = local_struct_from_optimoptions(o)
    try
        s = struct();
        keys = ["MaxFunctionEvaluations","UseParallel","PlotFcn", ...
                "MinSurrogatePoints","MinSampleDistance"];
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
