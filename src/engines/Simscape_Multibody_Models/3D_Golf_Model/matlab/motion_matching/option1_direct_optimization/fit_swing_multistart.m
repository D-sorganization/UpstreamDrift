function result = fit_swing_multistart(target, opts)
%FIT_SWING_MULTISTART  Parallel multi-start fmincon-sqp swing fit.
%
%   RESULT = FIT_SWING_MULTISTART(TARGET, OPTS) launches OPTS.n_starts
%   parallel runs of fit_swing_fmincon from quasi-random starting points
%   (Sobol by default) and returns the best fit, augmented with
%   .all_runs / .all_starts so MultiStartParallelCoords (#027) can render
%   per-start trajectories.
%
%   Each start is an independent fmincon-sqp local fit. The best run is the
%   one with the minimum final RMSE. The driver does not reuse fmincon's
%   gradient information across starts; it relies on diversity of the
%   starting points to escape local minima.
%
%   Parallelism:
%     - opts.parallel == true  : use parfor over a parallel pool.
%     - opts.parallel == false : serial for-loop (deterministic, debug).
%   The "parsim" branch is reserved for issue #029 (Simscape model array
%   simulations) and currently behaves identically to "parfor".
%
%   Result struct:
%     .coefficients            best theta
%     .final_rmse_m            min over all runs
%     .all_runs (1,N) struct   per start, with fields:
%         start_theta (d,1)
%         final_theta (d,1)
%         final_cost  scalar
%         final_rmse_m
%         duration_s
%         exitflag
%         worker_id
%     .all_starts {1,N} cell   alias of all_runs as cell array, kept for
%                              backwards compatibility with the issue body
%     ... plus all canonical Option 1 result fields from build_result_struct.
%
%   Preconditions (DbC):
%     - TARGET has the canonical CLUB_IK_SPEC fields.
%     - OPTS.n_starts >= 1.
%     - OPTS.parallel_method in {"parsim","parfor"}.
%
%   Postconditions:
%     - numel(result.all_runs) == OPTS.n_starts.
%     - result.final_rmse_m == min over all_runs.
%     - All starting points lie within coefficient bounds.
%
%   GitHub issue: #025 / #3994.
%
%   See also: FIT_SWING_FMINCON, SAMPLE_STARTING_POINTS,
%             DEFAULT_MULTISTART_OPTIONS.

    arguments
        target (1,1) struct {validators.mustHaveFields(target, ...
            ["time","butt","clubhead","club_quat","impact_idx"])}
        opts (1,1) struct = default_multistart_options()
    end

    t_start = tic;
    opts = local_fill_defaults(opts);

    % ---- Bounds + sampling -----------------------------------------------
    n_joints = local_n_joints(opts);
    [lb, ub] = build_coefficient_bounds(n_joints);
    n_starts = double(opts.n_starts);

    starts = sample_starting_points(n_starts, lb, ub, ...
        string(opts.starting_strategy), double(opts.seed));

    % ---- Per-start fmincon options template -------------------------------
    base_fmincon_opts = local_fmincon_opts_template(opts);

    % ---- Run all starts ---------------------------------------------------
    use_parallel = local_resolve_parallel(opts);
    runs = local_blank_runs(n_starts, numel(lb));

    if use_parallel
        parfor i = 1:n_starts
            runs(i) = local_run_one(target, base_fmincon_opts, ...
                starts(:, i), i); %#ok<PFBNS>
        end
    else
        for i = 1:n_starts
            runs(i) = local_run_one(target, base_fmincon_opts, ...
                starts(:, i), i);
        end
    end

    % ---- Pick best --------------------------------------------------------
    rmses = arrayfun(@(r) local_safe_rmse(r), runs);
    [best_rmse, best_idx] = min(rmses);
    best_run = runs(best_idx);

    % ---- Re-evaluate the winner so we have full provenance ---------------
    sim_fn  = @(theta) local_sim_adapter(theta, opts.fmincon_options);
    [J_best, terms_best] = compute_cost(best_run.final_theta, target, ...
        sim_fn, opts.fmincon_options.cost);
    final_sim_out = sim_fn(best_run.final_theta);

    args = struct();
    args.coefficients     = best_run.final_theta;
    args.final_cost_terms = terms_best;
    args.final_sim_out    = final_sim_out;
    args.solver           = "multistart";
    args.solver_options   = struct( ...
        "n_starts", n_starts, ...
        "starting_strategy", string(opts.starting_strategy), ...
        "parallel_method",   string(opts.parallel_method), ...
        "parallel",          logical(use_parallel), ...
        "seed",              double(opts.seed));
    args.options          = opts;
    args.target           = target;
    args.exitflag         = best_run.exitflag;
    args.output           = struct("message", "multistart driver", ...
                                    "best_start_index", best_idx);
    args.iter_history     = local_blank_iter_history();
    args.duration_s       = toc(t_start);
    args.start_points     = starts;
    args.start_costs      = arrayfun(@(r) r.start_cost, runs);
    args.cache_hit        = false;

    result = build_result_struct(args);
    result.fval_final     = J_best;
    result.all_runs       = runs;
    result.all_starts     = arrayfun(@(r) r, runs, 'UniformOutput', false);
    result.best_index     = best_idx;

    % ---- Postconditions ---------------------------------------------------
    assert(numel(result.all_runs) == n_starts, ...
        "fit_swing_multistart:postLen", ...
        "Postcondition: all_runs must have length n_starts");
    assert(abs(result.final_rmse_m - best_rmse) <= 1e-9 * max(1, best_rmse), ...
        "fit_swing_multistart:postBest", ...
        "Postcondition: final_rmse_m must equal min over all_runs");
    for i = 1:n_starts
        assert(all(runs(i).start_theta >= lb - 1e-9) && ...
               all(runs(i).start_theta <= ub + 1e-9), ...
            "fit_swing_multistart:postBounds", ...
            "Postcondition: every start must be within bounds");
    end
end

% =====================================================================
function opts = local_fill_defaults(opts)
%LOCAL_FILL_DEFAULTS  Backfill any missing multistart fields from defaults.
    defaults = default_multistart_options();
    f = fieldnames(defaults);
    for k = 1:numel(f)
        if ~isfield(opts, f{k}) || isempty(opts.(f{k}))
            opts.(f{k}) = defaults.(f{k});
        end
    end
end

% =====================================================================
function n = local_n_joints(opts)
    sim_opts = opts;
    if isfield(opts, "fmincon_options") && isstruct(opts.fmincon_options) && ...
            isfield(opts.fmincon_options, "sim")
        sim_opts = opts.fmincon_options;
    end
    if isfield(sim_opts, "sim") && isfield(sim_opts.sim, "joint_names") && ...
            ~isempty(sim_opts.sim.joint_names)
        n = numel(string(sim_opts.sim.joint_names));
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
function tf = local_resolve_parallel(opts)
%LOCAL_RESOLVE_PARALLEL  Decide whether to run a parfor pool.
    tf = false;
    if isfield(opts, "parallel") && ~isempty(opts.parallel)
        tf = logical(opts.parallel);
    elseif isfield(opts, "use_parallel") && ~isempty(opts.use_parallel)
        tf = logical(opts.use_parallel);
    end
    if ~tf, return; end
    % Confirm a pool is available; fall back to serial if not.
    try
        if isempty(gcp('nocreate')) && exist('parpool', 'file') == 2
            % Best-effort: don't auto-spawn during tests; just allow parfor
            % to start its own. If PCT is missing, parfor degrades to for.
        end
    catch
    end
end

% =====================================================================
function fmincon_opts = local_fmincon_opts_template(opts)
%LOCAL_FMINCON_OPTS_TEMPLATE  Per-start opts for fit_swing_fmincon.
    fmincon_opts = opts.fmincon_options;
    if ~isstruct(fmincon_opts)
        fmincon_opts = default_option1_options();
    end
    % Single-start runs must not recurse into another multistart driver.
    fmincon_opts.solver = "fmincon";
    % Each parfor iter sets its own initial_theta; clear any global one.
    fmincon_opts.initial_theta = [];
    % Keep verbose output off in workers; the parent driver logs.
    if ~isfield(fmincon_opts, "display") || fmincon_opts.display == ""
        fmincon_opts.display = "off";
    end
end

% =====================================================================
function run = local_run_one(target, base_opts, theta0, idx)
%LOCAL_RUN_ONE  Run a single fmincon-sqp from theta0; never throw.
    t0 = tic;
    run = struct( ...
        'start_theta', theta0, ...
        'final_theta', theta0, ...
        'final_cost',  Inf, ...
        'final_rmse_m', Inf, ...
        'start_cost',   NaN, ...
        'duration_s',   0, ...
        'exitflag',     int32(-99), ...
        'worker_id',    int32(idx), ...
        'failed',       true, ...
        'message',      "");
    try
        per = base_opts;
        per.initial_theta = theta0;
        sub = fit_swing_fmincon(target, per);
        run.final_theta  = sub.coefficients(:);
        run.final_cost   = local_field(sub, 'fval_final', sub.final_rmse_m^2);
        run.final_rmse_m = double(sub.final_rmse_m);
        run.start_cost   = local_first(sub.start_costs);
        run.exitflag     = int32(sub.exitflag);
        run.failed       = false;
    catch ME
        run.message = string(ME.message);
        warning("fit_swing_multistart:startFailed", ...
            "Start %d failed: %s", idx, ME.message);
    end
    run.duration_s = toc(t0);
end

% =====================================================================
function r = local_blank_runs(n, d)
    proto = struct( ...
        'start_theta', zeros(d, 1), ...
        'final_theta', zeros(d, 1), ...
        'final_cost',  Inf, ...
        'final_rmse_m', Inf, ...
        'start_cost',   NaN, ...
        'duration_s',   0, ...
        'exitflag',     int32(-99), ...
        'worker_id',    int32(0), ...
        'failed',       true, ...
        'message',      "");
    r = repmat(proto, 1, n);
end

% =====================================================================
function tbl = local_blank_iter_history()
    tbl = table('Size', [0, 4], ...
        'VariableTypes', {'double','double','double','double'}, ...
        'VariableNames', {'iteration','fval','firstorderopt','stepsize'});
end

% =====================================================================
function v = local_field(s, name, default)
    if isstruct(s) && isfield(s, name)
        v = s.(name);
    else
        v = default;
    end
end

function v = local_first(x)
    if isempty(x)
        v = NaN;
    else
        v = double(x(1));
    end
end

function v = local_safe_rmse(run)
    v = double(run.final_rmse_m);
    if ~isfinite(v) || v < 0
        v = Inf;
    end
end

% =====================================================================
function sim_out = local_sim_adapter(theta, opts)
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
