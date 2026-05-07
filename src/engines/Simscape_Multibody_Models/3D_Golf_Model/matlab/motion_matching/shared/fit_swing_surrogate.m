function result = fit_swing_surrogate(target, opts)
%FIT_SWING_SURROGATE  Option-2 NN-surrogate-based swing fit (#4075).
%
%   RESULT = FIT_SWING_SURROGATE(TARGET, OPTS) optimises polynomial torque
%   coefficients THETA against TARGET using a trained PyTorch surrogate as
%   the forward model -- *not* a fresh Simscape simulation per fmincon
%   evaluation. The surrogate maps a 189-vec of polynomial coefficients to
%   the resulting clubhead/grip trajectory in milliseconds.
%
%   The function is a thin orchestrator:
%     1. Resolves the surrogate-checkpoint path (OPTS.surrogate_checkpoint
%        or the default ``output/surrogate/checkpoint_best.pt`` on the repo
%        root).
%     2. Builds an initial guess THETA0 from OPTS.initial_theta when given,
%        otherwise zeros (the trained surrogate is most reliable inside its
%        training distribution and the dataset is centred near zero after
%        normalisation).
%     3. Wraps a Python predictor in a MATLAB function handle via
%        ``pyrunfile`` so each fmincon evaluation is a sub-millisecond NN
%        forward pass.
%     4. Runs fmincon('algorithm','sqp') against the surrogate's predicted
%        clubhead/grip trajectory.
%     5. Assembles the canonical RESULT struct used elsewhere in
%        motion_matching/ -- same field set as the Option 1 fit, with
%        ``solver = "swing_surrogate"``.
%
%   TARGET is the canonical struct from shared/CLUB_IK_SPEC.md.
%   OPTS is a 1x1 struct with optional fields:
%       surrogate_checkpoint  string  Path to the trained .pt checkpoint.
%       initial_theta         double  (189,1) starting coefficients.
%       n_joints              double  Defaults to 27 (compact-schema).
%       coeffs_per_joint      double  Defaults to 7.
%       max_iter              double  fmincon MaxIterations (default 200).
%       grip_weight           double  Weight on grip-RMSE term (default 1.0).
%       clubhead_weight       double  Weight on clubhead-RMSE term (default 1.0).
%       impact_weight         double  Weight on clubhead-speed-at-impact (1.0).
%       python_executable     string  Override for the python interpreter.
%
%   Preconditions (DbC, validated by `arguments`):
%     - TARGET has the required fields {time, butt|r_grip, clubhead}.
%     - OPTS is a 1x1 struct.
%     - The Python surrogate package is importable in the active pyenv.
%
%   Postconditions:
%     - result.solver == "swing_surrogate".
%     - 0 <= result.final_rmse_m < Inf.
%     - result.coefficients has the documented length (n_joints * 7).
%
%   GitHub issue: #4075.
%
%   See also: FIT_SWING_FMINCON, COMPUTE_COST,
%             src/shared/python/motion_matching/surrogate/compact/

    arguments
        target (1,1) struct
        opts (1,1) struct = struct()
    end

    t_start = tic;

    opts = local_apply_defaults(opts);
    surrogate_checkpoint = local_resolve_checkpoint(opts);
    n_joints = opts.n_joints;
    coeffs_per_joint = opts.coeffs_per_joint;
    coeff_dim = n_joints * coeffs_per_joint;

    % ---- 1. Initial guess --------------------------------------------------
    theta0 = local_initial_theta(opts, coeff_dim);

    % ---- 2. Build the Python predictor closure -----------------------------
    %        Each call hits a cached SwingSurrogate inside the Python pyenv,
    %        so the per-evaluation cost is dominated by the NN forward pass
    %        (typically 2-5 ms on CPU).
    predictor = local_make_python_predictor(surrogate_checkpoint, opts);

    target_traj = local_extract_target_trajectory(target);

    % ---- 3. fmincon options ------------------------------------------------
    [lb, ub] = local_default_bounds(n_joints);
    cost_fn = @(theta) local_surrogate_cost(theta, predictor, target_traj, opts);
    fmincon_opts = optimoptions('fmincon', ...
        'Algorithm', 'sqp', ...
        'Display', 'none', ...
        'MaxIterations', opts.max_iter, ...
        'MaxFunctionEvaluations', 50 * coeff_dim, ...
        'StepTolerance', 1e-8, ...
        'OptimalityTolerance', 1e-6, ...
        'SpecifyObjectiveGradient', false);

    [theta_star, fval, exitflag, output] = fmincon( ...
        cost_fn, theta0, [], [], [], [], lb, ub, [], fmincon_opts);

    % ---- 4. Final eval + RMSE breakdown ------------------------------------
    final_pred = predictor(theta_star);
    rmse_grip = local_rmse(final_pred.r_grip, target_traj.r_grip);
    rmse_ch   = local_rmse(final_pred.r_clubhead, target_traj.r_clubhead);
    final_rmse_m = sqrt(rmse_grip^2 + rmse_ch^2);

    % ---- 5. Assemble result struct -----------------------------------------
    result = struct();
    result.coefficients       = theta_star(:);
    result.final_rmse_m       = final_rmse_m;
    result.final_grip_rmse_m  = rmse_grip;
    result.final_ch_rmse_m    = rmse_ch;
    result.final_cost_terms   = struct( ...
        'cost', double(fval), ...
        'grip_rmse_m', rmse_grip, ...
        'clubhead_rmse_m', rmse_ch);
    result.final_total_work_J = NaN;  % Surrogate doesn't predict work.
    result.solver             = "swing_surrogate";
    result.solver_options     = fmincon_opts;
    result.surrogate_checkpoint = string(surrogate_checkpoint);
    result.target_hash        = local_hash_target(target);
    result.git_commit         = local_safe_git_commit();
    result.matlab_version     = string(version);
    result.duration_s         = toc(t_start);
    result.timestamp_utc      = string(datetime("now","TimeZone","UTC", ...
                                  "Format","yyyy-MM-dd'T'HH:mm:ss'Z'"));
    result.iter_history       = table();
    result.exitflag           = exitflag;
    result.output             = output;
    result.start_points       = theta0(:);
    result.start_costs        = double(cost_fn(theta0));
    result.cache_hit          = false;
    result.options            = opts;

    assert(isfinite(result.final_rmse_m) && result.final_rmse_m >= 0, ...
        "fit_swing_surrogate:badRmse", ...
        "Postcondition: final_rmse_m must be finite and non-negative");
end


% ========================================================================= %
% Helpers                                                                   %
% ========================================================================= %


function opts = local_apply_defaults(opts)
    if ~isfield(opts, 'n_joints'),         opts.n_joints = 27;          end
    if ~isfield(opts, 'coeffs_per_joint'), opts.coeffs_per_joint = 7;   end
    if ~isfield(opts, 'max_iter'),         opts.max_iter = 200;         end
    if ~isfield(opts, 'grip_weight'),      opts.grip_weight = 1.0;      end
    if ~isfield(opts, 'clubhead_weight'),  opts.clubhead_weight = 1.0;  end
    if ~isfield(opts, 'impact_weight'),    opts.impact_weight = 1.0;    end
    if ~isfield(opts, 'python_executable'), opts.python_executable = ""; end
end


function path = local_resolve_checkpoint(opts)
    if isfield(opts, 'surrogate_checkpoint') && ...
            ~isempty(opts.surrogate_checkpoint) && ...
            strlength(string(opts.surrogate_checkpoint)) > 0
        path = char(opts.surrogate_checkpoint);
    else
        % Default: <repo_root>/output/surrogate/checkpoint_best.pt
        here = fileparts(mfilename('fullpath'));
        repo_root = fullfile(here, '..', '..', '..', '..', '..', '..', '..');
        path = fullfile(repo_root, 'output', 'surrogate', 'checkpoint_best.pt');
    end
    assert(exist(path, 'file') == 2, ...
        "fit_swing_surrogate:missingCheckpoint", ...
        "Surrogate checkpoint not found: %s", path);
end


function theta0 = local_initial_theta(opts, coeff_dim)
    if isfield(opts, 'initial_theta') && ~isempty(opts.initial_theta)
        theta0 = double(opts.initial_theta(:));
        assert(numel(theta0) == coeff_dim, ...
            "fit_swing_surrogate:badInitial", ...
            "initial_theta length %d != expected %d", ...
            numel(theta0), coeff_dim);
    else
        theta0 = zeros(coeff_dim, 1);
    end
end


function predictor = local_make_python_predictor(checkpoint_path, opts)
%LOCAL_MAKE_PYTHON_PREDICTOR  Build a function handle that runs the surrogate.
%
%   The handle is ``predictor(theta)`` where THETA is a (D,1) double; the
%   return value is a struct with fields {r_clubhead, v_clubhead, r_grip,
%   clubhead_speed} as ``T x 3`` (or ``T x 1``) MATLAB doubles.
    if strlength(string(opts.python_executable)) > 0
        % Best-effort pyenv switch -- silently noop if already configured.
        try
            pyenv('Version', char(opts.python_executable));
        catch
            % pyenv is process-global; ignore failures.
        end
    end

    % We use ``pyrun`` with persistent state so the surrogate stays in
    % memory between fmincon iterations.
    pyrun_setup = sprintf([ ...
        "import torch\n", ...
        "from src.shared.python.motion_matching.surrogate.compact import (\n", ...
        "    SwingSurrogate, predict_trajectory)\n", ...
        "_surrogate_model = SwingSurrogate.from_checkpoint(r'''%s''')\n"], ...
        checkpoint_path);
    pyrun(pyrun_setup);

    predictor = @(theta) local_invoke_python(theta);
end


function out = local_invoke_python(theta)
%LOCAL_INVOKE_PYTHON  Run the surrogate once for a single (D,1) coefficient vec.
    theta_py = py.numpy.asarray(double(theta(:)).');
    res = pyrun( ...
        "result = predict_trajectory(_surrogate_model, theta)", ...
        "result", ...
        "theta", theta_py);
    out = struct();
    out.r_clubhead     = double(res{'r_clubhead'});
    out.v_clubhead     = double(res{'v_clubhead'});
    out.r_grip         = double(res{'r_grip'});
    out.clubhead_speed = double(res{'clubhead_speed'});
    if isfield(res, 'shaft_axis')
        out.shaft_axis = double(res{'shaft_axis'});
    end
    % predict_trajectory returns (B=1, T, k); squeeze leading batch dim.
    fns = fieldnames(out);
    for ii = 1:numel(fns)
        v = out.(fns{ii});
        if size(v, 1) == 1 && ndims(v) >= 2
            out.(fns{ii}) = squeeze(v);
        end
    end
end


function tgt = local_extract_target_trajectory(target)
%LOCAL_EXTRACT_TARGET_TRAJECTORY  Pull the channels the surrogate predicts.
    tgt = struct();
    if isfield(target, 'r_grip')
        tgt.r_grip = double(target.r_grip);
    elseif isfield(target, 'butt')
        tgt.r_grip = double(target.butt);  % butt-end is grip-equivalent.
    else
        error("fit_swing_surrogate:noGrip", ...
              "TARGET must contain r_grip (or butt as fallback).");
    end
    if isfield(target, 'r_clubhead')
        tgt.r_clubhead = double(target.r_clubhead);
    elseif isfield(target, 'clubhead')
        tgt.r_clubhead = double(target.clubhead);
    else
        error("fit_swing_surrogate:noCH", ...
              "TARGET must contain clubhead (or r_clubhead).");
    end
end


function cost = local_surrogate_cost(theta, predictor, tgt, opts)
%LOCAL_SURROGATE_COST  Weighted RMSE on grip/clubhead + impact-speed term.
    pred = predictor(theta);
    grip_diff = pred.r_grip - tgt.r_grip;
    ch_diff   = pred.r_clubhead - tgt.r_clubhead;
    grip_rmse = sqrt(mean(grip_diff(:).^2));
    ch_rmse   = sqrt(mean(ch_diff(:).^2));
    cost = opts.grip_weight * grip_rmse^2 + opts.clubhead_weight * ch_rmse^2;
    if isfield(tgt, 'clubhead_speed') && isfield(pred, 'clubhead_speed')
        speed_diff = pred.clubhead_speed - tgt.clubhead_speed;
        cost = cost + opts.impact_weight * mean(speed_diff(:).^2);
    end
end


function [lb, ub] = local_default_bounds(n_joints)
%LOCAL_DEFAULT_BOUNDS  Per-letter bounds from PROJECT_SPEC.md §4, tiled.
    bounds_per_letter = [1000; 1000; 500; 500; 100; 100; 25];
    ub = repmat(bounds_per_letter, n_joints, 1);
    lb = -ub;
end


function rmse = local_rmse(a, b)
    rmse = sqrt(mean((a(:) - b(:)).^2));
end


function h = local_hash_target(target)
    try
        h = string(matlab.lang.makeValidName( ...
            sprintf('%g_%g', sum(target.r_grip(:), 'omitnan'), ...
                    sum(target.r_clubhead(:), 'omitnan'))));
    catch
        h = "unknown";
    end
end


function commit = local_safe_git_commit()
    try
        [status, out] = system('git rev-parse --short HEAD');
        if status == 0
            commit = strtrim(string(out));
        else
            commit = "unknown";
        end
    catch
        commit = "unknown";
    end
end
