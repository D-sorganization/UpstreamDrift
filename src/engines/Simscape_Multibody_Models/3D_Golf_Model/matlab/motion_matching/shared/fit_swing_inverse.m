function result = fit_swing_inverse(target, opts)
%FIT_SWING_INVERSE  Option-3 cVAE inverse fit of polynomial coefficients.
%
%   RESULT = FIT_SWING_INVERSE(TARGET, OPTS) draws N candidate coefficient
%   vectors from a trained ``SwingInverseCVAE`` (GH issue #4076) conditioned
%   on the measured TARGET trajectory and returns the best one. The model
%   is the conditional VAE described in
%   ``shared/python/motion_matching/inverse/cvae.py`` (1-D conv encoder,
%   189-dim coefficient decoder, latent dim 32).
%
%   TARGET is the canonical struct from ``load_club_target_excel`` — at
%   minimum it must carry ``{time, butt, clubhead, club_quat}``. The
%   12-channel trajectory the cVAE consumes is built by stacking
%   ``[butt(:,1:3), clubhead(:,1:3), grip(:,1:3), v_clubhead(:,1:3)]``
%   matching ``training.py::_build_trajectory``.
%
%   OPTS is a struct with the following fields (all optional unless noted):
%       checkpoint_path  (1,1) string  REQUIRED. Path to a SwingInverseCVAE
%                                       checkpoint produced by
%                                       ``train_inverse_cvae``.
%       n_samples        (1,1) double  Default 8. Posterior draws.
%       seed             (1,1) double  Default 0. RNG seed for sampling.
%       resample_T       (1,1) double  Default 32. Number of timesteps to
%                                       resample the target onto.
%       rerank           (1,1) logical Default false. When true, run each
%                                       sample through OPTS.forward_fn (a
%                                       function handle) and return the one
%                                       with the smallest grip-RMSE.
%       forward_fn       function_handle  Optional. Used only when rerank
%                                       is true.
%
%   RESULT is a struct with fields:
%       coefficients     (189,1) double — best (or mean) coefficient vector.
%       samples          (n_samples, 189) double — every drawn sample.
%       sample_index     scalar double — index of the chosen sample (NaN
%                        when rerank is off and the prior mean is used).
%       solver           string "swing_inverse_cvae"
%       final_rmse_m     scalar double — RMSE from the ranking pass, NaN
%                        when no rerank was performed.
%       checkpoint_path  string — copy of OPTS.checkpoint_path.
%       n_samples        scalar double — copy of OPTS.n_samples.
%       duration_s       scalar double — wall-clock time taken.
%       timestamp_utc    string — ISO-8601 UTC timestamp.
%
%   The shim calls Python via ``pyrunfile`` so MATLAB does not need to
%   import ``torch`` directly. The Python entry point is
%   ``shared/python/motion_matching/inverse/predict.py::predict_coefficients_from_checkpoint``.
%
%   See also LOAD_CLUB_TARGET_EXCEL, FIT_SWING_FMINCON, FIT_SWING_SURROGATE.
    arguments
        target (1,1) struct
        opts   (1,1) struct
    end

    t_start = tic;

    if ~isfield(opts, "checkpoint_path") || strlength(string(opts.checkpoint_path)) == 0
        error("fit_swing_inverse:missingCheckpoint", ...
              "OPTS.checkpoint_path is required");
    end

    n_samples = local_field_or(opts, "n_samples", 8);
    seed       = local_field_or(opts, "seed", 0);
    resample_T = local_field_or(opts, "resample_T", 32);
    rerank     = logical(local_field_or(opts, "rerank", false));

    % ---- 1. Build the 12-channel trajectory ------------------------------
    traj = local_build_trajectory(target, resample_T);

    % ---- 2. Call Python predict_coefficients_from_checkpoint -------------
    %        Ensure the repo root is on sys.path so the import resolves
    %        regardless of MATLAB's cwd. The shim lives 7 levels under
    %        the repo root (src/engines/.../matlab/motion_matching/shared).
    here = fileparts(mfilename('fullpath'));
    repo_root = fullfile(here, '..', '..', '..', '..', '..', '..', '..');
    repo_root_str = char(string(py.os.path.abspath(repo_root)));
    sys_path = py.sys.path;
    if ~any(cellfun(@(p) strcmp(char(p), repo_root_str), cell(sys_path)))
        insert(sys_path, int32(0), repo_root_str);
    end
    pred = py.importlib.import_module( ...
        "src.shared.python.motion_matching.inverse.predict" ...
    ).predict_coefficients_from_checkpoint( ...
        string(opts.checkpoint_path), py.numpy.asarray(traj), ...
        pyargs("n_samples", int64(n_samples), "seed", int64(seed)));

    samples = double(pred.samples);   % (n_samples, 189)
    mean_vec = double(pred.mean);

    % ---- 3. Pick the best sample ----------------------------------------
    sample_index = NaN;
    final_rmse_m = NaN;
    if rerank && isfield(opts, "forward_fn") && ~isempty(opts.forward_fn)
        [best_coeffs, sample_index, final_rmse_m] = ...
            local_rerank(samples, target, opts.forward_fn);
    else
        best_coeffs = mean_vec(:);
    end

    % ---- 4. Pack result -------------------------------------------------
    result = struct();
    result.coefficients    = best_coeffs(:);
    result.samples         = samples;
    result.sample_index    = sample_index;
    result.solver          = "swing_inverse_cvae";
    result.final_rmse_m    = final_rmse_m;
    result.checkpoint_path = string(opts.checkpoint_path);
    result.n_samples       = double(n_samples);
    result.duration_s      = toc(t_start);
    result.timestamp_utc   = string(datetime("now","TimeZone","UTC", ...
                                  "Format","yyyy-MM-dd'T'HH:mm:ss'Z'"));
end


function val = local_field_or(s, name, default)
    if isfield(s, name)
        val = s.(name);
    else
        val = default;
    end
end


function traj = local_build_trajectory(target, T)
    % Resample butt / clubhead / grip / v_clubhead onto T uniform timesteps
    % and stack into a (T, 12) single-precision matrix matching
    % training.py::_build_trajectory.
    if isfield(target, "grip") && ~isempty(target.grip)
        grip = target.grip;
    else
        grip = (target.butt + target.clubhead) / 2;
    end

    % Velocity by forward differences; pad final row with zero so size matches.
    v_clubhead = [diff(target.clubhead, 1, 1); zeros(1, size(target.clubhead, 2))] ...
        ./ max(mean(diff(target.time)), eps);

    raw = [target.butt(:,1:3), target.clubhead(:,1:3), grip(:,1:3), v_clubhead(:,1:3)];
    n = size(raw, 1);
    if n == T
        traj = single(raw);
        return;
    end
    src_t = linspace(0, 1, n);
    dst_t = linspace(0, 1, T);
    traj_double = zeros(T, size(raw, 2));
    for k = 1:size(raw, 2)
        traj_double(:, k) = interp1(src_t, raw(:, k), dst_t, "linear", "extrap");
    end
    traj = single(traj_double);
end


function [best, idx, rmse] = local_rerank(samples, target, forward_fn)
    n_samples = size(samples, 1);
    rmse_per = nan(n_samples, 1);
    sims = cell(n_samples, 1);
    for k = 1:n_samples
        sim_out = forward_fn(samples(k, :)');
        sims{k} = sim_out;
        rmse_per(k) = local_grip_rmse(sim_out, target);
    end
    [rmse, idx] = min(rmse_per);
    best = samples(idx, :)';
end


function rmse = local_grip_rmse(sim_out, target)
    % sim_out and target both expected to expose grip-equivalent (e.g. butt
    % or grip) (N,3) trajectories. We try ``grip`` first, then ``butt``.
    pred = local_pick_grip(sim_out);
    truth = local_pick_grip(target);
    n = min(size(pred, 1), size(truth, 1));
    diffs = pred(1:n, :) - truth(1:n, :);
    rmse = sqrt(mean(sum(diffs .^ 2, 2)));
end


function g = local_pick_grip(s)
    if isfield(s, "grip") && ~isempty(s.grip)
        g = s.grip;
    elseif isfield(s, "butt") && ~isempty(s.butt)
        g = s.butt;
    else
        error("fit_swing_inverse:noGripField", ...
              "rerank requires forward output with 'grip' or 'butt' field");
    end
end
