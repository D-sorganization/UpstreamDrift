function results = perf_maxstep_sweep(opts)
%PERF_MAXSTEP_SWEEP  Sweep solver MaxStep on GolfSwing3D_Kinetic and quantify
%   the wall-clock / accuracy trade-off so we can pick a sensible default.
%
%   RESULTS = PERF_MAXSTEP_SWEEP() runs the model with the canonical Impact
%   inputs (`src/model/inputs/3DModelInputs_Impact.mat`) at each MaxStep in
%       {0.0001, 0.001, 0.002, 0.005, 0.01, 0.02}
%   The 0.0001 run is treated as the "ground truth" reference for accuracy
%   metrics (grip RMSE, total-work relative error). For each MaxStep we run
%   one cold sim (discarded) plus three warm FastRestart sims and record the
%   per-call wall-clock.
%
%   For accuracy we extract:
%     - Grip position trace from CombinedSignalBus.MidpointCalcsLogs.MPGlobalPosition
%       (mm RMSE on the reference timegrid via interp1).
%     - Total mechanical work W = trapz(t, sum(|tau .* qd|, 2)) using the
%       joint signals exposed via logsout / CombinedSignalBus.
%
%   Output:
%     output/maxstep_sweep_<timestamp>/results.mat
%     output/maxstep_sweep_<timestamp>/MAX_STEP_REPORT.md
%
%   OPTS (optional struct):
%     .max_steps      vector of MaxStep values (default the canonical 6).
%     .n_warm_runs    integer >=1 (default 3).
%     .stop_time      seconds (default 0.30).
%     .output_root    directory containing output/ (default current pwd
%                     resolved up to the matlab/ folder).
%     .model_name     default 'GolfSwing3D_Kinetic'.
%     .inputs_mat     path to the Impact MAT (default
%                     'src/model/inputs/3DModelInputs_Impact.mat').
%
%   Issue: D-sorganization/UpstreamDrift#4078.
%
%   See also: PROBE_PERF, PREPARE_FAST_SIM_INPUT.

    arguments
        opts (1,1) struct = struct()
    end

    if ~isfield(opts, 'max_steps');   opts.max_steps   = [0.0001, 0.001, 0.002, 0.005, 0.01, 0.02]; end
    if ~isfield(opts, 'n_warm_runs'); opts.n_warm_runs = 3;   end
    if ~isfield(opts, 'stop_time');   opts.stop_time   = 0.30; end
    if ~isfield(opts, 'model_name');  opts.model_name  = 'GolfSwing3D_Kinetic'; end
    if ~isfield(opts, 'inputs_mat');  opts.inputs_mat  = 'src/model/inputs/3DModelInputs_Impact.mat'; end
    if ~isfield(opts, 'output_root'); opts.output_root = pwd; end

    % --- 1. Resolve paths and load model -----------------------------------
    if ~isfile(opts.inputs_mat)
        error('perf_maxstep_sweep:noInputs', ...
              'Impact inputs MAT not found: %s. Run from the matlab/ root.', ...
              opts.inputs_mat);
    end
    S = load(opts.inputs_mat);

    if ~bdIsLoaded(opts.model_name)
        load_system(opts.model_name);
    end

    timestamp = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<DATST,TNOW1>
    out_dir = fullfile(opts.output_root, 'output', sprintf('maxstep_sweep_%s', timestamp));
    if ~isfolder(out_dir); mkdir(out_dir); end

    fprintf('[perf_maxstep_sweep] sweeping %d MaxStep values, %d warm runs each\n', ...
            numel(opts.max_steps), opts.n_warm_runs);
    fprintf('[perf_maxstep_sweep] output dir: %s\n', out_dir);

    % --- 2. Build base SimulationInput (re-parameterised per MaxStep) ------
    fns = fieldnames(S);

    n_settings = numel(opts.max_steps);
    results = struct( ...
        'max_step',           num2cell(opts.max_steps(:)), ...
        'cold_s',             num2cell(nan(n_settings, 1)), ...
        'warm_times_s',       cell(n_settings, 1), ...
        'mean_warm_s',        num2cell(nan(n_settings, 1)), ...
        'min_warm_s',         num2cell(nan(n_settings, 1)), ...
        'grip_rmse_mm',       num2cell(nan(n_settings, 1)), ...
        'total_work_J',       num2cell(nan(n_settings, 1)), ...
        'rel_work_error_pct', num2cell(nan(n_settings, 1)), ...
        'solver_ok',          num2cell(false(n_settings, 1)));

    % Reference traces (filled by the first MaxStep, which must be the
    % smallest / most accurate setting in the sweep).
    ref_time = [];
    ref_grip = [];
    ref_work = NaN;

    for i = 1:n_settings
        ms = opts.max_steps(i);
        fprintf('\n[perf_maxstep_sweep] (%d/%d) MaxStep=%g\n', i, n_settings, ms);

        % Force a clean recompile per setting — solver params (MaxStep)
        % cannot be changed inside an active FastRestart session.
        try
            if bdIsLoaded(opts.model_name)
                set_param(opts.model_name, 'FastRestart', 'off');
            end
        catch
        end

        in = Simulink.SimulationInput(opts.model_name);
        in = in.setModelParameter('StopTime',           num2str(double(opts.stop_time)));
        in = in.setModelParameter('FastRestart',        'on');
        in = in.setModelParameter('SaveOutput',         'on');
        in = in.setModelParameter('ReturnWorkspaceOutputs', 'on');
        in = in.setModelParameter('SimscapeLogType',    'all');
        in = in.setModelParameter('SignalLogging',      'on');
        in = in.setModelParameter('MaxStep',            num2str(double(ms)));
        for k = 1:numel(fns)
            try
                in = in.setVariable(fns{k}, S.(fns{k}));
            catch
            end
        end

        % --- Cold run (discard timing, primes FastRestart) ------------------
        try
            t0 = tic; sim(in); cold_t = toc(t0);
        catch ME
            fprintf('  cold run FAILED: %s\n', ME.message);
            results(i).solver_ok = false;
            continue;
        end
        results(i).cold_s = cold_t;
        fprintf('  cold = %.2f s\n', cold_t);

        % --- Warm runs ------------------------------------------------------
        wt = nan(opts.n_warm_runs, 1);
        last_out = [];
        for r = 1:opts.n_warm_runs
            try
                t0 = tic; out = sim(in); wt(r) = toc(t0);
            catch ME
                fprintf('  warm run %d FAILED: %s\n', r, ME.message);
                wt(r) = NaN;
                continue;
            end
            last_out = out;
        end
        results(i).warm_times_s = wt;
        results(i).mean_warm_s  = mean(wt(~isnan(wt)));
        results(i).min_warm_s   = min(wt(~isnan(wt)));
        fprintf('  warm runs = %s s  (mean=%.2f, min=%.2f)\n', ...
                mat2str(wt', 4), results(i).mean_warm_s, results(i).min_warm_s);

        if isempty(last_out)
            results(i).solver_ok = false;
            continue;
        end

        % --- Extract grip + work --------------------------------------------
        [grip_t, grip_xyz] = local_extract_grip(last_out);
        W                  = local_extract_total_work(last_out);
        results(i).total_work_J = W;
        results(i).solver_ok    = ~isnan(W) && ~isempty(grip_xyz);

        if i == 1
            % First setting MUST be the reference (smallest MaxStep).
            ref_time = grip_t;
            ref_grip = grip_xyz;
            ref_work = W;
            results(i).grip_rmse_mm       = 0;
            results(i).rel_work_error_pct = 0;
            fprintf('  reference grip trace captured (%d samples), W_ref = %.4g J\n', ...
                    numel(ref_time), ref_work);
        else
            results(i).grip_rmse_mm       = local_grip_rmse_mm(ref_time, ref_grip, grip_t, grip_xyz);
            if isfinite(ref_work) && ref_work ~= 0
                results(i).rel_work_error_pct = 100 * (W - ref_work) / abs(ref_work);
            end
            fprintf('  grip RMSE = %.3f mm,  total work = %.4g J  (rel err %.3f%%)\n', ...
                    results(i).grip_rmse_mm, W, results(i).rel_work_error_pct);
        end
    end

    % --- 3. Recommend a MaxStep -------------------------------------------
    rec = local_recommend(results);

    % --- 4. Write artefacts -----------------------------------------------
    save(fullfile(out_dir, 'results.mat'), 'results', 'rec', 'opts');
    md_path = fullfile(out_dir, 'MAX_STEP_REPORT.md');
    local_write_report(md_path, results, rec, opts);
    fprintf('\n[perf_maxstep_sweep] wrote %s\n', md_path);

    % --- 5. Tear down FastRestart -----------------------------------------
    try
        if bdIsLoaded(opts.model_name)
            set_param(opts.model_name, 'FastRestart', 'off');
        end
    catch
    end
end

%% =====================================================================
function [t, xyz] = local_extract_grip(simOut)
%LOCAL_EXTRACT_GRIP  Pull MidpointCalcsLogs.MPGlobalPosition from CSB.
    t   = [];
    xyz = [];
    try
        if ~(isprop(simOut, 'CombinedSignalBus') || isfield(simOut, 'CombinedSignalBus'))
            return;
        end
        csb = simOut.CombinedSignalBus;
        if ~isfield(csb, 'MidpointCalcsLogs') || ~isfield(csb.MidpointCalcsLogs, 'MPGlobalPosition')
            return;
        end
        ts = csb.MidpointCalcsLogs.MPGlobalPosition;
        d  = double(ts.Data);
        if ndims(d) == 3
            % Could be 1x3xN or 3x1xN; squeeze.
            d = squeeze(d);
        end
        if size(d, 2) ~= 3 && size(d, 1) == 3
            d = d.';
        end
        if size(d, 2) ~= 3
            return;
        end
        if isprop(ts, 'Time')
            t = double(ts.Time(:));
        else
            t = (0:size(d,1)-1)';
        end
        xyz = d;
    catch
        t = [];
        xyz = [];
    end
end

%% =====================================================================
function rmse_mm = local_grip_rmse_mm(t_ref, xyz_ref, t, xyz)
%LOCAL_GRIP_RMSE_MM  RMSE in millimetres on the reference grid.
    rmse_mm = NaN;
    if isempty(t_ref) || isempty(t) || isempty(xyz_ref) || isempty(xyz)
        return;
    end
    try
        % Interpolate the candidate trace onto the reference grid.
        xi = interp1(t, xyz, t_ref, 'linear', 'extrap');
        d  = xi - xyz_ref;
        % Per-frame Euclidean error in metres -> RMSE in mm.
        per_frame = sqrt(sum(d.^2, 2));
        rmse_mm   = 1000 * sqrt(mean(per_frame.^2));
    catch
        rmse_mm = NaN;
    end
end

%% =====================================================================
function W = local_extract_total_work(simOut)
%LOCAL_EXTRACT_TOTAL_WORK  W = trapz(t, sum(|tau .* qd|, 2)).
%   Pulls joint torques (Ideal_Torque_Source.t) and velocities
%   (Kinetically_Driven_*.{Rx,Ry,Rz,Px,Py,Pz}.w) from the Simscape simlog.
    W = NaN;
    try
        if ~(isprop(simOut, 'simlog') || isfield(simOut, 'simlog'))
            return;
        end
        sl = simOut.simlog;
        if isempty(sl); return; end
    catch
        return;
    end
    tau_cells = {};
    w_cells   = {};
    [tau_cells, w_cells] = local_walk_simlog(sl, tau_cells, w_cells);
    if isempty(tau_cells); return; end

    % Snap to the first series's time grid; resample the rest if needed.
    T = tau_cells{1}.series.time;
    n_pairs = numel(tau_cells);
    integrand = zeros(numel(T), 1);
    for k = 1:n_pairs
        tv = tau_cells{k}.series.values;
        tt = tau_cells{k}.series.time;
        wv = w_cells{k}.series.values;
        wt = w_cells{k}.series.time;
        if numel(tt) ~= numel(T)
            tv = interp1(tt(:), tv(:), T, 'linear', 'extrap');
        end
        if numel(wt) ~= numel(T)
            wv = interp1(wt(:), wv(:), T, 'linear', 'extrap');
        end
        integrand = integrand + abs(tv(:) .* wv(:));
    end
    try
        W = trapz(T, integrand);
    catch
        W = NaN;
    end
    if ~isfinite(W)
        W = NaN;
    end
end

%% =====================================================================
function [tau_cells, w_cells] = local_walk_simlog(node, tau_cells, w_cells)
%LOCAL_WALK_SIMLOG  Recursively collect (Ideal_Torque_Source.t,
%   Kinetically_Driven_*.<axis>.w) pairs from the simlog tree.
    try
        ids = node.childIds;
    catch
        return;
    end
    has_torque_source = false;
    for k = 1:numel(ids)
        if startsWith(ids{k}, 'Ideal_Torque_Source')
            has_torque_source = true;
        end
    end
    if has_torque_source
        kdrev  = [];
        kduniv = [];
        for k = 1:numel(ids)
            if startsWith(ids{k}, 'Kinetically_Driven_Revolute')
                kdrev = node.(ids{k});
            elseif startsWith(ids{k}, 'Kinetically_Driven_Universal') || ...
                   startsWith(ids{k}, 'Kinetically_Driven_Gimbal')
                kduniv = node.(ids{k});
            end
        end
        for k = 1:numel(ids)
            id = ids{k};
            if ~startsWith(id, 'Ideal_Torque_Source')
                continue;
            end
            ts_node = node.(id);
            tau_series = [];
            try
                tau_series = ts_node.t;
            catch
                continue;
            end
            suffix = '';
            if numel(id) > numel('Ideal_Torque_Source')
                suffix = id(numel('Ideal_Torque_Source')+1:end);
            end
            w_node = local_pick_axis(kdrev, kduniv, suffix);
            if ~isempty(w_node) && ~isempty(tau_series)
                tau_cells{end+1} = tau_series; %#ok<AGROW>
                w_cells{end+1}   = w_node;     %#ok<AGROW>
            end
        end
    end
    for k = 1:numel(ids)
        ch = node.(ids{k});
        try
            if isa(ch, 'simscape.logging.Node')
                [tau_cells, w_cells] = local_walk_simlog(ch, tau_cells, w_cells);
            end
        catch
        end
    end
end

%% =====================================================================
function w_node = local_pick_axis(kdrev, kduniv, suffix)
    w_node = [];
    target_axis = '';
    switch upper(suffix)
        case '_X'; target_axis = 'Rx';
        case '_Y'; target_axis = 'Ry';
        case '_Z'; target_axis = 'Rz';
        case '';   target_axis = 'Rz'; % single revolute usually Rz
    end
    if isempty(target_axis); return; end
    candidates = {kdrev, kduniv};
    for c = 1:numel(candidates)
        nd = candidates{c};
        if isempty(nd); continue; end
        try
            if any(strcmp(target_axis, nd.childIds))
                w_node = nd.(target_axis).w;
                return;
            end
        catch
        end
    end
end

%% =====================================================================
function rec = local_recommend(results)
%LOCAL_RECOMMEND  Pick the largest MaxStep whose grip RMSE stays under
%   the accuracy floor (5 mm). Speedup is reported vs the model default
%   MaxStep=0.001. If looser MaxStep values produce the SAME trajectory
%   as the default (i.e. solver step-size is bounded by RelTol/AbsTol,
%   not MaxStep), the recommendation is to keep the default since looser
%   values give no speedup.
    rec = struct( ...
        'max_step',           NaN, ...
        'speedup_vs_default', NaN, ...
        'grip_rmse_mm',       NaN, ...
        'rel_work_error_pct', NaN, ...
        'rationale',          "no recommendation");

    accuracy_floor_mm   = 5.0;
    work_floor_pct      = 5.0;
    catastrophic_mm     = 100.0;

    valid = arrayfun(@(r) r.solver_ok && isfinite(r.mean_warm_s), results);
    if ~any(valid); return; end

    % Default reference for speedup = MaxStep == 0.001 if present, else
    % the second-most-precise setting.
    ms_vec = [results.max_step];
    valid_vec = reshape(valid, 1, []);
    if numel(valid_vec) ~= numel(ms_vec)
        valid_vec = valid_vec(1:numel(ms_vec));
    end
    default_idx = find(ms_vec == 0.001 & valid_vec, 1);
    if isempty(default_idx)
        finite_settings = find(valid_vec);
        if isempty(finite_settings); return; end
        default_idx = finite_settings(min(2, numel(finite_settings)));
    end
    default_t = results(default_idx).mean_warm_s;
    default_ms = results(default_idx).max_step;

    % First, check whether MaxStep is even biting. If runs at the default
    % AND at every looser setting all share the same grip RMSE vs the
    % reference (i.e. the solver auto-picks dt below MaxStep), then no
    % loosening helps and we recommend keeping the default.
    looser_idx = find(ms_vec >= default_ms & valid_vec);
    if numel(looser_idx) >= 2
        rmses = [results(looser_idx).grip_rmse_mm];
        rmse_spread = max(rmses) - min(rmses);
        if all(isfinite(rmses)) && rmse_spread < 1.0  % within 1 mm
            r = results(default_idx);
            rec.max_step           = default_ms;
            rec.speedup_vs_default = 1.0;
            rec.grip_rmse_mm       = r.grip_rmse_mm;
            rec.rel_work_error_pct = r.rel_work_error_pct;
            rec.rationale = sprintf( ...
                ['MaxStep is not the binding step-size constraint at the ', ...
                 'current default (%g): all values >= %g produce the same ', ...
                 'grip trace within %.2f mm. The solver is bounded by ', ...
                 'RelTol/AbsTol, not MaxStep, so loosening yields no speedup. ', ...
                 'Recommend keeping MaxStep=%g as the default; use the new ', ...
                 'high_precision opt-in (MaxStep=%g) only when ground-truth ', ...
                 'accuracy is required (~%.1fx slower per call).'], ...
                default_ms, default_ms, rmse_spread, default_ms, ...
                results(1).max_step, ...
                results(1).mean_warm_s / max(default_t, eps));
            return;
        end
    end

    % Candidate set: solver succeeded, RMSE under the floor, work error
    % under the floor, NOT catastrophic.
    candidates = false(numel(results), 1);
    for i = 1:numel(results)
        r = results(i);
        if ~r.solver_ok; continue; end
        if i == 1
            candidates(i) = true; % reference always valid
            continue;
        end
        if ~isfinite(r.grip_rmse_mm); continue; end
        if r.grip_rmse_mm > catastrophic_mm; continue; end
        if r.grip_rmse_mm > accuracy_floor_mm; continue; end
        if isfinite(r.rel_work_error_pct) && abs(r.rel_work_error_pct) > work_floor_pct
            continue;
        end
        candidates(i) = true;
    end
    if ~any(candidates); return; end

    % Of the candidates, choose the one with the largest MaxStep (fastest).
    cand_idx = find(candidates);
    [~, j]  = max([results(cand_idx).max_step]);
    pick    = cand_idx(j);
    r       = results(pick);

    rec.max_step           = r.max_step;
    rec.speedup_vs_default = default_t / r.mean_warm_s;
    rec.grip_rmse_mm       = r.grip_rmse_mm;
    rec.rel_work_error_pct = r.rel_work_error_pct;
    rec.rationale          = sprintf( ...
        ['MaxStep=%g is the largest sweep value that keeps grip RMSE under ', ...
         '%.1f mm and total-work error under %.1f%% relative to the ', ...
         'MaxStep=%g reference. Mean warm runtime drops from %.2fs (default) ', ...
         'to %.2fs (%.2fx).'], ...
        r.max_step, accuracy_floor_mm, work_floor_pct, results(1).max_step, ...
        default_t, r.mean_warm_s, rec.speedup_vs_default);
end

%% =====================================================================
function local_write_report(path, results, rec, opts)
    fid = fopen(path, 'w');
    if fid < 0
        warning('perf_maxstep_sweep:reportOpen', 'Could not open %s for writing.', path);
        return;
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, '# MaxStep Performance Sweep\n\n');
    fprintf(fid, '_GitHub issue: D-sorganization/UpstreamDrift#4078._\n\n');
    fprintf(fid, '- Model: `%s`\n', opts.model_name);
    fprintf(fid, '- Stop time: %.3f s\n', opts.stop_time);
    fprintf(fid, '- Warm runs per setting: %d (plus 1 cold cache-priming run, discarded)\n', opts.n_warm_runs);
    fprintf(fid, '- Reference for accuracy: MaxStep = %g (smallest swept value, treated as ground truth)\n', results(1).max_step);
    fprintf(fid, '- Generated: %s\n\n', datestr(now)); %#ok<DATST,TNOW1>

    fprintf(fid, '## Results\n\n');
    fprintf(fid, '| MaxStep | mean_warm_s | grip_rmse_mm | total_work_J | rel_work_error_pct |\n');
    fprintf(fid, '|---|---|---|---|---|\n');
    for i = 1:numel(results)
        r = results(i);
        fprintf(fid, '| %g | %s | %s | %s | %s |\n', ...
                r.max_step, ...
                local_fmt(r.mean_warm_s, '%.3f'), ...
                local_fmt(r.grip_rmse_mm, '%.3f'), ...
                local_fmt(r.total_work_J, '%.4g'), ...
                local_fmt(r.rel_work_error_pct, '%.3f'));
    end

    fprintf(fid, '\n## Recommendation\n\n');
    if isfinite(rec.max_step)
        fprintf(fid, 'Use **MaxStep = %g**. Speedup vs default %.2fx; grip RMSE %.3f mm vs reference.\n\n', ...
                rec.max_step, rec.speedup_vs_default, rec.grip_rmse_mm);
    else
        fprintf(fid, 'No MaxStep > reference satisfied the accuracy gates; recommend keeping the existing default.\n\n');
    end
    fprintf(fid, '%s\n\n', char(rec.rationale));

    fprintf(fid, '## Method\n\n');
    fprintf(fid, ['Each setting runs 1 cold sim (discarded — pays the FastRestart compile cost) ', ...
                  'then %d warm sims with `FastRestart=on`. Wall-clock is per-call seconds. ', ...
                  'Grip position is `CombinedSignalBus.MidpointCalcsLogs.MPGlobalPosition`; ', ...
                  'RMSE is computed against the smallest MaxStep''s trace interpolated onto its ', ...
                  'native grid. Total work uses `W = trapz(t, sum(abs(tau .* qd), 2))` with ', ...
                  'joint torque and velocity traces from `logsout` / `CombinedSignalBus`. ', ...
                  'Acceptance gates: grip RMSE <= 5 mm and |rel_work_error| <= 5%%; results above ', ...
                  '100 mm grip RMSE are flagged as broken rather than recommended.\n\n'], ...
            opts.n_warm_runs);
end

%% =====================================================================
function s = local_fmt(v, fmt)
    if isnan(v) || ~isfinite(v)
        s = '—';
    else
        s = sprintf(fmt, v);
    end
end
