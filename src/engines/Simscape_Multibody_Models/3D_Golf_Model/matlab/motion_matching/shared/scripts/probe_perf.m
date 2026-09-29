function probe_perf()
%PROBE_PERF  Time GolfSwing3D_Kinetic under different logging configurations
%   so we can quantify what each output is costing us in wall-clock per
%   simulation, ahead of deciding whether to strip pieces out for fitting.
    addpath(genpath('src')); addpath(genpath('motion_matching/shared'));
    load_system('GolfSwing3D_Kinetic');
    S = load('src/model/inputs/3DModelInputs_Impact.mat');

    configs = {
        struct('label', 'baseline (model defaults)',                'simlog', 'all',  'signal_logging', 'on',  'csb_log',  true);
        struct('label', 'simlog=all, signal_logging=off',            'simlog', 'all',  'signal_logging', 'off', 'csb_log',  true);
        struct('label', 'simlog=local, signal_logging=on',           'simlog', 'local','signal_logging', 'on',  'csb_log',  true);
        struct('label', 'simlog=local, signal_logging=off',          'simlog', 'local','signal_logging', 'off', 'csb_log',  true);
        struct('label', 'simlog=none, signal_logging=off',           'simlog', 'none', 'signal_logging', 'off', 'csb_log',  true);
    };

    n_runs    = 3;
    stop_t_s  = 0.30;    % matches motion_matching default sim window
    test_fast = true;    % include a FastRestart benchmark at the end

    results = cell(numel(configs), 1);

    for c = 1:numel(configs)
        cfg = configs{c};
        in = Simulink.SimulationInput('GolfSwing3D_Kinetic');
        in = in.setModelParameter('StopTime', num2str(stop_t_s));
        in = in.setModelParameter('SimscapeLogType',  cfg.simlog);
        in = in.setModelParameter('SignalLogging',    cfg.signal_logging);
        in = in.setModelParameter('SaveOutput',       'on');
        in = in.setModelParameter('ReturnWorkspaceOutputs', 'on');
        in = in.setModelParameter('FastRestart', 'off');
        fns = fieldnames(S);
        for k = 1:numel(fns)
            try; in = in.setVariable(fns{k}, S.(fns{k})); catch; end
        end

        t = nan(n_runs, 1);
        n_csb = NaN; has_simlog = false; n_logsout = NaN; bytes = NaN;
        for r = 1:n_runs
            tic;
            out = sim(in);
            t(r) = toc;
            if r == n_runs
                if isprop(out, 'CombinedSignalBus') || isfield(out, 'CombinedSignalBus')
                    n_csb = numel(local_flatten_csb(out.CombinedSignalBus));
                end
                has_simlog = isprop(out, 'simlog') && ~isempty(out.simlog);
                if isprop(out, 'logsout') && ~isempty(out.logsout)
                    try; n_logsout = out.logsout.numElements; catch; n_logsout = NaN; end
                end
                bytes = local_struct_bytes(out);
            end
        end
        result = struct( ...
            'label',       cfg.label, ...
            'mean_s',      mean(t(2:end)), ...    % drop first run (load cost)
            'min_s',       min(t(2:end)), ...
            'csb_signals', n_csb, ...
            'simlog',      has_simlog, ...
            'logsout',     n_logsout, ...
            'sim_out_MB',  bytes / 1e6);
        results{c} = result;
        fprintf('  [%2d/%d] %-50s  mean=%6.2fs  min=%6.2fs  csb=%5d  out=%6.1f MB  simlog=%d  logsout=%d\n', ...
                c, numel(configs), cfg.label, result.mean_s, result.min_s, ...
                result.csb_signals, result.sim_out_MB, result.simlog, result.logsout);
    end

    if test_fast
        % FastRestart benchmark: 5 sequential sims with FastRestart on,
        % only the first pays the model-init / compile cost.
        in = Simulink.SimulationInput('GolfSwing3D_Kinetic');
        in = in.setModelParameter('StopTime', num2str(stop_t_s));
        in = in.setModelParameter('FastRestart', 'on');
        in = in.setModelParameter('SimscapeLogType', 'all');
        in = in.setModelParameter('SignalLogging', 'on');
        fns = fieldnames(S);
        for k = 1:numel(fns)
            try; in = in.setVariable(fns{k}, S.(fns{k})); catch; end
        end
        n_seq = 5;
        t = nan(n_seq, 1);
        for r = 1:n_seq
            tic; sim(in); t(r) = toc;
        end
        fprintf('\n[fast-restart sequence]  per-call times: %s\n', mat2str(t', 4));
        fprintf('  first run = %.2fs (cold), warm-mean = %.2fs over %d runs\n', ...
                t(1), mean(t(2:end)), n_seq - 1);
    end

    try
        % Fast-restart leaves the model initialised; turn it off before
        % closing.
        if bdIsLoaded('GolfSwing3D_Kinetic')
            set_param('GolfSwing3D_Kinetic', 'FastRestart', 'off');
        end
    catch
    end
    close_system('GolfSwing3D_Kinetic', 0);
    save('output/perf_probe_results.mat', 'results');
    fprintf('\nSaved to output/perf_probe_results.mat\n');
end

function n = local_flatten_csb(s)
    n = 0;
    if isstruct(s)
        f = fieldnames(s);
        for k = 1:numel(f); n = n + local_flatten_csb(s.(f{k})); end
    elseif isa(s, 'timeseries') || isnumeric(s)
        n = n + 1;
    end
end

function b = local_struct_bytes(s)
    try
        w = whos('s'); b = w.bytes;
    catch
        b = NaN;
    end
end
