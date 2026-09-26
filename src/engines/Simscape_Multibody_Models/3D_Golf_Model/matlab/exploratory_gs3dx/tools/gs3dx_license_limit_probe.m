function results = gs3dx_license_limit_probe(opts)
%GS3DX_LICENSE_LIMIT_PROBE  Probe Home/Student license nonvirtual block limits.
%
%   RESULTS = GS3DX_LICENSE_LIMIT_PROBE() constructs synthetic in-memory
%   Simulink and Simscape models to determine whether and where license-imposed
%   block limits occur:
%     1. Chained Gain blocks (N = 900, 1000, 1100) from Constant to Terminator
%     2. Chained Simulink-PS + PS-Simulink converter pairs (N = 400, 500, 600)
%        around a Simscape Solver Configuration block.
%
%   Returns a table with columns:
%     - ProbeType: string ("Gain" or "ConverterPair")
%     - N: count parameter (number of Gains or Converter pairs)
%     - NonvirtualTotal: total nonvirtual blocks in the synthetic model
%     - Status: "Success" or "Error"
%     - ErrorMessage: exact error message if an error occurred, else ""
%     - DurationSeconds: elapsed duration for simulation attempt
%
%   Contract:
%     - Never throws on a license error (or simulation error).
%     - Every synthetic system created is closed with close_system(name, 0).
%     - No model is saved to disk in models/ (built in memory only).

    arguments
        opts.gain_counts (1,:) double = [900, 1000, 1100]
        opts.conv_counts (1,:) double = [400, 500, 600]
        opts.verbose     (1,1) logical = false
    end

    total_runs = numel(opts.gain_counts) + numel(opts.conv_counts);
    probe_type = strings(total_runs, 1);
    counts     = zeros(total_runs, 1);
    nv_totals  = zeros(total_runs, 1);
    statuses   = strings(total_runs, 1);
    err_msgs   = strings(total_runs, 1);
    durations  = zeros(total_runs, 1);

    row = 0;

    % Part 1: Chained Gain blocks
    for N = opts.gain_counts
        row = row + 1;
        probe_type(row) = "Gain";
        counts(row) = N;

        mdl_name = sprintf('gs3dx_synth_gain_%d', round(N));
        if bdIsLoaded(mdl_name)
            close_system(mdl_name, 0);
        end
        new_system(mdl_name);
        c = onCleanup(@() close_system(mdl_name, 0));
        set_param(mdl_name, 'StopTime', '0.001');

        add_block('simulink/Sources/Constant', [mdl_name '/Constant']);
        add_block('simulink/Sinks/Terminator', [mdl_name '/Terminator']);

        for k = 1:N
            g_name = sprintf('Gain_%d', k);
            add_block('simulink/Math Operations/Gain', [mdl_name '/' g_name]);
            if k == 1
                add_line(mdl_name, 'Constant/1', [g_name '/1']);
            else
                add_line(mdl_name, sprintf('Gain_%d/1', k-1), [g_name '/1']);
            end
        end
        add_line(mdl_name, sprintf('Gain_%d/1', N), 'Terminator/1');

        budget = gs3dx_block_budget(mdl_name);
        nv_totals(row) = budget.nonvirtual_total;

        t0 = tic;
        try
            sim(mdl_name);
            statuses(row) = "Success";
            err_msgs(row) = "";
        catch err
            statuses(row) = "Error";
            err_msgs(row) = string(err.message);
        end
        durations(row) = toc(t0);

        if opts.verbose
            fprintf('[Gain Probe] N=%d, Nonvirtual=%d: %s (%.2f s)\n', ...
                N, nv_totals(row), statuses(row), durations(row));
            if statuses(row) == "Error"
                fprintf('  Error: %s\n', err_msgs(row));
            end
        end

        clear c; % Invokes close_system(mdl_name, 0)
    end

    % Part 2: Converter pairs around a Simscape Solver Configuration block
    for N = opts.conv_counts
        row = row + 1;
        probe_type(row) = "ConverterPair";
        counts(row) = N;

        mdl_name = sprintf('gs3dx_synth_conv_%d', round(N));
        if bdIsLoaded(mdl_name)
            close_system(mdl_name, 0);
        end
        new_system(mdl_name);
        c = onCleanup(@() close_system(mdl_name, 0));
        set_param(mdl_name, 'StopTime', '0.001');

        add_block('simulink/Sources/Constant', [mdl_name '/Constant']);
        add_block('nesl_utility/Solver Configuration', [mdl_name '/SolverConfig']);
        add_block('simulink/Sinks/Terminator', [mdl_name '/Terminator']);

        for k = 1:N
            sps = sprintf('SPS_%d', k);
            pss = sprintf('PSS_%d', k);
            add_block('nesl_utility/Simulink-PS Converter', [mdl_name '/' sps]);
            add_block('nesl_utility/PS-Simulink Converter', [mdl_name '/' pss]);
            if k == 1
                add_line(mdl_name, 'Constant/1', [sps '/1']);
            else
                prev_pss = sprintf('PSS_%d', k-1);
                add_line(mdl_name, [prev_pss '/1'], [sps '/1']);
            end
            if k == N
                add_line(mdl_name, [pss '/1'], 'Terminator/1');
            end
        end

        budget = gs3dx_block_budget(mdl_name);
        nv_totals(row) = budget.nonvirtual_total;

        t0 = tic;
        try
            sim(mdl_name);
            statuses(row) = "Success";
            err_msgs(row) = "";
        catch err
            statuses(row) = "Error";
            err_msgs(row) = string(err.message);
        end
        durations(row) = toc(t0);

        if opts.verbose
            fprintf('[Conv Probe] N=%d pairs, Nonvirtual=%d: %s (%.2f s)\n', ...
                N, nv_totals(row), statuses(row), durations(row));
            if statuses(row) == "Error"
                fprintf('  Error: %s\n', err_msgs(row));
            end
        end

        clear c; % Invokes close_system(mdl_name, 0)
    end

    results = table(probe_type, counts, nv_totals, statuses, err_msgs, durations, ...
        'VariableNames', {'ProbeType', 'N', 'NonvirtualTotal', 'Status', 'ErrorMessage', 'DurationSeconds'});

    % Postconditions
    assert(istable(results), 'gs3dx:probe', 'Postcondition: results must be a table.');
    assert(height(results) == total_runs, 'gs3dx:probe', ...
        'Postcondition: results height (%d) must match total probe runs (%d).', ...
        height(results), total_runs);
end
