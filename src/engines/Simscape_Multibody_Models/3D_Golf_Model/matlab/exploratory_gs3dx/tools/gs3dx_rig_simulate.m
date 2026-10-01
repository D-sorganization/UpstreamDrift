function out = gs3dx_rig_simulate(rig, log_port, log_name, opts)
%GS3DX_RIG_SIMULATE  Simulate a joint rig and return one logged bus.
%
%   OUT = GS3DX_RIG_SIMULATE(RIG, LOG_PORT, LOG_NAME, OPTS) logs the bus on
%   Simulink output port LOG_PORT (its line is named LOG_NAME), simulates
%   the unsaved model RIG with ode15s at RelTol = AbsTol = OPTS.rel_tol to
%   OPTS.stop_time, and returns, at exactly OPTS.output_times:
%     .t        (N,1) output times
%     .signals  struct of (N,k) arrays, one field per bus element
%     .wall_s   wall-clock seconds of the simulation

    arguments
        rig (1,:) char
        log_port (1,1) double
        log_name (1,:) char
        opts (1,1) struct
    end
    set_param(log_port, 'DataLogging', 'on');
    set_param(rig, 'StopTime', num2str(opts.stop_time, 17), 'SolverType', 'Variable-step', ...
        'Solver', 'ode15s', 'RelTol', num2str(opts.rel_tol, 17), 'AbsTol', num2str(opts.rel_tol, 17), ...
        'OutputOption', 'SpecifiedOutputTimes', 'OutputTimes', mat2str(opts.output_times, 17), ...
        'ReturnWorkspaceOutputs', 'on', 'SignalLogging', 'on', 'SignalLoggingName', 'logsout');
    tic;
    sim_out = sim(rig);
    out.wall_s = toc;
    ts = sim_out.logsout.get(log_name).Values;
    out.t = opts.output_times;
    out.signals = struct();
    for f = reshape(fieldnames(ts), 1, [])
        v = ts.(f{1});
        if isa(v, 'timeseries')
            d = reshape(v.Data, [], numel(v.Time)).';
            if isscalar(v.Time)                % constant signal: logged once
                d = repmat(d, numel(out.t), 1);
            else
                assert(isequal(v.Time(:), out.t), 'gs3dx:rig', 'Output times of %s differ', f{1});
            end
            out.signals.(matlab.lang.makeValidName(f{1})) = d;
        end
    end
end
