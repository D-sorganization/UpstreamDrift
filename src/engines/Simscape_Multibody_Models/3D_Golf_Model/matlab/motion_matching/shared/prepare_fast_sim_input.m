function in = prepare_fast_sim_input(theta_or_struct, opts)
%PREPARE_FAST_SIM_INPUT  Build a Simulink.SimulationInput optimised for
%   high-throughput repeat simulations (e.g. an inner optimizer loop).
%
%   IN = PREPARE_FAST_SIM_INPUT(THETA, OPTS) takes a polynomial coefficient
%   vector THETA and returns a SimulationInput with:
%     - FastRestart = on               (saves ~50% per call after the first)
%     - SaveOutput  = on
%     - SignalLogging = on             (kept on; CSB is what the loaders read)
%     - SimscapeLogType per opts       (default 'all')
%     - Polynomial coefficients applied via theta_to_polynomial_struct
%     - Optional starting-pose overrides applied via opts.input_overrides
%
%   IN = PREPARE_FAST_SIM_INPUT(SI, OPTS) where SI is an existing
%   SimulationInput just augments it with the fast-restart settings.
%
%   OPTS:
%     .model_name           default 'GolfSwing3D_Kinetic'
%     .stop_time            default 0.30 (motion_matching window)
%     .simscape_log         'all' | 'local' | 'none' (default 'all')
%     .input_overrides      struct of additional setVariable assignments
%                           (e.g. starting-pose perturbations on top of
%                           the Impact MAT).  Each field is the workspace
%                           variable name; value is the override.
%     .joint_names          string array; auto-resolved if empty.
%
%   Performance notes
%   -----------------
%   Empirical measurement on this codebase (probe_perf.m, R2025b on this
%   machine):
%       cold sim                     ~15.0 s
%       warm sim (FastRestart=on)     ~7.0 s   ← the real win for fitting
%       toggling SimscapeLogType      < 3% impact (noise)
%       disabling SignalLogging       < 3% impact (noise)
%   The solver dominates wall-clock; **don't** bother stripping the
%   CombinedSignalBus / simlog plumbing — it's free.  Use FastRestart and
%   minimise the number of sim calls instead.
%
%   See also: SIMULATE_WITH_COEFFICIENTS, PROBE_PERF, FIT_SWING_FMINCON.

    arguments
        theta_or_struct
        opts (1,1) struct = struct()
    end
    if ~isfield(opts, 'model_name');    opts.model_name    = 'GolfSwing3D_Kinetic'; end
    if ~isfield(opts, 'stop_time');     opts.stop_time     = 0.30;                  end
    if ~isfield(opts, 'simscape_log');  opts.simscape_log  = 'all';                 end
    if ~isfield(opts, 'input_overrides'); opts.input_overrides = struct();          end
    if ~isfield(opts, 'joint_names');   opts.joint_names   = string([]);            end

    if ~bdIsLoaded(opts.model_name)
        load_system(opts.model_name);
    end

    if isa(theta_or_struct, 'Simulink.SimulationInput')
        in = theta_or_struct;
    else
        in = Simulink.SimulationInput(opts.model_name);
    end

    in = in.setModelParameter('StopTime',          num2str(double(opts.stop_time)));
    in = in.setModelParameter('FastRestart',       'on');
    in = in.setModelParameter('SaveOutput',        'on');
    in = in.setModelParameter('ReturnWorkspaceOutputs', 'on');
    in = in.setModelParameter('SimscapeLogType',   char(opts.simscape_log));
    in = in.setModelParameter('SignalLogging',     'on');

    % Apply theta if a coefficient vector was provided.
    if isnumeric(theta_or_struct) && ~isempty(theta_or_struct)
        if isempty(opts.joint_names)
            param_info = getPolynomialParameterInfo();
            joint_names = string(param_info.joint_names);
        else
            joint_names = string(opts.joint_names(:)).';
        end
        coeff_struct = theta_to_polynomial_struct(theta_or_struct, joint_names);
        f = fieldnames(coeff_struct);
        for k = 1:numel(f)
            in = in.setVariable(f{k}, coeff_struct.(f{k}));
        end
    end

    % Layer caller-supplied overrides on top.
    f = fieldnames(opts.input_overrides);
    for k = 1:numel(f)
        in = in.setVariable(f{k}, opts.input_overrides.(f{k}));
    end
end
