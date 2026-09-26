function run = gs3dx_simulate(mdl, opts)
%GS3DX_SIMULATE  Single forward-simulation entry point for every GS3DX model.
%
%   RUN = GS3DX_SIMULATE(MDL) simulates model MDL (already resolvable by name)
%   for 0.3 s with the persisted model settings and returns:
%     .model        char model name
%     .release      MATLAB release string
%     .wall_s       wall-clock seconds for sim()
%     .n_steps      number of solver output steps (numel(tout))
%     .flat         GS3DX_FLATTEN_BUS of CombinedSignalBus on a 1 kHz grid
%     .logsout      signal-logging Dataset (empty when nothing is logged)
%     .status       "success" | "failed"
%     .message      error text when status == "failed"
%
%   Name-value options:
%     stop_time        (default 0.3 s, the motion-matching window)
%     sample_rate      (default 1000 Hz) for the comparison grid
%     model_parameters struct of set_param overrides (e.g. Solver, MaxStep)
%     variables        struct of model-workspace variable overrides
%     block_parameters (:,3) cell of {block path, parameter, value} overrides
%
%   The model is simulated through Simulink.SimulationInput, so nothing is
%   written back to the .slx file.

    arguments
        mdl (1,:) char
        opts.stop_time   (1,1) double {mustBePositive} = 0.3
        opts.sample_rate (1,1) double {mustBePositive} = 1000
        opts.model_parameters (1,1) struct = struct()
        opts.variables        (1,1) struct = struct()
        opts.block_parameters (:,3) cell = cell(0, 3)
    end

    if ~bdIsLoaded(mdl)
        load_system(mdl);
    end
    in = Simulink.SimulationInput(mdl);
    in = in.setModelParameter('StopTime', num2str(opts.stop_time));
    in = in.setModelParameter('ReturnWorkspaceOutputs', 'on');
    p = fieldnames(opts.model_parameters);
    for k = 1:numel(p)
        in = in.setModelParameter(p{k}, opts.model_parameters.(p{k}));
    end
    v = fieldnames(opts.variables);
    for k = 1:numel(v)
        in = in.setVariable(v{k}, opts.variables.(v{k}), 'Workspace', mdl);
    end

    for k = 1:size(opts.block_parameters, 1)
        in = in.setBlockParameter(opts.block_parameters{k, :});
    end

    run = struct('model', mdl, 'release', version('-release'), 'wall_s', NaN, ...
        'n_steps', NaN, 'flat', [], 'logsout', [], 'status', "failed", 'message', "");
    grid = (0:1/opts.sample_rate:opts.stop_time).';
    t0 = tic;
    try
        out = sim(in);
    catch ME
        run.wall_s  = toc(t0);
        run.message = string(ME.message);
        return;
    end
    run.wall_s = toc(t0);
    if ~isempty(out.ErrorMessage)
        run.message = string(out.ErrorMessage);
        return;
    end
    run.n_steps = numel(out.tout);
    run.flat    = gs3dx_flatten_bus(out.CombinedSignalBus, grid);
    if any(strcmp(out.who, 'logsout'))
        run.logsout = out.logsout;
    end
    run.status  = "success";
end
