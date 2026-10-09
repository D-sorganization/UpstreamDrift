function binding = capture_native_simscape_execution(mdl, input)
%CAPTURE_NATIVE_SIMSCAPE_EXECUTION Observe diagnostic simulation compatibility.
%   Compilation is explicitly for simulation. Its structural checksum is
%   separate from effective tunable values and the immutable SLX byte digest.
%   The temporary model-workspace input is restored even if compilation fails.
    arguments
        mdl (1,1) string
        input (1,1) timeseries
    end
    assert(strcmp(version('-release'), '2025b'), 'NativeOwned:R2025bRequired');
    assert(strcmp(mdl, 'native_restart_fixture_11921'), ...
        'NativeOwned:UnsupportedModel');
    assert(strcmp(get_param(mdl, 'SimulationStatus'), 'stopped'), ...
        'NativeOwned:ModelMustBeStopped');
    before = local_configuration(mdl);
    workspace = get_param(mdl, 'ModelWorkspace');
    existed = hasVariable(workspace, 'native_force_input');
    previous = [];
    if existed; previous = getVariable(workspace, 'native_force_input'); end
    input_cleanup = onCleanup(@() local_restore_input(workspace, existed, previous)); %#ok<NASGU>
    assignin(workspace, 'native_force_input', input);
    term_cleanup = onCleanup(@() local_terminate(mdl)); %#ok<NASGU>
    feval(char(mdl), [], [], [], 'compile');
    checksum = Simulink.BlockDiagram.getChecksum(char(mdl));
    during = local_configuration(mdl);
    assert(isequal(before, during), 'NativeOwned:CompileChangedConfiguration');
    compatibility = struct('context', 'explicit-simulation-compile', ...
        'checksum_uint32', double(checksum(:).'));
    binding = struct('runtime_release', version('-release'), ...
        'runtime_id', version, 'solver_id', during.model.Solver, ...
        'solver_version', regexp(version, '^[0-9]+\.[0-9]+\.[0-9]+', 'match', 'once'), ...
        'effective_configuration', during, 'compatibility', compatibility, ...
        'effective_configuration_sha256', local_json_sha256(during), ...
        'compatibility_sha256', local_json_sha256(compatibility));
end

function configuration = local_configuration(mdl)
    names = {'SimulationMode','StartTime','Solver','SolverType','MaxStep','RelTol','AbsTol', ...
        'BlockReduction','SaveFinalState','SaveOperatingPoint','FinalStateName', ...
        'ReturnWorkspaceOutputs','OperatingPointInterfaceChecksumMismatchMsg', ...
        'OperatingPointContentsChecksumMismatchMsg','NonCurrentReleaseOperatingPointMsg'};
    model = struct();
    for k = 1:numel(names); model.(names{k}) = get_param(mdl, names{k}); end
    strict = names(end-2:end);
    for k = 1:numel(strict)
        assert(strcmp(model.(strict{k}), 'error'), ...
            'NativeOwned:StrictRestoreRequired', 'Partial native restore is forbidden');
    end
    blocks = {'Input','Input','Input','Input','Input','ToPhysical','ToPhysical', ...
        'ToSimulink','Delay','Delay','Mass','Spring','Damper','Sensor'};
    parameters = {'VariableName','Interpolate','OutputAfterFinalValue','SampleTime','ZeroCross', ...
        'Unit','FilteringAndDerivatives','Unit','SampleTime','InitialCondition', ...
        'mass','spr_rate','D','reference'};
    values = cell(size(blocks));
    for k = 1:numel(blocks)
        values{k} = struct('block', blocks{k}, 'parameter', parameters{k}, ...
            'value', get_param(mdl + "/" + blocks{k}, parameters{k}));
    end
    solver_path = mdl + "/Solver";
    solver_parameters = get_param(solver_path, 'DialogParameters');
    solver_names = sort(fieldnames(solver_parameters));
    solver = struct();
    for k = 1:numel(solver_names)
        solver.(solver_names{k}) = get_param(solver_path, solver_names{k});
    end
    configuration = struct('schema_version', 'simscape-diagnostic-config/1.0.0', ...
        'model', model, 'blocks', {values}, 'physical_solver', solver);
end

function local_restore_input(workspace, existed, previous)
    if existed
        assignin(workspace, 'native_force_input', previous);
    else
        evalin(workspace, 'clear native_force_input');
    end
end

function local_terminate(mdl)
    if bdIsLoaded(mdl) && ~strcmp(get_param(mdl, 'SimulationStatus'), 'stopped')
        feval(char(mdl), [], [], [], 'term');
    end
end

function hex = local_json_sha256(value)
    bytes = unicode2native(jsonencode(value), 'UTF-8');
    hex = native_simscape_bytes_sha256(bytes);
end
