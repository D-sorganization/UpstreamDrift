function receipt = run_owned_native_simscape_replay(directory, request_sha256)
%RUN_OWNED_NATIVE_SIMSCAPE_REPLAY Consume trusted, Python-validated owned bytes.
%   REQUEST_SHA256 is passed separately by the owner. MAT/SLX are executable
%   trusted producer artifacts, never arbitrary uploaded files. No retiming,
%   prehistory simulation, partial restore, or numeric-state reconstruction.
    arguments
        directory (1,1) string
        request_sha256 (1,1) string
    end
    request_path = fullfile(directory, 'native-request.json');
    local_digest(request_path, request_sha256);
    request = jsondecode(fileread(request_path));
    assert(strcmp(request.schema_version, 'simscape-owned-replay-request/1.0.0'), ...
        'NativeOwned:RequestVersion');
    assert(strcmp(request.model_name, 'native_restart_fixture_11921'), ...
        'NativeOwned:UnsupportedModel');
    assert(strcmp(request.runtime_id, version), 'NativeOwned:RuntimeMismatch');
    assert(strcmp(request.provider_sha256, native_simscape_provider_sha256()), ...
        'NativeOwned:ProviderMismatch');
    model_path = fullfile(directory, string(request.model_name) + ".slx");
    local_digest(model_path, request.model_sha256);
    assert(strcmp(request.model_sha256, request.loaded_model_sha256), ...
        'NativeOwned:LoadedModelMismatch');
    local_digest(fullfile(directory, 'native-operating-point.mat'), request.snapshot_sha256);
    local_digest(fullfile(directory, 'replay-input.csv'), request.replay_input_file_sha256);
    local_digest(fullfile(directory, 'native-envelope.json'), request.envelope_file_sha256);
    envelope = jsondecode(fileread(fullfile(directory, 'native-envelope.json')));
    assert(strcmp(envelope.integrity.envelope_sha256, request.envelope_sha256), ...
        'NativeOwned:EnvelopeMismatch');
    rows = readmatrix(fullfile(directory, 'replay-input.csv'));
    assert(size(rows, 2) == 2 && size(rows, 1) >= 2 && all(isfinite(rows), 'all') ...
        && all(diff(rows(:,1)) > 0) && rows(1,1) == request.snapshot_time_s ...
        && rows(end,1) == request.stop_time_s, 'NativeOwned:InputClockMismatch');
    input = timeseries(rows(:,2), rows(:,1));
    mdl = string(request.model_name);
    assert(~bdIsLoaded(mdl), 'NativeOwned:ModelAlreadyLoaded');
    cleanup = onCleanup(@() local_close(mdl)); %#ok<NASGU>
    load_system(model_path);
    observed = capture_native_simscape_execution(mdl, input);
    local_binding(observed, request);
    expected = struct('owned_directory', directory, ...
        'snapshot_sha256', request.snapshot_sha256, 'model_name', mdl, ...
        'model_sha256', request.model_sha256, ...
        'input_sha256', request.producer_input_sha256, 'solver_id', request.solver_id, ...
        'runtime_release', version('-release'), 'snapshot_time_s', request.snapshot_time_s);
    snapshot_path = fullfile(directory, 'native-operating-point.mat');
    op = load_native_simscape_snapshot(snapshot_path, expected);
    meta = load(snapshot_path, 'native_meta');
    local_binding(meta.native_meta.execution_binding, request);
    assert(double(op.startTime) == request.start_time_s, 'NativeOwned:StartClockMismatch');
    simulation = Simulink.SimulationInput(mdl);
    simulation = setVariable(simulation, 'native_force_input', input, 'Workspace', mdl);
    simulation = setInitialState(simulation, op);
    output = sim(setModelParameter(simulation, 'StopTime', num2str(request.stop_time_s, 17)));
    after = capture_native_simscape_execution(mdl, input);
    local_binding(after, request);
    local_digest(request_path, request_sha256);
    local_digest(model_path, request.model_sha256);
    local_digest(snapshot_path, request.snapshot_sha256);
    local_digest(fullfile(directory, 'replay-input.csv'), request.replay_input_file_sha256);
    local_digest(fullfile(directory, 'native-envelope.json'), request.envelope_file_sha256);
    receipt = local_outputs(directory, output, rows, request);
    fid = fopen(fullfile(directory, 'native-owned-replay-receipt.json'), 'w');
    assert(fid > 0, 'NativeOwned:ReceiptOpenFailed');
    file_cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s\n', jsonencode(receipt, 'PrettyPrint', true));
end

function local_digest(path, expected)
    assert(strcmp(native_simscape_file_sha256(path), expected), ...
        'NativeOwned:FileDigestMismatch', 'Owned file digest differs');
end

function local_binding(observed, request)
    fields = {'runtime_id','solver_id','solver_version', ...
        'effective_configuration_sha256','compatibility_sha256'};
    for k = 1:numel(fields)
        assert(strcmp(observed.(fields{k}), request.(fields{k})), ...
            'NativeOwned:ExecutionMismatch', 'Native execution binding differs');
    end
end

function receipt = local_outputs(directory, output, rows, request)
    names = {'physical_position','discrete_state','executed_force'};
    records = struct();
    for k = 1:numel(names)
        series = output.(names{k});
        t = double(series.Time(:)); v = double(series.Data(:));
        assert(numel(t) >= 2 && all(isfinite(t)) && all(diff(t) >= 0) ...
            && t(1) == request.snapshot_time_s && t(end) == request.stop_time_s ...
            && all(isfinite(v)), ...
            'NativeOwned:OutputClockMismatch');
        path = fullfile(directory, string(names{k}) + ".csv");
        writetable(table(t, v, 'VariableNames', {'time_s','value'}), path);
        records.(names{k}) = native_simscape_file_sha256(path);
        if strcmp(names{k}, 'executed_force')
            error_max = max(abs(v - interp1(rows(:,1), rows(:,2), t, 'previous', 'extrap')));
            assert(error_max <= 1e-12, 'NativeOwned:ExecutedInputMismatch');
        end
    end
    receipt = struct('issue', '#11942', 'scope', 'synthetic diagnostic only', ...
        'request_sha256', native_simscape_file_sha256(fullfile(directory, 'native-request.json')), ...
        'snapshot_time_s', request.snapshot_time_s, 'stop_time_s', request.stop_time_s, ...
        'max_executed_input_error_n', error_max, 'outputs_sha256', records);
end

function local_close(mdl)
    if bdIsLoaded(mdl); close_system(mdl, 0); end
end
