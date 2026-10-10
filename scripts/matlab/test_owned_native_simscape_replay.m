function receipt = test_owned_native_simscape_replay(directory, request_sha256, producer)
%TEST_OWNED_NATIVE_SIMSCAPE_REPLAY Compare suffix-only continuation to full run.
%   The reference shares original prehistory and the declared future inputs.
%   Conflicting workspace values must not replace the saved input player.
    arguments
        directory (1,1) string
        request_sha256 (1,1) string
        producer (1,1) string
    end
    request_path = fullfile(directory, 'native-request.json');
    original = fileread(request_path);
    request = jsondecode(original);
    local_reject(directory, repmat('0', 1, 64), 'NativeOwned:FileDigestMismatch');
    bad = request; bad.effective_configuration_sha256 = repmat('0', 1, 64);
    local_write(request_path, jsonencode(bad));
    local_reject(directory, native_simscape_file_sha256(request_path), ...
        'NativeOwned:ExecutionMismatch');
    local_write(request_path, original);
    rows = readmatrix(fullfile(directory, 'replay-input.csv'));
    prefix = load(fullfile(producer, 'native-restart-input.mat'));
    earlier = prefix.input_time < request.snapshot_time_s;
    time = [prefix.input_time(earlier); rows(:,1)];
    force = [prefix.input_force(earlier); rows(:,2)];
    mdl = string(request.model_name);
    load_system(fullfile(directory, mdl + ".slx"));
    cleanup = onCleanup(@() local_close(mdl)); %#ok<NASGU>
    workspace = get_param(mdl, 'ModelWorkspace');
    assignin(workspace, 'native_force_input', timeseries(77 * ones(size(time)), time));
    existed = evalin('base', "exist('native_force_input','var') ~= 0");
    previous = [];
    if existed; previous = evalin('base', 'native_force_input'); end
    base_cleanup = onCleanup(@() local_restore_base(existed, previous)); %#ok<NASGU>
    assignin('base', 'native_force_input', timeseries(-99 * ones(size(time)), time));
    full = Simulink.SimulationInput(mdl);
    full = setVariable(full, 'native_force_input', timeseries(force, time), 'Workspace', mdl);
    reference = sim(setModelParameter(full, 'StopTime', num2str(request.stop_time_s, 17)));
    close_system(mdl, 0);
    receipt = run_owned_native_simscape_replay(directory, request_sha256);
    physical = readmatrix(fullfile(directory, 'physical_position.csv'));
    discrete = readmatrix(fullfile(directory, 'discrete_state.csv'));
    position_reference = interp1(double(reference.physical_position.Time), ...
        double(reference.physical_position.Data), physical(:,1), 'linear');
    discrete_reference = interp1(double(reference.discrete_state.Time), ...
        double(reference.discrete_state.Data), discrete(:,1), 'previous');
    receipt.max_reference_position_error_m = max(abs(position_reference - physical(:,2)));
    receipt.max_reference_discrete_error = max(abs(discrete_reference - discrete(:,2)));
    receipt.position_bound_m = 1e-5;
    receipt.discrete_bound = 1e-12;
    receipt.suffix_only_inputs = true;
    receipt.base_and_model_workspace_sentinels = true;
    assert(receipt.max_reference_position_error_m <= receipt.position_bound_m, ...
        'NativeOwned:ReferencePhysicalMismatch');
    assert(receipt.max_reference_discrete_error <= receipt.discrete_bound, ...
        'NativeOwned:ReferenceDiscreteMismatch');
    assert(~bdIsLoaded(mdl), 'NativeOwned:ModelCleanupFailed');
    local_write(fullfile(directory, 'native-owned-reference-tests.json'), ...
        jsonencode(receipt, 'PrettyPrint', true));
end

function local_reject(directory, digest, identifier)
    rejected = false;
    try
        run_owned_native_simscape_replay(directory, digest);
    catch exception
        rejected = strcmp(exception.identifier, identifier);
    end
    assert(rejected, 'NativeOwned:ExpectedRejectionMissing');
    assert(~bdIsLoaded('native_restart_fixture_11921'), 'NativeOwned:FailureCleanupFailed');
end

function local_write(path, text)
    fid = fopen(path, 'w');
    assert(fid > 0, 'NativeOwned:TestFileOpenFailed');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fwrite(fid, text, 'char');
end

function local_restore_base(existed, previous)
    if existed; assignin('base', 'native_force_input', previous);
    else; evalin('base', 'clear native_force_input'); end
end

function local_close(mdl)
    if bdIsLoaded(mdl); close_system(mdl, 0); end
end
