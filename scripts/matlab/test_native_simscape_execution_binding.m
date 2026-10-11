function receipt = test_native_simscape_execution_binding(out_dir)
%TEST_NATIVE_SIMSCAPE_EXECUTION_BINDING Native compilation/configuration tests.
%   Uses a previously generated trusted diagnostic model, never capture data.
    arguments
        out_dir (1,1) string
    end
    mdl = 'native_restart_fixture_11921';
    load_system(fullfile(out_dir, mdl + ".slx"));
    cleanup = onCleanup(@() close_system(mdl, 0)); %#ok<NASGU>
    data = load(fullfile(out_dir, 'native-restart-input.mat'));
    input = timeseries(data.input_force, data.input_time);
    names = {'OperatingPointInterfaceChecksumMismatchMsg', ...
        'OperatingPointContentsChecksumMismatchMsg','NonCurrentReleaseOperatingPointMsg'};
    for k = 1:numel(names); set_param(mdl, names{k}, 'error'); end
    first = capture_native_simscape_execution(mdl, input);
    second = capture_native_simscape_execution(mdl, input);
    assert(isequal(first, second), 'NativeBindingMustBeRepeatable');
    assert(strcmp(get_param(mdl, 'SimulationStatus'), 'stopped'), ...
        'NativeCompileMustTerminate');
    workspace = get_param(mdl, 'ModelWorkspace');
    assert(~hasVariable(workspace, 'native_force_input'), 'NativeInputScopeLeaked');
    sentinel = timeseries(-77 * ones(size(data.input_time)), data.input_time);
    assignin(workspace, 'native_force_input', sentinel);
    capture_native_simscape_execution(mdl, input);
    assert(isequal(getVariable(workspace, 'native_force_input'), sentinel), ...
        'ExistingNativeInputMustBeRestored');
    evalin(workspace, 'clear native_force_input');
    set_param(mdl, 'MaxStep', '0.005');
    changed = capture_native_simscape_execution(mdl, input);
    assert(~strcmp(first.effective_configuration_sha256, ...
        changed.effective_configuration_sha256), 'TunableConfigurationMustBeBound');
    set_param(mdl, 'MaxStep', '0.01');
    set_param(mdl, names{1}, 'warning');
    rejected = false;
    try
        capture_native_simscape_execution(mdl, input);
    catch exception
        rejected = strcmp(exception.identifier, 'NativeOwned:StrictRestoreRequired');
    end
    assert(rejected, 'PartialRestoreWarningMustBeRejected');
    receipt = struct('binding', first, 'repeatable', true, ...
        'compile_terminated', true, 'input_scope_restored', true, ...
        'tunable_configuration_bound', true, 'partial_restore_rejected', true);
    fid = fopen(fullfile(out_dir, 'native-binding-tests.json'), 'w');
    assert(fid > 0, 'NativeBindingReceiptOpenFailed');
    file_cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s\n', jsonencode(receipt, 'PrettyPrint', true));
end
