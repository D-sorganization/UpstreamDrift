function receipt = test_native_simscape_restart(out_dir)
%TEST_NATIVE_SIMSCAPE_RESTART Native R2025b split/reload replay contract.
%   This test uses a small generated physical network, not a golfer model.

    arguments
        out_dir (1,1) string
    end
    assert(strcmp(version('-release'), '2025b'), 'R2025bRequired');
    receipt = run_native_simscape_restart(out_dir);
    assert(receipt.native_state_saved, 'NativeStateNotSaved');
    assert(receipt.native_state_reloaded, 'NativeStateNotReloaded');
    assert(receipt.split_time_s > 0, 'SnapshotMustBeNonzeroTime');
    assert(abs(receipt.actual_snapshot_time_s - receipt.split_time_s) < 1e-10, ...
        'SavedSnapshotTimeMismatch');
    assert(abs(receipt.actual_replay_start_s - receipt.split_time_s) < 1e-10, ...
        'RestoredNativeClockMismatch');
    assert(receipt.max_physical_position_error_m <= receipt.position_bound_m, ...
        'PhysicalReplayMismatch');
    assert(receipt.max_discrete_state_error <= receipt.discrete_bound, ...
        'DiscreteReplayMismatch');
    assert(receipt.physical_span_m > 1e-6, 'PhysicalFixtureDidNotMove');
    assert(receipt.discrete_span > 0.1, 'DiscreteFixtureDidNotAdvance');
    assert(strcmp(receipt.saved_input_sha256, receipt.replayed_input_sha256), ...
        'SavedInputMismatch');
    assert(strcmp(receipt.input_interpolation, 'zero_order_hold'), ...
        'InputInterpolationMustBeZOH');
    assert(strcmp(receipt.input_after_final_value, 'Holding final value'), ...
        'InputTailPolicyMismatch');
    assert(strcmp(receipt.block_reduction, 'off'), ...
        'NativeBlockReductionMustBeOff');
    assert(strcmp(receipt.input_converter_unit, 'N') && ...
        strcmp(receipt.input_converter_filtering, 'zero'), ...
        'NativeConverterPolicyMismatch');
    assert(receipt.max_executed_input_error_n <= 1e-12, ...
        'ExecutedInputMismatch');
    for name = {'full_physical','replay_physical','full_discrete', ...
            'replay_discrete','executed_input','saved_input'}
        record = receipt.diagnostic_files.(name{1});
        assert(isfile(fullfile(out_dir, record.path)), ...
            'MissingDiagnosticSeries: %s', name{1});
        assert(strcmp(native_simscape_file_sha256( ...
            fullfile(out_dir, record.path)), record.sha256), ...
            'DiagnosticSeriesHashMismatch');
    end
    expected = struct('model_name', receipt.fixture, ...
        'model_sha256', receipt.model_sha256, ...
        'solver_id', receipt.solver_id, ...
        'snapshot_time_s', receipt.actual_snapshot_time_s, ...
        'runtime_release', receipt.matlab_release, ...
        'input_sha256', receipt.saved_input_sha256, ...
        'snapshot_sha256', receipt.snapshot_sha256, ...
        'owned_directory', out_dir);
    artifact = fullfile(out_dir, 'native-operating-point.mat');
    restored = load_native_simscape_snapshot(artifact, expected);
    assert(isa(restored, 'Simulink.op.ModelOperatingPoint'), ...
        'NativeBlobWasNotRestored');
    local_expect_rejection(fullfile(out_dir, 'absent-native-state.mat'), ...
        expected, 'MissingNativeSnapshot');
    wrong = expected; wrong.model_sha256 = repmat('0', 1, 64);
    local_expect_rejection(artifact, wrong, 'NativeModelIdentityMismatch');
    wrong = expected; wrong.snapshot_time_s = 0;
    local_expect_rejection(artifact, wrong, 'NativeSnapshotTimeMismatch');
    wrong = expected; wrong.runtime_release = 'R2026a';
    local_expect_rejection(artifact, wrong, 'NativeRuntimeMismatch');
    wrong = expected; wrong.snapshot_sha256 = repmat('0', 1, 64);
    local_expect_rejection(artifact, wrong, 'NativeBlobDigestMismatch');
    bad_class_path = fullfile(out_dir, 'invalid-native-class.mat');
    op = struct('numeric_state', 1); %#ok<NASGU>
    native_meta = struct(); %#ok<NASGU>
    save(bad_class_path, 'op', 'native_meta');
    cleanup = onCleanup(@() delete(bad_class_path)); %#ok<NASGU>
    wrong = expected;
    wrong.snapshot_sha256 = native_simscape_file_sha256(bad_class_path);
    local_expect_rejection(bad_class_path, wrong, 'NativeClassMismatch');
end

function local_expect_rejection(path, expected, identifier)
    try
        load_native_simscape_snapshot(path, expected);
    catch err
        assert(strcmp(err.identifier, ['NativeRestart:' identifier]), ...
            'UnexpectedRejection: %s', ...
            err.identifier);
        return
    end
    error('MissingRejection: expected %s', identifier);
end
