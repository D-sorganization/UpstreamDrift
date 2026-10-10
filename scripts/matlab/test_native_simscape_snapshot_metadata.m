function receipt = test_native_simscape_snapshot_metadata(producer)
%TEST_NATIVE_SIMSCAPE_SNAPSHOT_METADATA Reject invalid semantic restart metadata.
%   Uses the trusted existing producer's actual operating point. Each negative
%   saves a new MAT payload and supplies its correct digest, so byte-integrity
%   rejection cannot mask a missing semantic provider or clock check.
    arguments
        producer (1,1) string
    end
    assert(strcmp(version('-release'), '2025b'), 'NativeOwned:R2025bRequired');
    original = fullfile(producer, 'native-operating-point.mat');
    source = load(original, 'op', 'native_meta');
    meta = source.native_meta;
    expected = struct('owned_directory', producer, ...
        'snapshot_sha256', native_simscape_file_sha256(original), ...
        'model_name', meta.model_name, 'model_sha256', meta.model_sha256, ...
        'input_sha256', meta.input_sha256, 'solver_id', meta.solver_id, ...
        'runtime_release', meta.runtime_release, ...
        'snapshot_time_s', meta.snapshot_time_s, ...
        'provider_sha256', meta.provider_sha256);
    load_native_simscape_snapshot(original, expected);
    path = fullfile(producer, 'metadata-negative-11942.mat');
    assert(~isfile(path), 'NativeOwned:NegativePathAlreadyExists');
    cleanup = onCleanup(@() local_remove_owned_file(path)); %#ok<NASGU>
    names = {'nan_clock', 'vector_clock', 'nearby_clock', ...
        'wrong_provider', 'missing_provider'};
    identifiers = {'NativeRestart:NativeSnapshotTimeMismatch', ...
        'NativeRestart:NativeSnapshotTimeMismatch', ...
        'NativeRestart:NativeSnapshotTimeMismatch', ...
        'NativeRestart:NativeProviderMismatch', 'NativeRestart:NativeProviderMismatch'};
    cases = struct();
    for k = 1:numel(names)
        native_meta = meta;
        switch names{k}
            case 'nan_clock'; native_meta.snapshot_time_s = NaN;
            case 'vector_clock'; native_meta.snapshot_time_s = [0.2, 0.2];
            case 'nearby_clock'
                native_meta.snapshot_time_s = meta.snapshot_time_s + ...
                    2 * eps(meta.snapshot_time_s);
            case 'wrong_provider'; native_meta.provider_sha256 = repmat('0', 1, 64);
            case 'missing_provider'; native_meta = rmfield(native_meta, 'provider_sha256');
        end
        op = source.op; %#ok<NASGU>
        save(path, 'op', 'native_meta', '-v7.3');
        bound = expected;
        bound.snapshot_sha256 = native_simscape_file_sha256(path);
        observed = '';
        try
            load_native_simscape_snapshot(path, bound);
        catch exception
            observed = exception.identifier;
        end
        cases.(names{k}) = struct('expected_identifier', identifiers{k}, ...
            'observed_identifier', observed, 'passed', strcmp(observed, identifiers{k}));
    end
    receipt = struct('issue', '#11942', 'scope', 'trusted diagnostic metadata', ...
        'runtime_id', version, 'cases', cases);
    disp(jsonencode(receipt, 'PrettyPrint', true));
    results = struct2cell(cases);
    assert(all(cellfun(@(item) item.passed, results)), ...
        'NativeOwned:SemanticMetadataRejectionMissing', ...
        'All semantic metadata rejection checks must pass');
end

function local_remove_owned_file(path)
    % PATH is a fixed leaf under the supplied producer, absent before this test.
    if isfile(path); delete(path); end
end
