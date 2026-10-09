function op = load_native_simscape_snapshot(snapshot_path, expected)
%LOAD_NATIVE_SIMSCAPE_SNAPSHOT Admit a task-owned R2025b restart MAT file.
%   EXPECTED binds owned directory, file/model/input digests, solver, runtime,
%   and exact nonzero snapshot time. This fixture loader is not a general
%   untrusted-MAT decoder or a Tools 1.1 frozen-byte implementation.

    arguments
        snapshot_path (1,1) string
        expected (1,1) struct
    end
    required = {'owned_directory','snapshot_sha256','model_name', ...
        'model_sha256','input_sha256','solver_id', ...
        'runtime_release','snapshot_time_s'};
    assert(all(isfield(expected, required)), 'NativeRestart:ExpectedFieldsMissing');
    if ~isfile(snapshot_path)
        error('NativeRestart:MissingNativeSnapshot', '%s', snapshot_path);
    end
    owned = string(java.io.File(char(expected.owned_directory)).getCanonicalPath());
    source = string(java.io.File(char(snapshot_path)).getCanonicalPath());
    if ~startsWith(lower(source), lower(owned + string(filesep)))
        error('NativeRestart:UnownedNativeSnapshot', '%s', snapshot_path);
    end
    if ~strcmp(native_simscape_file_sha256(snapshot_path), ...
            expected.snapshot_sha256)
        error('NativeRestart:NativeBlobDigestMismatch', '%s', snapshot_path);
    end
    model_path = fullfile(expected.owned_directory, ...
        string(expected.model_name) + ".slx");
    if ~isfile(model_path) || ~strcmp(native_simscape_file_sha256(model_path), ...
            expected.model_sha256)
        error('NativeRestart:NativeModelIdentityMismatch', '%s', model_path);
    end
    if ~strcmp(expected.runtime_release, version('-release'))
        error('NativeRestart:NativeRuntimeMismatch', 'MATLAB release differs');
    end
    loaded = load(snapshot_path, 'op', 'native_meta');
    if ~isfield(loaded, 'op') || ...
            ~isa(loaded.op, 'Simulink.op.ModelOperatingPoint')
        error('NativeRestart:NativeClassMismatch', 'ModelOperatingPoint required');
    end
    if ~isfield(loaded, 'native_meta') || ~isstruct(loaded.native_meta)
        error('NativeRestart:NativeMetadataMissing', 'Native metadata required');
    end
    meta = loaded.native_meta;
    fields = {'model_name','model_sha256','input_sha256', ...
        'solver_id','runtime_release','snapshot_time_s'};
    if ~all(isfield(meta, fields))
        error('NativeRestart:NativeMetadataMissing', 'Native metadata incomplete');
    end
    if ~strcmp(meta.model_name, expected.model_name) || ...
            ~strcmp(meta.model_sha256, expected.model_sha256)
        error('NativeRestart:NativeModelIdentityMismatch', 'Model differs');
    end
    if ~strcmp(meta.solver_id, expected.solver_id) || ...
            ~strcmp(meta.input_sha256, expected.input_sha256)
        error('NativeRestart:NativeExecutionMismatch', 'Solver/input differs');
    end
    if ~strcmp(meta.runtime_release, expected.runtime_release)
        error('NativeRestart:NativeRuntimeMismatch', 'Runtime differs');
    end
    op = loaded.op;
    if ~isfinite(op.snapshotTime) || op.snapshotTime <= op.startTime || ...
            abs(op.snapshotTime - expected.snapshot_time_s) > 1e-10 || ...
            abs(meta.snapshot_time_s - expected.snapshot_time_s) > 1e-10
        error('NativeRestart:NativeSnapshotTimeMismatch', 'Native clock differs');
    end
    if ~contains(string(op.description), string(expected.model_name))
        error('NativeRestart:NativeModelIdentityMismatch', ...
            'Operating point model description differs');
    end
end
