function hex = native_simscape_provider_sha256()
%NATIVE_SIMSCAPE_PROVIDER_SHA256 Bind this diagnostic's actual MATLAB sources.
%   SHA-256 of sorted UTF-8 filename:lowercase-file-SHA-256 LF records.
    names = sort({'capture_native_simscape_execution.m', ...
        'load_native_simscape_snapshot.m', 'native_simscape_bytes_sha256.m', ...
        'native_simscape_file_sha256.m', 'native_simscape_provider_sha256.m', ...
        'run_native_simscape_restart.m', 'run_owned_native_simscape_replay.m'});
    records = '';
    for k = 1:numel(names)
        source = which(names{k});
        assert(~isempty(source), 'NativeOwned:MissingProviderSource');
        records = [records names{k} ':' ...
            native_simscape_file_sha256(string(source)) newline]; %#ok<AGROW>
    end
    hex = native_simscape_bytes_sha256(unicode2native(records, 'UTF-8'));
end
