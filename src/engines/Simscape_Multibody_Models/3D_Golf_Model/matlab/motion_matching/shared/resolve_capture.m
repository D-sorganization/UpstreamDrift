function resolved_path = resolve_capture(capture_id, opts)
%RESOLVE_CAPTURE  Resolve a neutral capture ID to a verified local file path.
%
%   PATH = RESOLVE_CAPTURE(CAPTURE_ID) resolves the capture ID using
%   data/capture_registry.json and returns the verified file path.
%
%   PATH = RESOLVE_CAPTURE(CAPTURE_ID, data_dir=DATA_DIR) uses DATA_DIR as the
%   root directory for private captures instead of CAPTURE_DATA_DIR.
%
%   PATH = RESOLVE_CAPTURE(CAPTURE_ID, repo_root=REPO_ROOT) specifies the
%   repository root directory.
%
%   Errors:
%     'capture:unknown'      - The capture ID is not in the registry.
%     'capture:unavailable'  - Private capture is missing or CAPTURE_DATA_DIR is unset.
%     'capture:integrity'    - SHA-256 hash does not match registry.
%
    arguments
        capture_id (1,1) string
        opts.data_dir (1,1) string = string(missing)
        opts.repo_root (1,1) string = string(missing)
    end

    % 1. Determine repository root
    if ~ismissing(opts.repo_root) && strlength(strtrim(opts.repo_root)) > 0
        root = char(opts.repo_root);
    else
        root = find_repo_root();
    end

    % 2. Read and decode data/capture_registry.json
    manifest_path = fullfile(root, 'data', 'capture_registry.json');
    if exist(manifest_path, 'file') ~= 2
        error('capture:unavailable', ...
            'Capture registry manifest not found at %s', manifest_path);
    end

    raw_text = fileread(manifest_path);
    registry_data = jsondecode(raw_text);

    if isfield(registry_data, 'captures')
        registry_data = registry_data.captures;
    end

    % 3. Find capture entry by ID
    capture_id_str = string(capture_id);
    entry = [];

    if isstruct(registry_data)
        fields = fieldnames(registry_data);
        for i = 1:numel(fields)
            item = registry_data.(fields{i});
            if isfield(item, 'id') && strcmp(string(item.id), capture_id_str)
                entry = item;
                break;
            end
        end
    elseif iscell(registry_data)
        for i = 1:numel(registry_data)
            item = registry_data{i};
            if isfield(item, 'id') && strcmp(string(item.id), capture_id_str)
                entry = item;
                break;
            end
        end
    end

    if isempty(entry)
        error('capture:unknown', 'Unknown capture ID: %s', capture_id_str);
    end

    % 4. Locate target file based on public/private
    where = string(entry.where);
    rel_path = char(entry.relative_path);
    expected_sha256 = lower(string(entry.sha256));

    if strcmp(where, "public")
        target_path = fullfile(root, rel_path);
        if exist(target_path, 'file') ~= 2
            error('capture:unavailable', ...
                'Public capture %s not found at %s', capture_id_str, target_path);
        end
    elseif strcmp(where, "private")
        eff_data_dir = string(opts.data_dir);
        if ismissing(eff_data_dir) || strlength(strtrim(eff_data_dir)) == 0
            env_val = getenv("CAPTURE_DATA_DIR");
            if ~isempty(env_val)
                eff_data_dir = string(env_val);
            end
        end
        if ismissing(eff_data_dir) || strlength(strtrim(eff_data_dir)) == 0
            error('capture:unavailable', ...
                'Capture %s is private and CAPTURE_DATA_DIR environment variable is not set.', capture_id_str);
        end
        target_path = fullfile(char(eff_data_dir), rel_path);
        if exist(target_path, 'file') ~= 2
            error('capture:unavailable', ...
                'Private capture %s not found at %s', capture_id_str, target_path);
        end
    else
        error('capture:unknown', 'Unknown capture location type: %s', where);
    end

    % 5. Verify SHA-256 integrity
    actual_sha256 = compute_file_sha256(target_path);
    if ~strcmpi(actual_sha256, expected_sha256)
        error('capture:integrity', ...
            'SHA-256 mismatch for capture %s at %s: expected %s, got %s', ...
            capture_id_str, target_path, expected_sha256, actual_sha256);
    end

    resolved_path = char(target_path);
end

function root = find_repo_root()
    curr = fileparts(mfilename('fullpath'));
    root = '';
    for i = 1:12
        cand_manifest = fullfile(curr, 'data', 'capture_registry.json');
        if exist(cand_manifest, 'file') == 2
            root = curr;
            return;
        end
        parent = fileparts(curr);
        if strcmp(parent, curr)
            break;
        end
        curr = parent;
    end
    if isempty(root)
        error('capture:unavailable', ...
            'Could not locate repository root containing data/capture_registry.json');
    end
end

function hash_hex = compute_file_sha256(filepath)
    persistent hash_cache;
    if isempty(hash_cache)
        hash_cache = containers.Map('KeyType', 'char', 'ValueType', 'char');
    end

    f_info = dir(filepath);
    if isempty(f_info)
        error('capture:unavailable', 'File does not exist: %s', filepath);
    end
    cache_key = sprintf('%s_%d_%s', filepath, f_info.bytes, f_info.date);
    if isKey(hash_cache, cache_key)
        hash_hex = hash_cache(cache_key);
        return;
    end

    md = java.security.MessageDigest.getInstance('SHA-256');
    fid = fopen(filepath, 'r');
    if fid == -1
        error('capture:unavailable', 'Could not open file: %s', filepath);
    end
    cleanup_fid = onCleanup(@() fclose(fid));
    chunk_size = 65536;
    while ~feof(fid)
        bytes = fread(fid, chunk_size, '*uint8');
        if ~isempty(bytes)
            md.update(bytes);
        end
    end
    raw_hash = typecast(md.digest(), 'uint8');
    hash_hex = lower(sprintf('%02x', raw_hash));
    hash_cache(cache_key) = hash_hex;
end
