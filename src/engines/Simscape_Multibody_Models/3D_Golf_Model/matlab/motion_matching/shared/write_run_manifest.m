function write_run_manifest(manifest_path, fields)
%WRITE_RUN_MANIFEST Write MS-60 Simscape run_manifest.json (R2025b only).
%
%   WRITE_RUN_MANIFEST(MANIFEST_PATH, FIELDS) validates and writes the
%   versioned run manifest consumed by Python
%   src/shared/python/motion_matching/simscape_run_manifest.py.
%
%   Required FIELDS keys:
%     run_id, matlab_release, matlab_version, host,
%     model_sha256, candidate_sha256, replay_npz_sha256,
%     wall_clock_s, qualification, evidence_dir, artifacts
%
%   GitHub issue: #10347 (MS-60).
    arguments
        manifest_path (1,1) string
        fields (1,1) struct
    end

    required = { ...
        'run_id', 'matlab_release', 'matlab_version', 'host', ...
        'model_sha256', 'candidate_sha256', 'replay_npz_sha256', ...
        'wall_clock_s', 'qualification', 'evidence_dir', 'artifacts'};
    for i = 1:numel(required)
        assert(isfield(fields, required{i}), ...
            'MissingRequiredField: %s', required{i});
    end

    release = lower(strtrim(char(fields.matlab_release)));
    if startsWith(release, 'r')
        release = extractAfter(release, 1);
    end
    assert(strcmp(release, '2025b'), ...
        'UnsupportedMatlabRelease: R2025b required, got %s (no R2026a substitution)', ...
        char(fields.matlab_release));

    assert(strlength(strtrim(string(fields.host))) > 0, 'EmptyHost');
    assert(double(fields.wall_clock_s) >= 0, 'NegativeWallClock');

    sha_keys = {'model_sha256', 'candidate_sha256', 'replay_npz_sha256'};
    for i = 1:numel(sha_keys)
        digest = lower(char(fields.(sha_keys{i})));
        assert(numel(digest) == 64 && all(ismember(digest, '0123456789abcdef')), ...
            'InvalidSha256: %s', sha_keys{i});
    end

    assert(isstruct(fields.artifacts) && ~isempty(fieldnames(fields.artifacts)), ...
        'EmptyArtifacts');

    payload = struct();
    payload.schema_version = 'simscape-run-manifest/1';
    if isfield(fields, 'issue')
        payload.issue = char(fields.issue);
    else
        payload.issue = '#10347';
    end
    payload.run_id = char(fields.run_id);
    payload.matlab_release = char(fields.matlab_release);
    payload.matlab_version = char(fields.matlab_version);
    payload.host = char(fields.host);
    if isfield(fields, 'machine')
        payload.machine = char(fields.machine);
    end
    payload.model_sha256 = lower(char(fields.model_sha256));
    payload.candidate_sha256 = lower(char(fields.candidate_sha256));
    payload.replay_npz_sha256 = lower(char(fields.replay_npz_sha256));
    payload.wall_clock_s = double(fields.wall_clock_s);
    payload.qualification = char(fields.qualification);
    payload.evidence_dir = char(fields.evidence_dir);
    payload.artifacts = fields.artifacts;
    if isfield(fields, 'extra')
        payload.extra = fields.extra;
    end

    text = jsonencode(payload, 'PrettyPrint', true);
    fid = fopen(manifest_path, 'w');
    assert(fid > 0, 'CannotOpenManifest: %s', manifest_path);
    cleaner = onCleanup(@() fclose(fid));
    fwrite(fid, text, 'char');
    fprintf(fid, '\n');
end
