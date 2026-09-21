function report = export_candidate(evidence_dir, opts)
%EXPORT_CANDIDATE Export Simscape run evidence toward MatchedSwingCandidate.
%
%   REPORT = EXPORT_CANDIDATE(EVIDENCE_DIR, OPTS) verifies the returned-replay
%   NPZ + returned-candidate.json pair that Python
%   convert_simscape_returned_replay / candidate_io.save_candidate consume.
%   This keeps MATLAB aligned with the MS-15 candidate_io layout without a
%   second parallel schema.
%
%   OPTS fields (all optional):
%     require_npz (default true) — fail closed if returned-replay.npz missing
%     require_candidate_json (default true)
%
%   GitHub issue: #10347 (MS-60). Reuses replay_returned102_r2025b evidence layout.
    arguments
        evidence_dir (1,1) string
        opts (1,1) struct = struct()
    end

    if ~isfield(opts, 'require_npz'); opts.require_npz = true; end
    if ~isfield(opts, 'require_candidate_json'); opts.require_candidate_json = true; end

    evidence_dir = string(evidence_dir);
    assert(isfolder(evidence_dir), 'MissingEvidenceDir: %s', evidence_dir);

    replay_npz = fullfile(evidence_dir, 'returned-replay.npz');
    candidate_json = fullfile(evidence_dir, 'returned-candidate.json');
    candidate_npz = fullfile(evidence_dir, 'candidate.npz');
    manifest_json = fullfile(evidence_dir, 'run_manifest.json');

    if opts.require_npz
        assert(isfile(replay_npz), 'MissingReturnedReplayNpz: %s', replay_npz);
    end
    if opts.require_candidate_json
        assert(isfile(candidate_json), 'MissingReturnedCandidateJson: %s', candidate_json);
        cand = jsondecode(fileread(candidate_json));
        assert(isfield(cand, 'coordinate_names') && numel(cand.coordinate_names) == 27, ...
            'Expected27CoordinateNames');
        assert(isfield(cand, 'model_sha256') && strlength(string(cand.model_sha256)) == 64, ...
            'MissingModelSha256');
    else
        cand = struct();
    end

    report = struct();
    report.evidence_dir = char(evidence_dir);
    report.returned_replay_npz = char(replay_npz);
    report.returned_candidate_json = char(candidate_json);
    report.candidate_npz = char(candidate_npz);
    report.run_manifest_json = char(manifest_json);
    report.candidate_npz_present = isfile(candidate_npz);
    report.run_manifest_present = isfile(manifest_json);
    if isfield(cand, 'model_sha256')
        report.model_sha256 = char(cand.model_sha256);
    end
    if isfield(cand, 'source_sha256')
        report.candidate_sha256 = char(cand.source_sha256);
    end
    report.note = [ ...
        'Python owns MatchedSwingCandidate .npz packaging via ', ...
        'convert_simscape_returned_replay + candidate_io.save_candidate; ', ...
        'this exporter verifies the MATLAB-side inputs and receipt paths.'];
end
