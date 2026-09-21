function report = run_simscape_candidate(repo, run_id, do_replay)
%RUN_SIMSCAPE_CANDIDATE  Documented R2025b entry for Simscape candidate runs.
%
%   REPORT = RUN_SIMSCAPE_CANDIDATE(REPO, RUN_ID, DO_REPLAY)
%   For run_id "two_window_fit_9967_102", optionally invokes
%   replay_returned102_r2025b and always refreshes run_manifest.json next to
%   the committed evidence using write_run_manifest.
%
%   Preconditions:
%     - MATLAB R2025b only.
%     - REPO points at an UpstreamDrift checkout containing native_evidence.
%   Postconditions:
%     - run_manifest.json exists under the run evidence directory.

    arguments
        repo (1,1) string
        run_id (1,1) string = "two_window_fit_9967_102"
        do_replay (1,1) logical = false
    end

    rel = version('-release');
    assert(strcmp(rel, '2025b'), 'R2025bRequired: detected %s', rel);

    evidence_rel = fullfile( ...
        'docs', 'development', 'simscape_tour_matching', 'native_evidence', run_id);
    evidence_dir = fullfile(repo, evidence_rel);
    assert(exist(evidence_dir, 'dir') == 7, 'EvidenceDirMissing: %s', evidence_dir);

    shared = fullfile(repo, 'src', 'engines', 'Simscape_Multibody_Models', ...
        '3D_Golf_Model', 'matlab', 'motion_matching', 'shared');
    addpath(shared);
    addpath(evidence_dir);

    report = struct();
    report.run_id = run_id;
    report.matlab_release = string(rel);
    report.replayed = false;

    if do_replay
        assert(strcmp(run_id, "two_window_fit_9967_102"), ...
            'ReplayNotImplemented: only two_window_fit_9967_102 is wired');
        report.replay = replay_returned102_r2025b(repo);
        report.replayed = true;
        wall_clock_s = double(report.replay.elapsed_s);
        qualification = "qualified_r2025b_cold_replay_returned102";
    else
        qual_path = fullfile(evidence_dir, 'qualified_candidate_replay.json');
        assert(isfile(qual_path), 'MissingQualifiedReceipt: %s', qual_path);
        qual = jsondecode(fileread(qual_path));
        wall_clock_s = double(qual.elapsed_s);
        qualification = string(qual.qualification);
        report.qualified_receipt = qual_path;
    end

    cand_path = fullfile(evidence_dir, 'returned-candidate.json');
    receipt_path = fullfile(evidence_dir, 'receipt.json');
    replay_npz = fullfile(evidence_dir, 'returned-replay.npz');
    assert(isfile(cand_path) && isfile(receipt_path) && isfile(replay_npz), ...
        'MissingRun102Inputs');

    cand = jsondecode(fileread(cand_path));
    receipt = jsondecode(fileread(receipt_path));

    opts = struct();
    opts.run_id = run_id;
    opts.issue = "#10347";
    host_name = getenv('COMPUTERNAME');
    if isempty(host_name)
        host_name = char(java.net.InetAddress.getLocalHost().getHostName());
    end
    opts.host = string(host_name);
    opts.machine = string(host_name);
    opts.matlab_release = string(rel);
    opts.matlab_version = string(version);
    opts.model_sha256 = string(cand.model_sha256);
    opts.candidate_sha256 = string(receipt.returned_sha256);
    opts.replay_npz_sha256 = "";
    % Prefer a precomputed SHA from committed Python manifest when present.
    existing = fullfile(evidence_dir, 'run_manifest.json');
    if isfile(existing)
        prev = jsondecode(fileread(existing));
        if isfield(prev, 'replay_npz_sha256')
            opts.replay_npz_sha256 = string(prev.replay_npz_sha256);
        end
    end
    if strlength(strtrim(opts.replay_npz_sha256)) ~= 64
        error('ReplayShaRequired: commit run_manifest.json with replay_npz_sha256 first');
    end
    opts.wall_clock_s = wall_clock_s;
    opts.qualification = qualification;
    opts.evidence_dir = string(evidence_rel);
    opts.artifacts = struct( ...
        'candidate_npz', "candidate.npz", ...
        'playback_gif', "playback.gif", ...
        'returned_replay_npz', "returned-replay.npz", ...
        'qualified_replay_json', "qualified_candidate_replay.json", ...
        'replay_script', "replay_returned102_r2025b.m");

    manifest_path = fullfile(evidence_dir, 'run_manifest.json');
    write_run_manifest(manifest_path, opts);
    report.manifest_path = manifest_path;
end
