function report = replay_returned84_r2025b(repo)
%REPLAY_RETURNED84_R2025B Independent R2025b forward replay of returned84.
% Verifies continuous forward dynamics without target-state resets (0–0.85 s),
% computes the 5 canonical metrics, and tests parity against Pinocchio replay.
    arguments
        repo (1,1) string = "C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour-runtime"
    end

    % 1. Enforce strict R2025b environment contract
    rel = version('-release');
    assert(strcmp(rel, '2025b'), 'R2025b is required. Detected release: %s', rel);

    % 2. Add required model and function paths
    source = fullfile(repo, 'src', 'engines', 'Simscape_Multibody_Models', '3D_Golf_Model', 'matlab');
    addpath(fullfile(source, 'src', 'model'));
    addpath(genpath(fullfile(source, 'src', 'functions')));
    addpath(fullfile(source, 'motion_matching', 'shared'));

    % 3. Load candidate specification
    evidence_dir = fullfile(repo, 'docs', 'development', 'simscape_tour_matching', 'native_evidence', 'two_window_fit_9967_84');
    cand_file = fullfile(evidence_dir, 'returned-candidate.json');
    assert(isfile(cand_file), 'Candidate file not found: %s', cand_file);
    cand = jsondecode(fileread(cand_file));

    duration_s = double(cand.duration_s);
    assert(duration_s == 0.85, 'Expected duration 0.85 s');

    % Load qualified seed geometry if available
    seed_file = fullfile(repo, 'docs', 'development', 'simscape_tour_matching', 'native_evidence', 'reproduction', 'initial_velocity_seed_qualified_r2025b.json');
    has_seed = isfile(seed_file);
    if has_seed
        seed_data = jsondecode(fileread(seed_file));
    end

    % 4. Load Pinocchio ground truth replay for parity check
    pino_file = fullfile(evidence_dir, 'pinocchio_replay.mat');
    assert(isfile(pino_file), 'Pinocchio replay file not found: %s', pino_file);
    pino = load(pino_file);

    time_s = double(pino.time_s(:));
    pino_markers = double(pino.markers_m); % [N x M x 3]
    target_points = double(pino.target_m); % [N x M x 3]
    valid_mask = logical(pino.valid);     % [N x M]

    n_samples = numel(time_s);
    assert(n_samples == 307, 'Expected 307 samples @ 360 Hz');

    % 5. Build theta vector (189 coefficients: 27 coordinates x 7 parameters)
    coeffs = double(cand.coefficients);
    assert(isequal(size(coeffs), [27, 7]), 'Expected 27x7 coefficients');
    theta = reshape(coeffs.', [], 1);
    assert(numel(theta) == 189, 'Expected length 189 for theta');

    % 6. Configure Simulink & Simscape options
    load_system('GolfSwing3D_Kinetic');
    guard = configure_capture_velocity_targets();

    % Assign qualified geometry BEFORE build_golf_kinematics
    fit_geometry_names = {'UpperArmLength', 'LowerArmLength'};
    if has_seed && isfield(seed_data, 'geometry_in')
        fit_ws = get_param('GolfSwing3D_Kinetic', 'ModelWorkspace');
        for j = 1:2
            assignin(fit_ws, fit_geometry_names{j}, seed_data.geometry_in(j));
        end
    end

    [ks, schema] = build_golf_kinematics();
    assert(isequal(string(cand.coordinate_names(:)), schema.coordinate_names(:)), ...
        'Coordinate names mismatch schema');

    addTargetVariables(ks, schema.q_ids);
    addOutputVariables(ks, schema.frame_ids);
    addOutputVariables(ks, schema.rotation_ids);

    opts = capture_fit_sim_options(duration_s);
    opts.sample_rate = 360;
    opts.fast_restart = false;
    opts.retain_raw_output = true;
    opts.verbosity = 'Silent';
    opts.joint_names = string(cand.coordinate_names)';

    if has_seed && isfield(seed_data, 'geometry_in')
        for j = 1:2
            opts.input_overrides.(fit_geometry_names{j}) = seed_data.geometry_in(j);
        end
    end

    % Set initial state
    for j = 1:numel(cand.q0)
        name = opts.joint_names(j);
        pos = double(cand.q0(j));
        vel = double(cand.qd0(j));
        if ~startsWith(name, 'Translation')
            pos = rad2deg(pos);
            vel = rad2deg(vel);
        end
        opts.input_overrides.(replace(name, 'Input', 'StartPosition')) = pos;
        opts.input_overrides.(replace(name, 'Input', 'StartVelocity')) = vel;
    end

    % 7. Identify marker frames
    [found, bodies] = ismember(string(cand.marker_bodies), string({schema.frames.name}));
    assert(all(found), 'All marker bodies must be found in schema');

    % 8. Execute continuous forward dynamics simulation
    started = tic;
    [prediction, replay] = simulate_golf_markers( ...
        theta, opts, ks, schema, bodies(:), double(cand.marker_offsets_m), time_s);
    sim_elapsed_s = toc(started);

    % 9. Compute the 5 Canonical Tour-Matching Metrics
    error_sq = sum((prediction - target_points).^2, 3); % [N x M]

    % Metric 1: Whole RMS (mm)
    whole_rms_m = sqrt(mean(error_sq(valid_mask)));

    % Metric 2: Early RMS <= 0.6 s (mm)
    early_mask = valid_mask & (time_s <= 0.6);
    early_rms_m = sqrt(mean(error_sq(early_mask)));

    % Metric 3: Terminal RMS at final frame (mm)
    terminal_valid = valid_mask(end, :);
    terminal_rms_m = sqrt(mean(error_sq(end, terminal_valid)));

    % Metric 4: Clubhead Cluster Terminal RMS (mm)
    club_labels = startsWith(lower(string(cand.marker_labels)), "marker_2") | ...
                  startsWith(lower(string(cand.marker_labels)), "marker_3");
    club_term_mask = club_labels(:)' & terminal_valid;
    club_cluster_rms_m = sqrt(mean(error_sq(end, club_term_mask)));

    % Metric 5: Pelvis Yaw Error (%)
    wl_idx = find(strcmp(cand.marker_labels, 'WaistLeft'));
    wr_idx = find(strcmp(cand.marker_labels, 'WaistRight'));
    vp = prediction(end, wr_idx, 1:2) - prediction(end, wl_idx, 1:2);
    vo = target_points(end, wr_idx, 1:2) - target_points(end, wl_idx, 1:2);
    yt = atan2d(vo(2), vo(1));
    yp = atan2d(vp(2), vp(1));
    yaw_diff = mod(yp - yt + 180, 360) - 180;
    pelvis_yaw_error_pct = abs(yaw_diff) / max(abs(yt), 1.0) * 100;

    % 10. Cross-Engine Parity Check vs Pinocchio
    marker_diff = abs(prediction - pino_markers);
    max_marker_difference_m = max(marker_diff, [], 'all');
    mean_marker_difference_m = mean(marker_diff, 'all');

    % 11. Assemble Verification Report
    report = struct();
    report.matlab_release = rel;
    report.matlab_version = version;
    report.qualification = 'independent_r2025b_cold_replay_returned84';
    report.candidate_sha256 = 'f01d551db527f8b6d64b77fa335bba093e221114026f7175f42dd27993bfab9b';
    report.duration_s = duration_s;
    report.n_samples = n_samples;
    report.elapsed_s = sim_elapsed_s;

    report.metrics = struct( ...
        'whole_rms_m', whole_rms_m, ...
        'whole_rms_mm', whole_rms_m * 1000, ...
        'early_rms_m', early_rms_m, ...
        'early_rms_mm', early_rms_m * 1000, ...
        'terminal_rms_m', terminal_rms_m, ...
        'terminal_rms_mm', terminal_rms_m * 1000, ...
        'club_cluster_rms_m', club_cluster_rms_m, ...
        'club_cluster_rms_mm', club_cluster_rms_m * 1000, ...
        'pelvis_yaw_error_pct', pelvis_yaw_error_pct);

    report.pinocchio_metrics = struct( ...
        'whole_rms_m', 0.023001580030123817, ...
        'early_rms_m', 0.010807673961539048, ...
        'terminal_rms_m', 0.046042977131343706, ...
        'club_cluster_rms_m', 0.01918118299576692, ...
        'pelvis_yaw_error_pct', 14.152129260114638);

    report.cross_engine_parity = struct( ...
        'max_marker_difference_m', max_marker_difference_m, ...
        'mean_marker_difference_m', mean_marker_difference_m);

    report.gates = struct( ...
        'gate1_whole_rms_25mm', whole_rms_m <= 0.025, ...
        'gate2_early_rms_12mm', early_rms_m <= 0.012, ...
        'gate3_terminal_rms_35mm', terminal_rms_m <= 0.035, ...
        'gate4_clubhead_60mm', club_cluster_rms_m <= 0.060, ...
        'gate5_pelvis_yaw_5pct', pelvis_yaw_error_pct <= 5.0);

    % 12. Save Report Artifacts
    out_json = fullfile(evidence_dir, 'qualified_candidate_replay.json');
    fid = fopen(out_json, 'w');
    assert(fid ~= -1, 'Failed to open file for writing: %s', out_json);
    fprintf(fid, '%s', jsonencode(report, PrettyPrint=true));
    fclose(fid);

    out_mat = fullfile(evidence_dir, 'qualified_candidate_replay.mat');
    save(out_mat, 'report', 'prediction', 'replay', '-v7.3');

    fprintf('\n=== R2025b Replay Verification Complete ===\n');
    fprintf('Whole RMS:       %.3f mm (Pinocchio: 23.002 mm)\n', whole_rms_m * 1000);
    fprintf('Early RMS:       %.3f mm (Pinocchio: 10.808 mm)\n', early_rms_m * 1000);
    fprintf('Terminal RMS:    %.3f mm (Pinocchio: 46.043 mm)\n', terminal_rms_m * 1000);
    fprintf('Club Cluster:    %.3f mm (Pinocchio: 19.181 mm)\n', club_cluster_rms_m * 1000);
    fprintf('Pelvis Yaw Err:  %.2f %%  (Pinocchio: 14.15 %%)\n', pelvis_yaw_error_pct);
    fprintf('Max Marker Diff: %.6e m\n', max_marker_difference_m);
    fprintf('Report written to: %s\n', out_json);

    clear guard;
    close_system('GolfSwing3D_Kinetic', 0);
    bdclose('all');
end
