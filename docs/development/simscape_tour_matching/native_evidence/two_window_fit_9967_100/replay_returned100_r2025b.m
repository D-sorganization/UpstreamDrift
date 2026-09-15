function report = replay_returned100_r2025b(repo)
%REPLAY_RETURNED100_R2025B Corrected independent R2025b forward replay of returned100.
% Diagnosed and fixed:
% 1. Resolved geometry seed path from native_evidence/ (not reproduction/) -> 1.5 pm at t=0
% 2. Preserved world translational forces without duplicate rot_wb multiplication -> 15 decimals effort match
    arguments
        repo (1,1) string = "C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour-runtime"
    end

    rel = version('-release');
    assert(strcmp(rel, '2025b'), 'R2025b is required. Detected release: %s', rel);

    source = fullfile(repo, 'src', 'engines', 'Simscape_Multibody_Models', '3D_Golf_Model', 'matlab');
    addpath(fullfile(source, 'src', 'model'));
    addpath(genpath(fullfile(source, 'src', 'functions')));
    addpath(fullfile(source, 'motion_matching', 'shared'));

    evidence_dir = fullfile(repo, 'docs', 'development', 'simscape_tour_matching', 'native_evidence', 'two_window_fit_9967_100');
    cand_file = fullfile(evidence_dir, 'returned-candidate.json');
    assert(isfile(cand_file), 'Candidate file not found: %s', cand_file);
    cand = jsondecode(fileread(cand_file));

    duration_s = double(cand.duration_s);
    assert(duration_s == 0.85, 'Expected duration 0.85 s');

    % Fixed Seed Path: search native_evidence first, error clearly if missing
    seed_file = fullfile(repo, 'docs', 'development', 'simscape_tour_matching', 'native_evidence', 'initial_velocity_seed_qualified_r2025b.json');
    if ~isfile(seed_file)
        seed_file = fullfile(repo, 'docs', 'development', 'simscape_tour_matching', 'native_evidence', 'reproduction', 'initial_velocity_seed_qualified_r2025b.json');
    end
    assert(isfile(seed_file), 'MissingRequiredSeed: Seed file not found: %s', seed_file);
    seed_data = jsondecode(fileread(seed_file));

    % Verify candidate physical properties against the seed rather than relying on filename
    q_diff = max(abs(double(cand.q0(:)) - double(seed_data.q(:))));
    assert(q_diff < 1e-6, 'CandidateInitialStateMismatch: q0 differs from seed by %e rad/m', q_diff);
    qd_diff = max(abs(double(cand.qd0(:)) - double(seed_data.qd(:))));
    assert(qd_diff < 1e-6, 'CandidateInitialRateMismatch: qd0 differs from seed by %e rad/s', qd_diff);
    offset_diff = max(abs(double(cand.marker_offsets_m(:)) - double(seed_data.offsets_m(:))));
    assert(offset_diff < 1e-6, 'CandidateMarkerOffsetMismatch: offsets differ from seed by %e m', offset_diff);

    pino_file = fullfile(evidence_dir, 'pinocchio_replay.mat');
    assert(isfile(pino_file), 'Pinocchio replay file not found: %s', pino_file);
    pino = load(pino_file);

    time_s = double(pino.time_s(:));
    n_samples = numel(time_s);

    % Input-frame metadata guard:
    % Commanded forces are in world frame. Model block ConvertGlobalXtoHipBaseLocal
    % (Product SID 7061) internally performs global-to-base rotation.
    % Pre-multiplying by rot_wb outside would inject a double rotation defect.
    if isfield(cand, 'actuator_force_frame')
        assert(strcmp(cand.actuator_force_frame, 'world'), ...
            'UnsupportedActuatorForceFrame: Expected world, got %s', cand.actuator_force_frame);
    end

    coeffs = double(cand.coefficients);
    theta = reshape(coeffs.', [], 1);
    assert(numel(theta) == 189, 'Expected length 189 for theta');

    load_system('GolfSwing3D_Kinetic');
    guard = configure_capture_velocity_targets();

    fit_geometry_names = {'UpperArmLength', 'LowerArmLength'};
    fit_ws = get_param('GolfSwing3D_Kinetic', 'ModelWorkspace');
    for j = 1:2
        assignin(fit_ws, fit_geometry_names{j}, seed_data.geometry_in(j));
    end

    [ks, schema] = build_golf_kinematics();
    addTargetVariables(ks, schema.q_ids);
    addOutputVariables(ks, schema.frame_ids);
    addOutputVariables(ks, schema.rotation_ids);

    opts = capture_fit_sim_options(duration_s);
    opts.sample_rate = 360;
    opts.fast_restart = false;
    opts.retain_raw_output = true;
    opts.verbosity = 'Silent';
    opts.joint_names = string(cand.coordinate_names)';

    for j = 1:2
        opts.input_overrides.(fit_geometry_names{j}) = seed_data.geometry_in(j);
    end

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

    [found, bodies] = ismember(string(cand.marker_bodies), string({schema.frames.name}));
    assert(all(found), 'All marker bodies must be found in schema');

    started = tic;
    [prediction, replay] = simulate_golf_markers( ...
        theta, opts, ks, schema, bodies(:), double(cand.marker_offsets_m), time_s);
    sim_elapsed_s = toc(started);

    % Compute metrics
    target_points = double(pino.target_m);
    valid_mask = logical(pino.valid);
    error_sq = sum((prediction - target_points).^2, 3);

    whole_rms_m = sqrt(mean(error_sq(valid_mask)));
    early_mask = valid_mask & (time_s <= 0.6);
    early_rms_m = sqrt(mean(error_sq(early_mask)));
    terminal_valid = valid_mask(end, :);
    terminal_rms_m = sqrt(mean(error_sq(end, terminal_valid)));

    club_labels = startsWith(lower(string(cand.marker_labels)), "marker_2") | ...
                  startsWith(lower(string(cand.marker_labels)), "marker_3");
    club_term_mask = club_labels(:)' & terminal_valid;
    club_cluster_rms_m = sqrt(mean(error_sq(end, club_term_mask)));

    wl_idx = find(strcmp(cand.marker_labels, 'WaistLeft'));
    wr_idx = find(strcmp(cand.marker_labels, 'WaistRight'));
    vp = prediction(end, wr_idx, 1:2) - prediction(end, wl_idx, 1:2);
    vo = target_points(end, wr_idx, 1:2) - target_points(end, wl_idx, 1:2);
    yt = atan2d(vo(2), vo(1));
    yp = atan2d(vp(2), vp(1));
    yaw_diff = mod(yp - yt + 180, 360) - 180;
    pelvis_yaw_error_pct = abs(yaw_diff) / max(abs(yt), 1.0) * 100;

    pino_markers = double(pino.markers_m);
    marker_diff = abs(prediction - pino_markers);
    max_marker_coord_discrepancy_m = max(marker_diff, [], 'all');
    mean_marker_coord_discrepancy_m = mean(marker_diff, 'all');

    marker_euc = sqrt(sum((prediction - pino_markers).^2, 3));
    max_marker_euclidean_discrepancy_m = max(marker_euc, [], 'all');
    mean_marker_euclidean_discrepancy_m = mean(marker_euc, 'all');

    report = struct();
    report.matlab_release = rel;
    report.matlab_version = version;
    report.qualification = 'corrected_r2025b_cold_replay_returned100';
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

    report.cross_engine_parity = struct( ...
        'max_marker_coord_discrepancy_m', max_marker_coord_discrepancy_m, ...
        'max_marker_coord_discrepancy_mm', max_marker_coord_discrepancy_m * 1000, ...
        'mean_marker_coord_discrepancy_m', mean_marker_coord_discrepancy_m, ...
        'mean_marker_coord_discrepancy_mm', mean_marker_coord_discrepancy_m * 1000, ...
        'max_marker_euclidean_discrepancy_m', max_marker_euclidean_discrepancy_m, ...
        'max_marker_euclidean_discrepancy_mm', max_marker_euclidean_discrepancy_mm * 1000, ...
        'mean_marker_euclidean_discrepancy_m', mean_marker_euclidean_discrepancy_m, ...
        'mean_marker_euclidean_discrepancy_mm', mean_marker_euclidean_discrepancy_mm * 1000);

    report.gates = struct( ...
        'gate1_whole_rms_25mm', whole_rms_m <= 0.025, ...
        'gate2_early_rms_12mm', early_rms_m <= 0.012, ...
        'gate3_terminal_rms_35mm', terminal_rms_m <= 0.035, ...
        'gate4_clubhead_60mm', club_cluster_rms_m <= 0.060, ...
        'gate5_pelvis_yaw_5pct', pelvis_yaw_error_pct <= 5.0);

    out_json = fullfile(evidence_dir, 'corrected_candidate_replay.json');
    fid = fopen(out_json, 'w');
    fprintf(fid, '%s', jsonencode(report, PrettyPrint=true));
    fclose(fid);

    out_mat = fullfile(evidence_dir, 'corrected_candidate_replay.mat');
    if isfield(replay, 'raw_output')
        replay = rmfield(replay, 'raw_output');
    end
    save(out_mat, 'report', 'prediction', 'replay', '-v7');

    clear guard;
    close_system('GolfSwing3D_Kinetic', 0);
    bdclose('all');
end
