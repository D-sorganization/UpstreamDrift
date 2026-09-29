function audit_results = run_solver_convergence_matrix(repo)
%RUN_SOLVER_CONVERGENCE_MATRIX Runs tolerance and step-size convergence matrix on candidate 100.
    arguments
        repo (1,1) string = "C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour-runtime"
    end

    rel = version('-release');
    assert(strcmp(rel, '2025b'), 'R2025b required. Detected: %s', rel);

    source = fullfile(repo, 'src', 'engines', 'Simscape_Multibody_Models', '3D_Golf_Model', 'matlab');
    addpath(fullfile(source, 'src', 'model'));
    addpath(genpath(fullfile(source, 'src', 'functions')));
    addpath(fullfile(source, 'motion_matching', 'shared'));

    evidence_dir = fullfile(repo, 'docs', 'development', 'simscape_tour_matching', 'native_evidence', 'two_window_fit_9967_100');
    cand_file = fullfile(evidence_dir, 'returned-candidate.json');
    cand = jsondecode(fileread(cand_file));

    seed_file = fullfile(repo, 'docs', 'development', 'simscape_tour_matching', 'native_evidence', 'initial_velocity_seed_qualified_r2025b.json');
    if ~isfile(seed_file)
        seed_file = fullfile(repo, 'docs', 'development', 'simscape_tour_matching', 'native_evidence', 'reproduction', 'initial_velocity_seed_qualified_r2025b.json');
    end
    seed_data = jsondecode(fileread(seed_file));

    pino_file = fullfile(evidence_dir, 'pinocchio_replay.mat');
    pino = load(pino_file);
    time_s = double(pino.time_s(:));
    duration_s = double(cand.duration_s);

    coeffs = double(cand.coefficients);
    theta = reshape(coeffs.', [], 1);

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

    [found, bodies] = ismember(string(cand.marker_bodies), string({schema.frames.name}));
    assert(all(found), 'All marker bodies must be found in schema');

    % Define convergence configurations to evaluate
    % 1. ode23t, RelTol 1e-3, MaxStep auto (baseline)
    % 2. ode23t, RelTol 1e-4, MaxStep auto
    % 3. ode23t, RelTol 1e-6, MaxStep auto
    % 4. ode23t, RelTol 1e-8, MaxStep auto
    % 5. ode23t, RelTol 1e-6, MaxStep 1/720 (0.0013889 s)
    % 6. ode23t, RelTol 1e-6, MaxStep 1/1440 (0.0006944 s)
    % 7. ode15s, RelTol 1e-3, MaxStep auto
    % 8. ode15s, RelTol 1e-6, MaxStep auto
    % 9. ode15s, RelTol 1e-6, MaxStep 1/1440
    configs = { ...
        struct('name', 'ode23t_tol1e3_auto', 'solver', 'ode23t', 'reltol', '1e-3', 'abstol', '1e-6', 'maxstep', 'auto'), ...
        struct('name', 'ode23t_tol1e4_auto', 'solver', 'ode23t', 'reltol', '1e-4', 'abstol', '1e-7', 'maxstep', 'auto'), ...
        struct('name', 'ode23t_tol1e6_auto', 'solver', 'ode23t', 'reltol', '1e-6', 'abstol', '1e-9', 'maxstep', 'auto'), ...
        struct('name', 'ode23t_tol1e8_auto', 'solver', 'ode23t', 'reltol', '1e-8', 'abstol', '1e-11', 'maxstep', 'auto'), ...
        struct('name', 'ode23t_tol1e6_step720', 'solver', 'ode23t', 'reltol', '1e-6', 'abstol', '1e-9', 'maxstep', num2str(1/720, '%.8f')), ...
        struct('name', 'ode23t_tol1e6_step1440', 'solver', 'ode23t', 'reltol', '1e-6', 'abstol', '1e-9', 'maxstep', num2str(1/1440, '%.8f')), ...
        struct('name', 'ode15s_tol1e3_auto', 'solver', 'ode15s', 'reltol', '1e-3', 'abstol', '1e-6', 'maxstep', 'auto'), ...
        struct('name', 'ode15s_tol1e6_auto', 'solver', 'ode15s', 'reltol', '1e-6', 'abstol', '1e-9', 'maxstep', 'auto'), ...
        struct('name', 'ode15s_tol1e6_step1440', 'solver', 'ode15s', 'reltol', '1e-6', 'abstol', '1e-9', 'maxstep', num2str(1/1440, '%.8f')) ...
    };

    audit_results = struct();
    audit_results.configs = cell(numel(configs), 1);

    for c = 1:numel(configs)
        cfg = configs{c};
        fprintf('Evaluating config %d/%d: %s (solver=%s, reltol=%s, maxstep=%s)...\n', ...
            c, numel(configs), cfg.name, cfg.solver, cfg.reltol, cfg.maxstep);

        opts = capture_fit_sim_options(duration_s);
        opts.sample_rate = 360;
        opts.fast_restart = false;
        opts.retain_raw_output = true;
        opts.verbosity = 'Silent';
        opts.joint_names = string(cand.coordinate_names)';
        opts.solver = cfg.solver;

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

        opts.model_parameters = struct( ...
            'Solver', cfg.solver, ...
            'RelTol', cfg.reltol, ...
            'AbsTol', cfg.abstol, ...
            'MaxStep', cfg.maxstep);

        t_sim_start = tic;
        [prediction, replay] = simulate_golf_markers( ...
            theta, opts, ks, schema, bodies(:), double(cand.marker_offsets_m), time_s);
        sim_elapsed_s = toc(t_sim_start);

        % Compute metrics vs target points
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

        % Cross-engine discrepancy vs pinocchio
        pino_markers = double(pino.markers_m);
        marker_diff = abs(prediction - pino_markers);
        max_coord_diff_m = max(marker_diff, [], 'all');
        mean_coord_diff_m = mean(marker_diff, 'all');

        marker_euc = sqrt(sum((prediction - pino_markers).^2, 3));
        max_euc_diff_m = max(marker_euc, [], 'all');
        mean_euc_diff_m = mean(marker_euc, 'all');

        % Pelvis yaw
        wl_idx = find(strcmp(cand.marker_labels, 'WaistLeft'));
        wr_idx = find(strcmp(cand.marker_labels, 'WaistRight'));
        vp = prediction(end, wr_idx, 1:2) - prediction(end, wl_idx, 1:2);
        vo = target_points(end, wr_idx, 1:2) - target_points(end, wl_idx, 1:2);
        yt = atan2d(vo(2), vo(1));
        yp = atan2d(vp(2), vp(1));
        yaw_diff = mod(yp - yt + 180, 360) - 180;
        pelvis_yaw_error_pct = abs(yaw_diff) / max(abs(yt), 1.0) * 100;

        res_entry = struct();
        res_entry.name = cfg.name;
        res_entry.solver = cfg.solver;
        res_entry.reltol = cfg.reltol;
        res_entry.abstol = cfg.abstol;
        res_entry.maxstep = cfg.maxstep;
        res_entry.elapsed_s = sim_elapsed_s;
        res_entry.whole_rms_mm = whole_rms_m * 1000;
        res_entry.early_rms_mm = early_rms_m * 1000;
        res_entry.terminal_rms_mm = terminal_rms_m * 1000;
        res_entry.club_cluster_rms_mm = club_cluster_rms_m * 1000;
        res_entry.pelvis_yaw_error_pct = pelvis_yaw_error_pct;
        res_entry.max_coord_diff_mm = max_coord_diff_m * 1000;
        res_entry.max_euc_diff_mm = max_euc_diff_m * 1000;
        res_entry.mean_euc_diff_mm = mean_euc_diff_m * 1000;
        res_entry.prediction = prediction;
        res_entry.q = replay.q;
        res_entry.qd = replay.qd;
        res_entry.qdd = replay.qdd;
        res_entry.tau = replay.tau;

        audit_results.configs{c} = res_entry;
        fprintf('  -> Done in %.2f s: Whole=%.2f mm, Early=%.2f mm, Term=%.2f mm, MaxEuc=%.2f mm\n', ...
            sim_elapsed_s, whole_rms_m*1000, early_rms_m*1000, terminal_rms_m*1000, max_euc_diff_m*1000);
    end

    % Save results
    out_mat = fullfile(evidence_dir, 'solver_convergence_simscape_results.mat');
    save(out_mat, 'audit_results', 'time_s', '-v7');
    fprintf('Saved all convergence trajectories to %s\n', out_mat);

    clear guard;
    close_system('GolfSwing3D_Kinetic', 0);
    bdclose('all');
end
