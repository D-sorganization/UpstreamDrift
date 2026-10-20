function receipt = gs3dx_grip_contact_run(repo_root, out_dir, opts)
%GS3DX_GRIP_CONTACT_RUN  Build/load GS3DX_GripContact, run all styles & scenarios, write receipt (#13334).
%
%   RECEIPT = GS3DX_GRIP_CONTACT_RUN(REPO_ROOT, OUT_DIR, OPTS) executes the
%   R2025b Simscape Multibody grip-contact qualification suite:
%     1. Asserts MATLAB release 2025b;
%     2. Builds (or loads) GS3DX_GripContact via GS3DX_BUILD_GRIP_CONTACT;
%     3. Simulates every style ('overlap', 'interlock', 'ten_finger') and
%        every scenario ('hold', 'short_swing') on the canonical 2 ms grid;
%     4. Evaluates frame probes at t = 0 against Python-exported poses;
%     5. Evaluates force-sign checks (normal force pushes club away from
%        each hand contact site);
%     6. Runs a timestep/tolerance convergence check on short_swing;
%     7. Runs a save/close/reload reproducibility check;
%     8. Writes versioned series JSON files (simscape-grip-contact-series/v1)
%        and gs3dx_grip_contact_receipt.json (simscape-grip-contact-receipt/v1)
%        to OUT_DIR.

    arguments
        repo_root = ""
        out_dir = ""
        opts.models_dir = ""
        opts.rebuild (1,1) logical = true
    end

    assert(string(version('-release')) == "2025b", ...
        'gs3dx:grip_contact:WrongRelease', ...
        'MATLAB R2025b is required; running on %s.', version('-release'));

    [repo_root_str, out_dir_str, models_dir_str, tools_dir, gen_dir] = ...
        local_resolve_dirs(repo_root, out_dir, opts.models_dir);
    addpath(tools_dir);
    addpath(gen_dir);

    if ~isfolder(out_dir_str)
        mkdir(out_dir_str);
    end
    if ~isfolder(models_dir_str)
        mkdir(models_dir_str);
    end

    scenarios_file = fullfile(gen_dir, 'gs3dx_grip_contact_scenarios.json');
    params_file = fullfile(gen_dir, 'gs3dx_grip_contact_params.json');
    kernel_file = fullfile(gen_dir, 'gs3dx_grip_contact_kernel.m');
    sc_data = jsondecode(fileread(scenarios_file));
    kp_data = jsondecode(fileread(params_file));

    names = gs3dx_names();
    mdl = char(names.variants.grip_contact);
    slx_path = fullfile(models_dir_str, [mdl '.slx']);

    if opts.rebuild || ~isfile(slx_path)
        build_rep = gs3dx_build_grip_contact(repo_root_str, models_dir_str, ...
            overwrite=true, save_model=true);
        budget = build_rep.budget;
    else
        load_system(slx_path);
        budget = gs3dx_block_budget(mdl, compiled=true);
    end
    if ~bdIsLoaded(mdl)
        load_system(slx_path);
    end
    cleanup = onCleanup(@() local_safe_close(mdl));

    styles = {'overlap', 'interlock', 'ten_finger'};
    scenarios = {'hold', 'short_swing'};

    max_pos_err_m = 0.0;
    max_rot_err_fro = 0.0;
    min_normal_push_dot_n = inf;
    baseline_ref_run = struct();

    for s_idx = 1:numel(styles)
        style_name = styles{s_idx};
        for c_idx = 1:numel(scenarios)
            scenario_name = scenarios{c_idx};
            sim_res = local_simulate_one(mdl, sc_data, kp_data, style_name, scenario_name, ...
                sc_data.solver.rel_tol, sc_data.solver.abs_tol, sc_data.solver.max_step_s);

            [pos_err, rot_err] = local_probe_frames_t0(sim_res, sc_data.scenarios.(scenario_name));
            max_pos_err_m = max(max_pos_err_m, pos_err);
            max_rot_err_fro = max(max_rot_err_fro, rot_err);

            if strcmp(scenario_name, 'hold')
                push_dot = local_check_force_signs(sim_res, kp_data, style_name);
                min_normal_push_dot_n = min(min_normal_push_dot_n, push_dot);
            end

            if strcmp(style_name, 'overlap') && strcmp(scenario_name, 'short_swing')
                baseline_ref_run = sim_res;
            end

            series_payload = local_build_series_payload(sim_res, sc_data, kp_data, style_name, scenario_name);
            series_file = fullfile(out_dir_str, ...
                sprintf('gs3dx_grip_contact_%s_%s.json', style_name, scenario_name));
            local_write_json(series_file, series_payload);
        end
    end

    % Timestep / tolerance convergence check on overlap + short_swing
    refined_run = local_simulate_one(mdl, sc_data, kp_data, 'overlap', 'short_swing', ...
        sc_data.solver.refined_rel_tol, sc_data.solver.refined_abs_tol, sc_data.solver.refined_max_step_s);
    f_base = baseline_ref_run.F_club_w;
    f_ref = refined_run.F_club_w;
    peak_f = max(sqrt(sum(f_ref.^2, 2)));
    max_rel_f_diff = max(sqrt(sum((f_base - f_ref).^2, 2))) / max(peak_f, 1e-9);
    conv_passed = max_rel_f_diff <= 0.02;

    % Save / close / reload check on overlap + short_swing
    save_system(mdl, slx_path);
    close_system(mdl, 0);
    load_system(slx_path);
    reload_run = local_simulate_one(mdl, sc_data, kp_data, 'overlap', 'short_swing', ...
        sc_data.solver.rel_tol, sc_data.solver.abs_tol, sc_data.solver.max_step_s);
    max_abs_f_diff = max(abs(baseline_ref_run.F_club_w(:) - reload_run.F_club_w(:)));
    max_abs_m_diff = max(abs(baseline_ref_run.M_club_w(:) - reload_run.M_club_w(:)));
    reload_passed = (max_abs_f_diff <= 1e-11) && (max_abs_m_diff <= 1e-11);

    probes_passed = (max_pos_err_m <= 1e-8) && (max_rot_err_fro <= 1e-8);
    signs_passed = isfinite(min_normal_push_dot_n) && (min_normal_push_dot_n > 0.0);

    receipt = struct( ...
        'schema_version', 'simscape-grip-contact-receipt/v1', ...
        'matlab_release', '2025b', ...
        'model_name', mdl, ...
        'model_sha256', local_sha256_file(slx_path), ...
        'input_sha256', local_sha256_file(scenarios_file), ...
        'kernel_sha256', local_sha256_file(kernel_file), ...
        'solver', struct( ...
            'solver_name', char(sc_data.solver.solver_name), ...
            'max_step_s', double(sc_data.solver.max_step_s), ...
            'rel_tol', double(sc_data.solver.rel_tol), ...
            'abs_tol', double(sc_data.solver.abs_tol)), ...
        'block_budget', struct( ...
            'uncompiled_total', double(budget.uncompiled_total), ...
            'compiled_total', double(budget.compiled_total), ...
            'license_ceiling', double(names.license_block_limit)), ...
        'timestep_convergence', struct( ...
            'baseline_max_step_s', double(sc_data.solver.max_step_s), ...
            'refined_max_step_s', double(sc_data.solver.refined_max_step_s), ...
            'max_rel_force_diff', double(max_rel_f_diff), ...
            'passed', logical(conv_passed)), ...
        'save_reload_check', struct( ...
            'max_abs_force_diff_n', double(max_abs_f_diff), ...
            'max_abs_moment_diff_nm', double(max_abs_m_diff), ...
            'passed', logical(reload_passed)), ...
        'frame_probes', struct( ...
            'max_pos_err_m', double(max_pos_err_m), ...
            'max_rot_err_fro', double(max_rot_err_fro), ...
            'passed', logical(probes_passed)), ...
        'force_sign_checks', struct( ...
            'min_normal_push_dot_n', double(min_normal_push_dot_n), ...
            'passed', logical(signs_passed)));

    receipt_file = fullfile(out_dir_str, 'gs3dx_grip_contact_receipt.json');
    local_write_json(receipt_file, receipt);
end

function [repo_root_str, out_dir_str, models_dir_str, tools_dir, gen_dir] = ...
    local_resolve_dirs(repo_root, out_dir, models_dir)
    here = fileparts(mfilename('fullpath'));
    repo_root_str = char(repo_root);
    if isempty(strtrim(repo_root_str))
        repo_root_str = char(fullfile(here, '..', '..', '..', '..', '..', '..'));
    end
    gs3dx_dir = fullfile(repo_root_str, ...
        'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx');
    if ~isfolder(gs3dx_dir)
        gs3dx_dir = fullfile(here, '..');
    end
    out_dir_str = char(out_dir);
    if isempty(strtrim(out_dir_str))
        out_dir_str = fullfile(gs3dx_dir, 'receipts', 'grip_contact');
    end
    models_dir_str = char(models_dir);
    if isempty(strtrim(models_dir_str))
        models_dir_str = fullfile(gs3dx_dir, 'models');
    end
    tools_dir = fullfile(gs3dx_dir, 'tools');
    gen_dir = fullfile(tools_dir, 'generated');
end

function local_safe_close(mdl)
    if bdIsLoaded(mdl)
        close_system(mdl, 0);
    end
end

function res = local_simulate_one(mdl, sc_data, kp_data, style_name, scenario_name, rtol, atol, max_step)
    st = sc_data.styles.(style_name);
    sc = sc_data.scenarios.(scenario_name);
    c0 = sc.initial_club_state;
    hl0 = sc.initial_hand_states.L;
    hr0 = sc.initial_hand_states.R;
    t_grid = reshape(double(sc.time_s), [], 1);

    in = Simulink.SimulationInput(mdl);
    in = in.setVariable('GripStyleId', double(st.style_id), 'Workspace', mdl);
    in = in.setVariable('ClubInitPos', reshape(double(c0.position_m), 1, 3), 'Workspace', mdl);
    in = in.setVariable('ClubInitRot', double(c0.rotation_world_from_club), 'Workspace', mdl);
    in = in.setVariable('ClubInitVelLocal', reshape(double(c0.velocity_local_m_s), 1, 3), 'Workspace', mdl);
    in = in.setVariable('ClubInitOmegaLocal', reshape(double(c0.omega_local_rad_s), 1, 3), 'Workspace', mdl);
    in = in.setVariable('HandLInitPos', reshape(double(hl0.position_m), 1, 3), 'Workspace', mdl);
    in = in.setVariable('HandLInitRot', double(hl0.rotation_world_from_hand), 'Workspace', mdl);
    in = in.setVariable('HandRInitPos', reshape(double(hr0.position_m), 1, 3), 'Workspace', mdl);
    in = in.setVariable('HandRInitRot', double(hr0.rotation_world_from_hand), 'Workspace', mdl);
    in = in.setVariable('TrajTimeGrid', t_grid, 'Workspace', mdl);
    in = in.setVariable('HandLQRel', double(sc.hand_trajectories.L.q_rel), 'Workspace', mdl);
    in = in.setVariable('HandLB', double(sc.hand_trajectories.L.hermite_b), 'Workspace', mdl);
    in = in.setVariable('HandLC', double(sc.hand_trajectories.L.hermite_c), 'Workspace', mdl);
    in = in.setVariable('HandLD', double(sc.hand_trajectories.L.hermite_d), 'Workspace', mdl);
    in = in.setVariable('HandRQRel', double(sc.hand_trajectories.R.q_rel), 'Workspace', mdl);
    in = in.setVariable('HandRB', double(sc.hand_trajectories.R.hermite_b), 'Workspace', mdl);
    in = in.setVariable('HandRC', double(sc.hand_trajectories.R.hermite_c), 'Workspace', mdl);
    in = in.setVariable('HandRD', double(sc.hand_trajectories.R.hermite_d), 'Workspace', mdl);

    in = in.setModelParameter( ...
        'StopTime', num2str(sc.duration_s, 17), ...
        'RelTol', num2str(rtol, 17), ...
        'AbsTol', num2str(atol, 17), ...
        'MaxStep', num2str(max_step, 17), ...
        'SignalLogging', 'on', ...
        'SignalLoggingName', 'logsout', ...
        'ReturnWorkspaceOutputs', 'on');

    out = sim(in);
    assert(isempty(out.ErrorMessage), 'gs3dx:grip_contact:SimFailed', '%s', out.ErrorMessage);
    logs = out.logsout;

    res = struct();
    res.time_s = t_grid;
    res.club_p_w = local_sample_vec(logs, 'club_p_w', t_grid, 3);
    res.club_R_w = local_sample_mat33(logs, 'club_R_w', t_grid);
    res.club_v_w = local_sample_vec(logs, 'club_v_w', t_grid, 3);
    res.club_omega_w = local_sample_vec(logs, 'club_omega_w', t_grid, 3);
    res.hand_L_p_w = local_sample_vec(logs, 'hand_L_p_w', t_grid, 3);
    res.hand_L_R_w = local_sample_mat33(logs, 'hand_L_R_w', t_grid);
    res.hand_R_p_w = local_sample_vec(logs, 'hand_R_p_w', t_grid, 3);
    res.hand_R_R_w = local_sample_mat33(logs, 'hand_R_R_w', t_grid);

    res.F_club_w = local_sample_vec(logs, 'F_club_w', t_grid, 3);
    res.M_club_w = local_sample_vec(logs, 'M_club_w', t_grid, 3);
    res.F_hand_L_w = local_sample_vec(logs, 'F_hand_L_w', t_grid, 3);
    res.M_hand_L_w = local_sample_vec(logs, 'M_hand_L_w', t_grid, 3);
    res.F_hand_R_w = local_sample_vec(logs, 'F_hand_R_w', t_grid, 3);
    res.M_hand_R_w = local_sample_vec(logs, 'M_hand_R_w', t_grid, 3);

    n_max = double(kp_data.max_physical_pairs);
    res.pair_normal_n = local_sample_vec(logs, 'pair_normal_n', t_grid, n_max);
    res.pair_tangent_n = local_sample_vec(logs, 'pair_tangent_n', t_grid, n_max);
    res.pair_spin_nm = local_sample_vec(logs, 'pair_spin_nm', t_grid, n_max);
    res.pair_slip_speed_m_s = local_sample_vec(logs, 'pair_slip_speed_m_s', t_grid, n_max);
    res.pair_force_club_w = local_sample_mat(logs, 'pair_force_club_w', t_grid, [n_max, 3]);
    res.pair_moment_club_w = local_sample_mat(logs, 'pair_moment_club_w', t_grid, [n_max, 3]);
    res.pair_contact_point_w = local_sample_mat(logs, 'pair_contact_point_w', t_grid, [n_max, 3]);
    res.z_state_m = local_sample_vec(logs, 'z_state_m', t_grid, n_max);
end

function [pos_err, rot_err] = local_probe_frames_t0(res, sc)
    c0 = sc.initial_club_state;
    hl0 = sc.initial_hand_states.L;
    hr0 = sc.initial_hand_states.R;

    err_p_c = norm(res.club_p_w(1, :).' - double(c0.position_m(:)));
    err_p_l = norm(res.hand_L_p_w(1, :).' - double(hl0.position_m(:)));
    err_p_r = norm(res.hand_R_p_w(1, :).' - double(hr0.position_m(:)));
    pos_err = max([err_p_c, err_p_l, err_p_r]);

    err_r_c = norm(squeeze(res.club_R_w(1, :, :)) - double(c0.rotation_world_from_club), 'fro');
    err_r_l = norm(squeeze(res.hand_L_R_w(1, :, :)) - double(hl0.rotation_world_from_hand), 'fro');
    err_r_r = norm(squeeze(res.hand_R_R_w(1, :, :)) - double(hr0.rotation_world_from_hand), 'fro');
    rot_err = max([err_r_c, err_r_l, err_r_r]);
end

function min_dot = local_check_force_signs(res, kp_data, style_name)
    tbl = double(kp_data.styles.(style_name).pair_table);
    n_act = double(kp_data.styles.(style_name).active_pairs);
    p_c = res.club_p_w(1, :).';
    r_c = squeeze(res.club_R_w(1, :, :));
    p_hl = res.hand_L_p_w(1, :).';
    r_hl = squeeze(res.hand_L_R_w(1, :, :));
    p_hr = res.hand_R_p_w(1, :).';
    r_hr = squeeze(res.hand_R_R_w(1, :, :));

    p_b_l = p_c + r_c * double(kp_data.club_grip_frames.L.position_m(:));
    r_b_l = r_c * double(kp_data.club_grip_frames.L.rotation);
    p_b_r = p_c + r_c * double(kp_data.club_grip_frames.R.position_m(:));
    r_b_r = r_c * double(kp_data.club_grip_frames.R.rotation);

    min_dot = inf;
    for k = 1:n_act
        fn = res.pair_normal_n(1, k);
        if fn <= 0.0
            continue;
        end
        row = tbl(k, :);
        if row(2) == 1.0
            p_a = p_hl; r_a = r_hl; p_b = p_b_l; r_b = r_b_l;
        else
            p_a = p_hr; r_a = r_hr; p_b = p_b_r; r_b = r_b_r;
        end
        a_mid = p_a + r_a * (0.5 * (row(4:6).' + row(7:9).'));
        b0_w = p_b + r_b * row(11:13).';
        b1_w = p_b + r_b * row(14:16).';
        ab = b1_w - b0_w;
        t_proj = min(max(dot(a_mid - b0_w, ab) / dot(ab, ab), 0.0), 1.0);
        p_axis = b0_w + t_proj * ab;
        u_push = p_axis - a_mid;
        u_push = u_push / norm(u_push);
        f_club_k = squeeze(res.pair_force_club_w(1, k, :));
        min_dot = min(min_dot, dot(f_club_k(:), u_push));
    end
end

function payload = local_build_series_payload(res, sc_data, kp_data, style_name, scenario_name)
    st_meta = sc_data.styles.(style_name);
    tbl = double(kp_data.styles.(style_name).pair_table);
    n_t = numel(res.time_s);

    l_mask = (tbl(:, 1) > 0.0) & (tbl(:, 2) == 1.0);
    r_mask = (tbl(:, 1) > 0.0) & (tbl(:, 2) == 2.0);
    f_l = squeeze(sum(res.pair_force_club_w(:, l_mask, :), 2));
    m_l = squeeze(sum(res.pair_moment_club_w(:, l_mask, :), 2));
    f_r = squeeze(sum(res.pair_force_club_w(:, r_mask, :), 2));
    m_r = squeeze(sum(res.pair_moment_club_w(:, r_mask, :), 2));

    p_norm = struct();
    p_tang = struct();
    p_spin = struct();
    p_slip = struct();
    reasons = struct();

    phys_names = cellstr(st_meta.physical_pair_names);
    for idx = 1:numel(phys_names)
        pn = phys_names{idx};
        p_norm.(pn) = res.pair_normal_n(:, idx);
        p_tang.(pn) = res.pair_tangent_n(:, idx);
        p_spin.(pn) = res.pair_spin_nm(:, idx);
        p_slip.(pn) = res.pair_slip_speed_m_s(:, idx);
    end

    alloc_names = cellstr(st_meta.allocated_pair_names);
    nan_col = nan(n_t, 1);
    for idx = 1:numel(alloc_names)
        an = alloc_names{idx};
        p_norm.(an) = nan_col;
        p_tang.(an) = nan_col;
        p_spin.(an) = nan_col;
        p_slip.(an) = nan_col;
        reasons.(an) = ['Allocated hand-hand pair ' an ' is unidentifiable under rigid-hand kinematics (#13334).'];
    end

    payload = struct( ...
        'schema_version', 'simscape-grip-contact-series/v1', ...
        'matlab_release', '2025b', ...
        'engine', 'simscape_contact', ...
        'model_id', 'GS3DX_GripContact', ...
        'style', style_name, ...
        'scenario', scenario_name, ...
        'mapping_hash', char(st_meta.topology_mapping_hash), ...
        'time_s', res.time_s, ...
        'club_position_w_m', res.club_p_w, ...
        'club_rotation_w', res.club_R_w, ...
        'club_velocity_w_m_s', res.club_v_w, ...
        'club_omega_w_rad_s', res.club_omega_w, ...
        'hand_l_position_w_m', res.hand_L_p_w, ...
        'hand_l_rotation_w', res.hand_L_R_w, ...
        'hand_r_position_w_m', res.hand_R_p_w, ...
        'hand_r_rotation_w', res.hand_R_R_w, ...
        'net_club_force_w_n', res.F_club_w, ...
        'net_club_moment_origin_w_nm', res.M_club_w, ...
        'force_on_club_l_w_n', f_l, ...
        'moment_on_club_l_w_nm', m_l, ...
        'force_on_club_r_w_n', f_r, ...
        'moment_on_club_r_w_nm', m_r, ...
        'pair_normal_force_n', p_norm, ...
        'pair_tangential_force_n', p_tang, ...
        'pair_spin_moment_nm', p_spin, ...
        'pair_slip_speed_m_s', p_slip, ...
        'pair_labels', st_meta.pair_labels, ...
        'unavailable_reasons', reasons, ...
        'z_state_m', res.z_state_m);
end

function out = local_sample_vec(logs, sig_name, t_grid, d)
    ts = logs.get(sig_name).Values;
    t_raw = ts.Time(:);
    data = ts.Data;
    n_raw = numel(t_raw);
    mat = zeros(n_raw, d);
    if ndims(data) == 3
        for k = 1:n_raw
            mat(k, :) = reshape(data(:, :, k), 1, d);
        end
    else
        mat = reshape(data, n_raw, d);
    end
    [t_u, ia] = unique(t_raw, 'stable');
    mat_u = mat(ia, :);
    out = interp1(t_u, mat_u, t_grid, 'linear', 'extrap');
end

function out = local_sample_mat33(logs, sig_name, t_grid)
    ts = logs.get(sig_name).Values;
    t_raw = ts.Time(:);
    data = ts.Data;
    n_raw = numel(t_raw);
    flat = zeros(n_raw, 9);
    if ndims(data) == 3 && size(data, 1) == 3 && size(data, 2) == 3
        for k = 1:n_raw
            % Row-major flatten so reshape(..., 3, 3) in Python matches MATLAB 3x3
            m = data(:, :, k).';
            flat(k, :) = m(:).';
        end
    else
        for k = 1:n_raw
            m = reshape(data(k, :), 3, 3).';
            flat(k, :) = m(:).';
        end
    end
    [t_u, ia] = unique(t_raw, 'stable');
    flat_i = interp1(t_u, flat(ia, :), t_grid, 'linear', 'extrap');
    n_g = numel(t_grid);
    out = zeros(n_g, 3, 3);
    for k = 1:n_g
        out(k, :, :) = reshape(flat_i(k, :), 3, 3).';
    end
end

function out = local_sample_mat(logs, sig_name, t_grid, dims)
    ts = logs.get(sig_name).Values;
    t_raw = ts.Time(:);
    data = ts.Data;
    n_raw = numel(t_raw);
    r = dims(1);
    c = dims(2);
    flat = zeros(n_raw, r * c);
    if ndims(data) == 3 && size(data, 1) == r && size(data, 2) == c
        for k = 1:n_raw
            m = data(:, :, k);
            flat(k, :) = m(:).';
        end
    else
        flat = reshape(data, n_raw, r * c);
    end
    [t_u, ia] = unique(t_raw, 'stable');
    flat_i = interp1(t_u, flat(ia, :), t_grid, 'linear', 'extrap');
    n_g = numel(t_grid);
    out = zeros(n_g, r, c);
    for k = 1:n_g
        out(k, :, :) = reshape(flat_i(k, :), r, c);
    end
end

function hex_str = local_sha256_file(filepath)
    fid = fopen(filepath, 'rb');
    assert(fid ~= -1, 'gs3dx:grip_contact:FileOpenError', 'Cannot open %s', filepath);
    cleaner = onCleanup(@() fclose(fid));
    bytes = fread(fid, inf, '*uint8');
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(bytes);
    digest = typecast(md.digest(), 'uint8');
    hex_str = lower(reshape(dec2hex(digest, 2).', 1, []));
end

function local_write_json(filepath, payload)
    txt = jsonencode(payload, 'PrettyPrint', true);
    fid = fopen(filepath, 'w', 'n', 'UTF-8');
    assert(fid ~= -1, 'gs3dx:grip_contact:FileWriteError', 'Cannot write %s', filepath);
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '%s\n', txt);
end
