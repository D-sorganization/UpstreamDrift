function receipt = export_grip_wrench_fixture(repo, out_dir, opts)
%EXPORT_GRIP_WRENCH_FIXTURE  R2025b run of GolfSwing3D_Kinetic -> grip CSV fixture.
%
%   RECEIPT = EXPORT_GRIP_WRENCH_FIXTURE(REPO, OUT_DIR, OPTS) runs the
%   canonical Simscape model headless, flattens CombinedSignalBus with the
%   dataset generator's extractFromCombinedSignalBus (so column names match
%   every exported dataset CSV), keeps the per-hand grip columns, decimates
%   to OPTS.n_rows evenly spaced samples and writes
%   OUT_DIR/simscape_grip_wrench_fixture.csv plus a JSON receipt (#11715).
%
%   Per-hand loading is the hand ON the club in the world frame:
%     LWLogs.LHonClubFGlobal / LHonClubTGlobal at LWLogs.LHGlobalPosition
%     RWLogs.RHonClubFGlobal / RHonClubTGlobal at RWLogs.RHGlobalPosition
%   The receipt records the residual of the two-contact reduction
%     M_M = sum_h (r_h - r_M) x F_h + tau_L + tau_R,  r_M = (r_L + r_R) / 2
%   against the logged MomentandCoupleLogs.EquivalentMidpointCoupleGlobal.
%
%   OPTS fields: stop_time (default 0.30 s), n_rows (default 31).
%   Coefficients are the model workspace's own polynomial values, run through
%   simulate_with_coefficients (the single sanctioned forward call).
%
%   Preconditions:
%     - MATLAB R2025b only (asserted).
%     - REPO is an UpstreamDrift checkout; OUT_DIR is writable.
%   Postconditions:
%     - CSV and receipt JSON exist in OUT_DIR; no persistent MATLAB path or
%       preference is changed (no savepath).

    arguments
        repo (1,1) string
        out_dir (1,1) string
        opts (1,1) struct = struct()
    end
    if ~isfield(opts, 'stop_time'); opts.stop_time = 0.30; end
    if ~isfield(opts, 'n_rows');    opts.n_rows    = 31;   end
    assert(opts.stop_time > 0, 'BadStopTime: stop_time must be > 0');
    assert(opts.n_rows >= 2, 'BadRows: n_rows must be >= 2');

    rel = version('-release');
    assert(strcmp(rel, '2025b'), 'R2025bRequired: detected %s', rel);

    matlab_root = fullfile(repo, 'src', 'engines', 'Simscape_Multibody_Models', ...
        '3D_Golf_Model', 'matlab');
    model_dir = fullfile(matlab_root, 'src', 'model');
    addpath(genpath(fullfile(matlab_root, 'src')));
    addpath(fullfile(matlab_root, 'motion_matching', 'shared'));
    if ~isfolder(out_dir); mkdir(out_dir); end

    cache = fullfile(tempdir, 'ud_grip_fixture_cache');
    Simulink.fileGenControl('set', 'CacheFolder', cache, ...
        'CodeGenFolder', cache, 'createDir', true);

    % The single sanctioned forward call (simulate_with_coefficients), fed
    % the model workspace's own polynomial coefficients (its designed swing).
    model = 'GolfSwing3D_Kinetic';
    load_system(model);
    sim_opts = default_sim_options();
    sim_opts.simulation_time = opts.stop_time;
    sim_opts.fast_restart = false;
    sim_opts.retain_raw_output = true;
    sim_opts.stop_on_error = true;
    sim_opts.verbosity = 'Silent';
    theta = local_model_theta(model);
    tic;
    sim_out = simulate_with_coefficients(theta, sim_opts);
    elapsed_s = toc;
    assert(sim_out.solver_status == "success", 'SimFailed: %s', sim_out.solver_status);

    csb = sim_out.raw_output.CombinedSignalBus;
    table_all = [];
    evalc('table_all = extractFromCombinedSignalBus(csb);');
    assert(~isempty(table_all), 'ExtractFailed: CombinedSignalBus was empty');

    vec = {'LWLogs_LHGlobalPosition_', 'RWLogs_RHGlobalPosition_', ...
        'LWLogs_LHonClubFGlobal_', 'RWLogs_RHonClubFGlobal_', ...
        'LWLogs_LHonClubTGlobal_', 'RWLogs_RHonClubTGlobal_', ...
        'MidpointCalcsLogs_MPGlobalPosition_', ...
        'CalculatedSignalsLogs_TotalHandForceGlobal_', ...
        'CalculatedSignalsLogs_TotalHandTorqueGlobal_', ...
        'MomentandCoupleLogs_LHMOFonClubGlobal_', ...
        'MomentandCoupleLogs_RHMOFonClubGlobal_', ...
        'MomentandCoupleLogs_EquivalentMidpointCoupleGlobal_'};
    names = {'time'};
    for k = 1:numel(vec)
        names = [names, strcat(vec{k}, {'1', '2', '3'})]; %#ok<AGROW>
    end
    missing = setdiff(names, table_all.Properties.VariableNames);
    assert(isempty(missing), 'MissingColumns: %s', strjoin(missing, ', '));

    n_all = height(table_all);
    idx = unique(round(linspace(1, n_all, min(opts.n_rows, n_all))));
    fixture = table_all(idx, names);
    csv_path = fullfile(out_dir, 'simscape_grip_wrench_fixture.csv');
    writetable(fixture, csv_path);
    % LF line endings so the receipt hash survives git text=auto normalization.
    text = strrep(fileread(csv_path), sprintf('\r\n'), newline);
    fid = fopen(csv_path, 'w');
    fwrite(fid, text, 'char');
    fclose(fid);

    % Two-contact reduction residual over every logged sample (not just
    % the decimated rows), in the same units as the logged couple.
    g = @(p) table_all{:, strcat(p, {'1', '2', '3'})};
    r_l = g(vec{1}); r_r = g(vec{2});
    f_l = g(vec{3}); f_r = g(vec{4});
    t_l = g(vec{5}); t_r = g(vec{6});
    r_m = (r_l + r_r) / 2;
    couple = cross(r_l - r_m, f_l, 2) + cross(r_r - r_m, f_r, 2) + t_l + t_r;
    logged_couple = g(vec{12});
    net = f_l + f_r;
    logged_net = g(vec{8});

    receipt = struct();
    receipt.issue = '#11715';
    receipt.matlab_release = rel;
    receipt.matlab_version = version;
    receipt.model = model;
    receipt.model_sha256 = local_sha256(fullfile(model_dir, [model '.slx']));
    receipt.coefficient_source = 'model_workspace';
    receipt.theta_sha256 = local_sha256_bytes(typecast(theta(:), 'uint8'));
    receipt.stop_time_s = opts.stop_time;
    receipt.sim_wall_clock_s = elapsed_s;
    receipt.samples_logged = n_all;
    receipt.fixture_rows = numel(idx);
    receipt.fixture_csv = 'simscape_grip_wrench_fixture.csv';
    receipt.fixture_sha256 = local_sha256(csv_path);
    receipt.couple_residual_max_abs_nm = max(abs(couple - logged_couple), [], 'all');
    receipt.couple_logged_max_abs_nm = max(abs(logged_couple), [], 'all');
    receipt.midpoint_residual_max_abs_m = max(abs(r_m - g(vec{7})), [], 'all');
    receipt.net_force_residual_max_abs_n = max(abs(net - logged_net), [], 'all');
    receipt.net_force_logged_max_abs_n = max(abs(logged_net), [], 'all');
    receipt.convention = 'hand on club, world frame, SI';

    fid = fopen(fullfile(out_dir, 'simscape_grip_wrench_receipt.json'), 'w');
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '%s\n', jsonencode(receipt, 'PrettyPrint', true));
    fprintf('%s\n', jsonencode(receipt, 'PrettyPrint', true));
    close_system(model, 0);
end

function theta = local_model_theta(model)
%LOCAL_MODEL_THETA  [A..G] per joint from the model workspace, canonical order.
    info = getPolynomialParameterInfo();
    joints = string(info.joint_names);
    ws = get_param(model, 'ModelWorkspace');
    letters = 'ABCDEFG';
    theta = zeros(numel(joints) * 7, 1);
    for j = 1:numel(joints)
        for c = 1:7
            name = char(joints(j) + letters(c));
            assert(hasVariable(ws, name), 'MissingCoefficient: %s', name);
            value = getVariable(ws, name);
            if isa(value, 'Simulink.Parameter'); value = value.Value; end
            theta((j - 1) * 7 + c) = double(value);
        end
    end
end

function hex = local_sha256(path)
    fid = fopen(path, 'r');
    assert(fid > 0, 'CannotOpen: %s', path);
    bytes = fread(fid, inf, '*uint8');
    fclose(fid);
    hex = local_sha256_bytes(bytes);
end

function hex = local_sha256_bytes(bytes)
    md = java.security.MessageDigest.getInstance('SHA-256');
    digest = typecast(md.digest(bytes), 'uint8');
    hex = lower(reshape(dec2hex(digest, 2).', 1, []));
end
