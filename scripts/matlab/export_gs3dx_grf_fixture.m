function receipt = export_gs3dx_grf_fixture(repo, out_dir, opts)
%EXPORT_GS3DX_GRF_FIXTURE  R2025b GS3DX_FullBodyContact run -> per-contact GRF CSV.
%
%   RECEIPT = EXPORT_GS3DX_GRF_FIXTURE(REPO, OUT_DIR, OPTS) simulates the
%   exploratory contact model (three sole spheres per foot on one ground
%   plane) with GS3DX_CONTACT_CHECK, places each sphere's ground contact
%   point in the world from the logged ankle pose, and writes
%   OUT_DIR/simscape_gs3dx_grf_fixture.csv plus a JSON receipt (#11709).
%
%   Columns (world frame, Z-up, SI; force = ground ON foot):
%     time
%     GroundContactLogs_<C>Force_1..3, GroundContactLogs_<C>Point_1..3
%         for C in LHeel, LToeIn, LToeOut, RHeel, RToeIn, RToeOut
%     COMLogs_GlobalPosition_1..3   whole-mechanism centre of mass
%     GroundContactLogs_GroundHeight  world Z of the (level) ground plane
%   The contact point is the sphere centre less one radius along +Z (the
%   sphere's lowest point); the receipt reports its height above the plane
%   for every loaded contact as a geometry check.
%
%   OPTS fields: stop_time (0.30 s), n_rows (31), rest (true: the standing
%   test; the impact drive tips the body over, see GROUND_CONTACT.md),
%   lock_joints (false: true locks every joint but the pelvis root with
%   GS3DX_LOCK_JOINTS, in memory, so the body stands as one rigid body on
%   its sole contacts: the stance compared with the Python engines).
%
%   Preconditions: MATLAB R2025b (asserted); REPO is an UpstreamDrift checkout.
%   Postconditions: CSV (LF line endings) and receipt exist in OUT_DIR; the
%   Newton momentum check of the run passed (asserted).

    arguments
        repo (1,1) string
        out_dir (1,1) string
        opts (1,1) struct = struct()
    end
    if ~isfield(opts, 'stop_time'); opts.stop_time = 0.30; end
    if ~isfield(opts, 'n_rows');    opts.n_rows    = 31;   end
    if ~isfield(opts, 'rest');      opts.rest      = true; end
    if ~isfield(opts, 'lock_joints'); opts.lock_joints = false; end
    assert(opts.stop_time > 0 && opts.n_rows >= 2, 'BadOpts: stop_time > 0, n_rows >= 2');

    rel = version('-release');
    assert(strcmp(rel, '2025b'), 'R2025bRequired: detected %s', rel);

    root = fullfile(repo, 'src', 'engines', 'Simscape_Multibody_Models', ...
        '3D_Golf_Model', 'matlab', 'exploratory_gs3dx');
    addpath(root);
    info = gs3dx_setup();
    mdl = char(gs3dx_names().variants.contact);
    geo = gs3dx_contact_geometry(mdl);
    if ~isfolder(out_dir); mkdir(out_dir); end

    locked = {};
    edited = {};
    if opts.lock_joints
        load_system(mdl);   % GS3DX_CONTACT_CHECK runs the loaded, locked model
        [locked, edited] = gs3dx_lock_joints(mdl);
    end
    discard = onCleanup(@() cellfun(@(bd) close_system(bd, 0), edited)); %#ok<NASGU> scope guard

    tic;
    c = gs3dx_contact_check(info, rest=opts.rest, stop_time=opts.stop_time);
    elapsed_s = toc;
    assert(c.newton.pass, 'NewtonFailed: residual %.3g > bound %.3g', c.newton.max, c.newton.bound);
    assert(isfield(c.feet.L, 'R') && isfield(c.feet.R, 'R'), ...
        'NoAnkleRotation: AnkleLogs carries no Rotation_Transform');

    n_t = numel(c.t);
    labels = ["LHeel", "LToeIn", "LToeOut", "RHeel", "RToeIn", "RToeOut"];
    feet_alt = struct( ...
        'L', struct('p', c.feet.L.p, 'R', pagetranspose(c.feet.L.R)), ...
        'R', struct('p', c.feet.R.p, 'R', pagetranspose(c.feet.R.R)));
    [~, points] = gs3dx_sole_contact_points(geo, c.feet);
    [~, points_alt] = gs3dx_sole_contact_points(geo, feet_alt);   % R transposed: orientation diagnostic only
    forces = c.contacts;   % 3 x 6 x N, World, ground on foot, same order
    assert(isequal(size(forces), size(points)), 'ContactCountMismatch');

    ground_z = geo.GroundOffset(3);
    loaded = squeeze(vecnorm(forces, 2, 1)) > 1.0;   % 6 x N
    gap = squeeze(points(3, :, :)) - ground_z;
    gap_alt = squeeze(points_alt(3, :, :)) - ground_z;

    idx = unique(round(linspace(1, n_t, min(opts.n_rows, n_t))));
    header = "time";
    data = c.t(idx).';
    for k = 1:numel(labels)
        for q = ["Force", "Point"]
            header = [header, "GroundContactLogs_" + labels(k) + q + "_" + string(1:3)]; %#ok<AGROW>
        end
        data = [data, squeeze(forces(:, k, idx)).', squeeze(points(:, k, idx)).']; %#ok<AGROW>
    end
    header = [header, "COMLogs_GlobalPosition_" + string(1:3), "GroundContactLogs_GroundHeight"];
    data = [data, c.com(:, idx).', repmat(ground_z, numel(idx), 1)];
    csv_path = fullfile(out_dir, 'simscape_gs3dx_grf_fixture.csv');
    gs3dx_write_csv(csv_path, header, data);

    g = 9.80665;
    receipt = struct();
    receipt.issue = '#11709';
    receipt.matlab_release = rel;
    receipt.matlab_version = version;
    receipt.model = mdl;
    receipt.model_sha256 = gs3dx_sha256(fullfile(info.models_dir, [mdl '.slx']));
    receipt.rest = opts.rest;
    receipt.lock_joints = opts.lock_joints;
    receipt.locked_joint_count = numel(locked);
    receipt.stop_time_s = opts.stop_time;
    receipt.sim_wall_clock_s = elapsed_s;
    receipt.mass_kg = c.mass;
    receipt.weight_n = c.mass * g;
    receipt.ground_height_m = ground_z;
    receipt.newton_residual_max_ns = c.newton.max;
    receipt.newton_bound_ns = c.newton.bound;
    receipt.support_bw_min = c.support(1);
    receipt.support_bw_max = c.support(2);
    receipt.slip_m = struct('L', c.feet.L.slip, 'R', c.feet.R.slip);
    receipt.lift_m = struct('L', c.feet.L.lift, 'R', c.feet.R.lift);
    receipt.contact_gap_m = struct('min', min(gap(loaded)), 'max', max(gap(loaded)));
    receipt.contact_gap_if_R_transposed_m = struct( ...
        'min', min(gap_alt(loaded)), 'max', max(gap_alt(loaded)));
    receipt.samples = n_t;
    receipt.fixture_rows = numel(idx);
    receipt.fixture_csv = 'simscape_gs3dx_grf_fixture.csv';
    receipt.fixture_sha256 = gs3dx_sha256(csv_path);
    receipt.convention = 'ground on foot, world Z-up, SI; point = sphere lowest point';
    fid = fopen(fullfile(out_dir, 'simscape_gs3dx_grf_receipt.json'), 'w');
    fprintf(fid, '%s\n', jsonencode(receipt, 'PrettyPrint', true));
    fclose(fid);
    fprintf('%s\n', jsonencode(receipt, 'PrettyPrint', true));
end
