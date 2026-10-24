function receipt = export_gs3dx_rigid_stance_fixture(repo, out_dir, opts)
%EXPORT_GS3DX_RIGID_STANCE_FIXTURE  R2025b rigid-body stance run -> per-contact GRF CSV.
%
%   RECEIPT = EXPORT_GS3DX_RIGID_STANCE_FIXTURE(REPO, OUT_DIR, OPTS) runs
%   GS3DX_RIGID_STANCE -- the director-approved rigid-body proxy for the
%   GS3DX_FullBodyContact stance (#11709 option (b): one rigid body at the
%   golfer's total mass and centre of mass, on the same six sole-sphere
%   contacts and Spatial Contact Force law and parameters) -- and writes
%   OUT_DIR/simscape_gs3dx_rigid_stance_fixture.csv plus a JSON receipt,
%   in the SAME column schema as EXPORT_GS3DX_GRF_FIXTURE so
%   src.engines.simscape.force_channels.load_simscape_force_series and
%   src.engines.simscape.ground_contact.ground_reaction_series read it
%   unchanged:
%
%     time
%     GroundContactLogs_<C>Force_1..3, GroundContactLogs_<C>Point_1..3
%         for C in LHeel, LToeIn, LToeOut, RHeel, RToeIn, RToeOut
%     COMLogs_GlobalPosition_1..3   rigid-body centre of mass
%     GroundContactLogs_GroundHeight
%
%   Sphere centres and the body's position are both logged at the
%   contacts' own six world points; the per-sample "Point" columns are the
%   same fixed sphere centres (minus one radius along +Z) at every row,
%   since this is a rigid body: the sole points do not move relative to
%   the body, and the body itself barely moves once at rest.
%
%   OPTS fields: probe_time, stop_time, solver, rel_tol, inertia_radius,
%   output_dt (all forwarded to GS3DX_RIGID_STANCE), n_rows (31, decimated
%   CSV row count), settled_window_s (0.5, the trailing window over which
%   the receipt's slip/sink summary is measured).
%
%   Preconditions: MATLAB R2025b (asserted); REPO is an UpstreamDrift checkout.
%   Postconditions: CSV (LF line endings) and receipt exist in OUT_DIR.

    arguments
        repo (1,1) string
        out_dir (1,1) string
        opts (1,1) struct = struct()
    end
    defaults = struct('probe_time', 0.004, 'stop_time', 2.0, 'solver', 'ode23t', ...
        'rel_tol', 1e-6, 'inertia_radius', 0.15, 'output_dt', 1e-3, ...
        'n_rows', 31, 'settled_window_s', 0.5);
    for f = reshape(fieldnames(defaults), 1, [])
        if ~isfield(opts, f{1}); opts.(f{1}) = defaults.(f{1}); end
    end
    assert(opts.n_rows >= 2 && opts.settled_window_s > 0, ...
        'BadOpts: n_rows >= 2, settled_window_s > 0');

    rel = version('-release');
    assert(strcmp(rel, '2025b'), 'R2025bRequired: detected %s', rel);

    root = fullfile(repo, 'src', 'engines', 'Simscape_Multibody_Models', ...
        '3D_Golf_Model', 'matlab', 'exploratory_gs3dx');
    addpath(root);
    info = gs3dx_setup();
    if ~isfolder(out_dir); mkdir(out_dir); end

    rig_opts = rmfield(opts, {'n_rows', 'settled_window_s'});
    rig_args = namedargs2cell(rig_opts);   % name-value options, not a struct
    r = gs3dx_rigid_stance(info, rig_args{:});

    labels = ["LHeel", "LToeIn", "LToeOut", "RHeel", "RToeIn", "RToeOut"];
    up = [0; 0; 1];
    points = r.sphere_points_m - r.geo.FootContactRadius * up;   % 3x6, fixed in the rigid body
    n_t = numel(r.t);
    idx = unique(round(linspace(1, n_t, min(opts.n_rows, n_t))));

    header = "time";
    data = r.t(idx).';
    for k = 1:numel(labels)
        header = [header, "GroundContactLogs_" + labels(k) + "Force_" + string(1:3), ...
                  "GroundContactLogs_" + labels(k) + "Point_" + string(1:3)]; %#ok<AGROW>
        data = [data, squeeze(r.forces_n(:, k, idx)).', repmat(points(:, k).', numel(idx), 1)]; %#ok<AGROW>
    end
    header = [header, "COMLogs_GlobalPosition_" + string(1:3), "GroundContactLogs_GroundHeight"];
    data = [data, r.com_p(:, idx).', repmat(r.geo.GroundOffset(3), numel(idx), 1)];
    csv_path = fullfile(out_dir, 'simscape_gs3dx_rigid_stance_fixture.csv');
    gs3dx_write_csv(csv_path, header, data);

    settled = r.t >= (opts.stop_time - opts.settled_window_s);
    assert(any(settled), 'gs3dx:rigid_stance_fixture', 'Empty settled window');
    com_settled = r.com_p(:, settled);
    com0 = r.com_p(:, 1);
    horiz0 = com_settled(1:2, :) - com0(1:2);
    g = 9.80665;
    receipt = struct();
    receipt.issue = '#11709';
    receipt.check = 'rigid_body_stance';
    receipt.matlab_release = rel;
    receipt.matlab_version = version;
    receipt.source_model = char(gs3dx_names().variants.contact);
    receipt.source_model_sha256 = gs3dx_sha256(fullfile(info.models_dir, [receipt.source_model '.slx']));
    receipt.probe_time_s = opts.probe_time;
    receipt.stop_time_s = opts.stop_time;
    receipt.solver = opts.solver;
    receipt.rel_tol = opts.rel_tol;
    receipt.inertia_radius_m = opts.inertia_radius;
    receipt.output_dt_s = opts.output_dt;
    receipt.mass_kg = r.mass_kg;
    receipt.weight_n = r.mass_kg * g;
    receipt.ground_height_m = r.geo.GroundOffset(3);
    receipt.com_probe_m = r.com_m.';
    receipt.sphere_points_m = r.sphere_points_m.';
    receipt.rigid_rig_compiled_blocks = r.budget.compiled_total;
    receipt.sim_wall_clock_s = r.wall_s;
    receipt.settled_window_s = [opts.stop_time - opts.settled_window_s, opts.stop_time];
    receipt.settled_slip_m = max(vecnorm(horiz0, 2, 1));
    receipt.settled_sink_m = com0(3) - min(com_settled(3, :));
    receipt.samples = n_t;
    receipt.fixture_rows = numel(idx);
    receipt.fixture_csv = 'simscape_gs3dx_rigid_stance_fixture.csv';
    receipt.fixture_sha256 = gs3dx_sha256(csv_path);
    receipt.convention = ['ground on body, world Z-up, SI; point = sphere lowest point, ' ...
        'fixed in the rigid body; net Fz/weight, left/right split and CoP are left to the ' ...
        'Python-side GCV-1/GCV-2 reduction (src.engines.simscape.ground_contact), not ' ...
        'recomputed here, so the pass/fail criteria have exactly one implementation'];
    fid = fopen(fullfile(out_dir, 'simscape_gs3dx_rigid_stance_receipt.json'), 'w');
    fprintf(fid, '%s\n', jsonencode(receipt, 'PrettyPrint', true));
    fclose(fid);
    fprintf('%s\n', jsonencode(receipt, 'PrettyPrint', true));
end
