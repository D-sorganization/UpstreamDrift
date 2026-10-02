function report = gs3dx_match_export(capture_id, opts)
%GS3DX_MATCH_EXPORT  Export Simscape inverse-kinematics visualization videos and stills (#10979, #11161).
%   REPORT = GS3DX_MATCH_EXPORT(CAPTURE_ID, OPTS) exports unqualified IK
%   visualization videos, stills, and structured provenance JSON using GS3DX_Human.
%   Parameters: CAPTURE_ID ("capture-A", "capture-O", or explicit C3D path).
%   OPTS: mode="ik", model="", views=["face-on","down-the-line"], output_dir (REQ),
%         export_video=true, export_stills=true, stills=[], stride=0,
%         overlay_markers=true, frames=[], pose_source=[], registry_repo="".

    arguments
        capture_id (1,1) string
        opts.mode (1,1) string = "ik"
        opts.model (1,1) string = ""
        opts.views (1,:) string = ["face-on", "down-the-line"]
        opts.output_dir (1,1) string = ""
        opts.export_video (1,1) logical = true
        opts.export_stills (1,1) logical = true
        opts.stills (1,:) double = []
        opts.stride (1,1) double = 0
        opts.overlay_markers (1,1) logical = true
        opts.frames (1,:) double = []
        opts.pose_source = []
        opts.registry_repo (1,1) string = ""
        opts.checkpoint_dir (1,1) string = string(fullfile(tempdir, "gs3dx_export_cache"))
    end

    % 1. Precondition validation (fail-closed before any filesystem writes)
    local_validate_options(capture_id, opts);

    % 2. Resolve capture file and manifest metadata
    [c3d_path, cap_alias, cap_sha256] = local_resolve_capture(capture_id, opts.registry_repo);
    assert(isfile(c3d_path), 'gs3dx:match_export:CaptureNotFound', ...
        'Resolved capture file not found: %s', c3d_path);

    % 3. Read capture markers and joint centres
    cap = gs3dx_capture_markers(c3d_path);
    jc = gs3dx_capture_joint_centres(cap);

    % 4. Validate user frames or compute uniform stride frames
    [frames_to_render, stride, fps] = gs3dx_video_sampling(cap.rate_hz, cap.n_frames, ...
        opts.stride, opts.frames);

    % 5. Model selection (GS3DX_Human ONLY) and segment length fitting
    [mdl_name, personal_lengths] = local_setup_model(opts.model, jc);

    model_file = which([mdl_name '.slx']);
    assert(~isempty(model_file), 'gs3dx:match_export:MissingModel', ...
        'Model %s not found on path. Build GS3DX_Human before exporting.', mdl_name);
    model_sha256 = local_sha256(model_file);

    % Inventory all dependencies consistently for identity hashing and provenance
    deps = local_dependency_inventory();
    local_verify_required_dependencies(deps.solve);
    local_verify_required_dependencies(deps.render);
    deps_hashes = struct( ...
        'solve', local_hash_inventory(deps.solve), ...
        'render', local_hash_inventory(deps.render));

    % Collect live runtime components once and generate fingerprint via pure helper
    v_mat = version;
    r_mat = version('-release');
    all_ver = ver;
    py_ver = local_get_python_version();
    np_ver = local_get_python_module_version('numpy');
    ez_ver = local_get_python_module_version('ezc3d');
    runtime = gs3dx_runtime_fingerprint(v_mat, r_mat, all_ver, py_ver, np_ver, ez_ver);

    identity = struct( ...
        'capture_sha256', char(cap_sha256), ...
        'model_name', char(mdl_name), ...
        'model_sha256', model_sha256, ...
        'solve_hashes', deps_hashes.solve, ...
        'geometry', personal_lengths.vars, ...
        'frames', frames_to_render, ...
        'runtime', runtime);

    checkpoint_file = "";
    if strlength(opts.checkpoint_dir) > 0
        if ~isfolder(opts.checkpoint_dir)
            mkdir(opts.checkpoint_dir);
        end
        checkpoint_file = fullfile(opts.checkpoint_dir, cap_alias + "_" + mdl_name + "_ik.mat");
    end

    % 6. Obtain and validate poses (closes model immediately after solve)
    [poses, t_vec] = local_obtain_poses(mdl_name, jc, cap.rate_hz, ...
        frames_to_render, opts.pose_source, personal_lengths, checkpoint_file, identity, deps.solve);

    % 7. Map stills from capture frames to pose column indices
    [stills_cols, still_names_map] = local_resolve_stills(opts.stills, ...
        poses.frames, cap, jc, opts.views, cap_alias, opts.export_stills);

    % 8. Format scene title for overlay: honest IK qualification labeling
    coverage_str = sprintf('Frames %d-%d (%.2f-%.2fs, %.0fHz, stride %d)', ...
        poses.frames(1), poses.frames(end), t_vec(1), t_vec(end), cap.rate_hz, stride);
    scene_title = sprintf('%s [%s] | IK / DYNAMICS UNQUALIFIED | %s', cap_alias, mdl_name, coverage_str);

    % 9. Prepare marker overlays (S'*(p-origin), waist origin, subset-aligned)
    markers_overlay = local_build_overlay_markers(cap, jc, poses.frames, opts.overlay_markers);

    % Create the public output directory only after input and pose validation.
    if ~isfolder(opts.output_dir)
        mkdir(opts.output_dir);
    end
    % 10. Execute rendering per view through gs3dx_render
    [rendered_videos, rendered_stills, setup_info] = local_render_views(mdl_name, poses, ...
        opts.views, cap_alias, opts.output_dir, opts.export_video, opts.export_stills, ...
        stills_cols, still_names_map, markers_overlay, t_vec, fps, scene_title, personal_lengths);

    % 11. Build and save structured provenance JSON
    report = local_build_report(cap_alias, cap_sha256, mdl_name, poses, cap, ...
        stride, fps, t_vec, personal_lengths, rendered_videos, rendered_stills, ...
        opts.output_dir, model_sha256, deps_hashes, setup_info, jc.foot_calibration, ...
        deps, runtime);
end

% -------------------------------------------------------------------------
% Helper: Validate options fail-closed before writes (GS3DX_Human ONLY)
% -------------------------------------------------------------------------
function local_validate_options(capture_id, opts)
    assert(strlength(capture_id) > 0, 'gs3dx:match_export:EmptyCaptureId', ...
        'Capture ID cannot be empty');
    assert(strlength(opts.output_dir) > 0, 'gs3dx:match_export:MissingOutputDir', ...
        'Output directory must be explicitly specified (no implicit user path)');
    if opts.mode ~= "ik"
        error('gs3dx:match_export:UnsupportedMode', ...
            'Unsupported mode: "%s". Default only honest IK visualization supported.', opts.mode);
    end
    assert(isempty(opts.pose_source), 'gs3dx:match_export:UnverifiedPoses', ...
        'Precomputed poses are unsupported without a capture/model/geometry identity contract');
    assert(opts.model == "" || opts.model == "GS3DX_Human", 'gs3dx:match_export:UnsupportedModel', ...
        'gs3dx_match_export supports GS3DX_Human only. Legacy baseline GS3DX_Fit is rejected for public export.');
    assert(~isempty(opts.views) && all(ismember(opts.views, ["face-on", "down-the-line", "top"])), ...
        'gs3dx:match_export:InvalidViews', 'Unsupported view');
    assert(all(isfinite(opts.stills) & opts.stills >= 1 & opts.stills == fix(opts.stills)), ...
        'gs3dx:match_export:InvalidStills', 'Still frames must be positive integer capture indices');
    assert(isfinite(opts.stride) && opts.stride >= 0 && opts.stride == fix(opts.stride), ...
        'gs3dx:match_export:InvalidStride', 'Stride must be a nonnegative integer');
    if ~isempty(opts.frames)
        assert(all(isfinite(opts.frames) & opts.frames >= 1 & opts.frames == round(opts.frames)), ...
            'gs3dx:match_export:InvalidFrames', 'Frames must be positive integers');
        assert(issorted(opts.frames, 'strictascend'), ...
            'gs3dx:match_export:InvalidFrames', 'Frames must be strictly monotonically increasing');
    end
end

% -------------------------------------------------------------------------
% Helper: Resolve capture and metadata without hardcoded user paths
% -------------------------------------------------------------------------
function [c3d_path, cap_alias, cap_sha256] = local_resolve_capture(capture_id, registry_repo)
    if isfile(capture_id) || endsWith(lower(capture_id), '.c3d')
        c3d_path = char(capture_id);
        cap_alias = "custom";
        assert(isfile(c3d_path), 'gs3dx:match_export:CaptureNotFound', ...
            'Explicit capture file not found: %s', c3d_path);
        cap_sha256 = local_sha256(c3d_path);
        return;
    end

    cap_alias = string(capture_id);

    % Try finding resolve_capture in path or in registry repo
    if exist('resolve_capture', 'file') ~= 2 && strlength(registry_repo) > 0
        shared_dir = fullfile(registry_repo, 'src', 'engines', ...
            'Simscape_Multibody_Models', '3D_Golf_Model', 'matlab', 'motion_matching', 'shared');
        if isfolder(shared_dir)
            addpath(shared_dir);
        end
    end

    if exist('resolve_capture', 'file') == 2
        if strlength(registry_repo) > 0
            c3d_path = resolve_capture(cap_alias, repo_root=registry_repo);
        else
            c3d_path = resolve_capture(cap_alias);
        end
    else
        % Fallback for canonical public tour capture in repository
        if strcmp(cap_alias, "capture-A")
            repo_root = local_find_repo_root();
            c3d_path = fullfile(repo_root, 'data', 'C3D_TA_Driver.c3d');
            if ~isfile(c3d_path) && strlength(registry_repo) > 0
                c3d_path = fullfile(registry_repo, 'data', 'C3D_TA_Driver.c3d');
            end
        else
            error('gs3dx:match_export:CaptureNotFound', ...
                'Cannot resolve capture "%s" without resolve_capture provider', cap_alias);
        end
    end

    assert(isfile(c3d_path), 'gs3dx:match_export:CaptureNotFound', ...
        'Capture file not found: %s', c3d_path);
    cap_sha256 = local_sha256(c3d_path);
end

% -------------------------------------------------------------------------
% Helper: Find current repository root
% -------------------------------------------------------------------------
function root = local_find_repo_root()
    curr = fileparts(fileparts(mfilename('fullpath')));
    root = '';
    for i = 1:10
        if isfile(fullfile(curr, 'data', 'C3D_TA_Driver.c3d')) || isfolder(fullfile(curr, '.git'))
            root = curr;
            return;
        end
        parent = fileparts(curr);
        if strcmp(parent, curr)
            break;
        end
        curr = parent;
    end
    if isempty(root)
        root = pwd;
    end
end

% -------------------------------------------------------------------------
% Helper: Model selection and segment length fitting (GS3DX_Human ONLY)
% -------------------------------------------------------------------------
function [mdl, personal_lengths] = local_setup_model(user_model, jc)
    names = gs3dx_names();
    if strlength(user_model) > 0
        mdl = char(user_model);
    else
        mdl = char(names.variants.human);
    end
    assert(strcmp(mdl, "GS3DX_Human"), 'gs3dx:match_export:UnsupportedModel', ...
        'gs3dx_match_export requires GS3DX_Human. Legacy baseline GS3DX_Fit is rejected for public export.');

    % Reuse gs3dx_fit_lengths for BOTH tour and owner captures
    personal_lengths = gs3dx_fit_lengths(jc);
    assert(~isempty(personal_lengths) && isfield(personal_lengths, 'vars'), ...
        'gs3dx:match_export', 'Failed to calculate segment lengths from markers');
end

% -------------------------------------------------------------------------
% Helper: DRY model setup & visual adaptation before solve and rendering
% -------------------------------------------------------------------------
function setup_info = local_apply_model_setup(mdl, personal_lengths)
    mdl_name = char(mdl);
    assert(strcmp(mdl_name, "GS3DX_Human"), 'gs3dx:match_export:UnsupportedModel', ...
        'gs3dx_match_export requires GS3DX_Human. Legacy baseline GS3DX_Fit is rejected for public export.');

    if bdIsLoaded(mdl_name)
        close_system(mdl_name, 0);
    end
    load_system(mdl_name);

    report = gs3dx_fit_human_visuals(mdl_name, personal_lengths);
    setup_info = struct();
    setup_info.model = "GS3DX_Human";
    setup_info.style_type = "ellipsoid_human";
    setup_info.visual_geometry = report.visual_geometry;
    setup_info.adapted_blocks = report.adapted_blocks;
    setup_info.scales = report.scales;
    setup_info.notes = report.notes;
end

% -------------------------------------------------------------------------
% Helper: Obtain and validate poses (closes model immediately after solve)
% -------------------------------------------------------------------------
function [poses, t_vec] = local_obtain_poses(mdl, jc, rate_hz, frames_to_render, ...
    user_poses, personal_lengths, checkpoint_file, identity, solve_deps)

    assert(isempty(user_poses), 'gs3dx:match_export:UnverifiedPoses', 'Fresh IK is required');
    reuse = false;
    if strlength(checkpoint_file) > 0 && isfile(checkpoint_file)
        cached = load(checkpoint_file, 'poses', 'identity');
        reuse = isfield(cached, 'poses') && isfield(cached, 'identity') && ...
            local_check_cache(cached.identity, identity);
    end
    if reuse
        poses = cached.poses;
        fprintf('IK_CHECKPOINT_REUSED\n');
    else
        local_apply_model_setup(mdl, personal_lengths);
        cleanup = onCleanup(@() local_close_model(mdl));
        f0 = jc.impact_frame;
        cal = 1:15:max(1, f0 - 90);
        % Opt into shared IK foot refinement (foot_orientation_weight=0.1)
        poses = gs3dx_whole_body_ik(jc, frames=frames_to_render, calibration_frames=cal, ...
            model=mdl, backward=false, posture_weight=0.04, smooth_weight=0.025, ...
            gap_weight=0.1, rom_weight=0.025, foot_orientation_weight=0.1, verbose=true);
        clear cleanup;

        % Verify all recorded solve dependency hashes remain equal after fresh solve
        fresh_solve_hashes = local_hash_inventory(solve_deps);
        assert(isequal(fresh_solve_hashes, identity.solve_hashes), ...
            'gs3dx:match_export:SourceChanged', ...
            'Solve dependency source files were mutated during execution; identity is unverified');

        if strlength(checkpoint_file) > 0
            save(checkpoint_file, 'poses', 'identity');
            fprintf('IK_CHECKPOINT_SAVED\n');
        end
    end

    % Strict pose validation contract
    assert(isfield(poses, 'joint_ids') && ~isempty(poses.joint_ids), ...
        'gs3dx:match_export:InvalidPoses', 'Poses missing joint_ids');
    assert(isfield(poses, 'joint') && ~isempty(poses.joint), ...
        'gs3dx:match_export:InvalidPoses', 'Poses missing joint matrix');
    assert(isfield(poses, 'frames') && ~isempty(poses.frames), ...
        'gs3dx:match_export:InvalidPoses', 'Poses missing frames vector');
    assert(isfield(poses, 't') && ~isempty(poses.t), ...
        'gs3dx:match_export:InvalidPoses', 'Poses missing t vector');

    assert(size(poses.joint, 1) == numel(poses.joint_ids), ...
        'gs3dx:match_export:InvalidPoses', 'Pose joint rows (%d) != joint_ids (%d)', ...
        size(poses.joint, 1), numel(poses.joint_ids));
    assert(size(poses.joint, 2) == numel(poses.frames), ...
        'gs3dx:match_export:InvalidPoses', 'Pose joint columns (%d) != frames (%d)', ...
        size(poses.joint, 2), numel(poses.frames));
    assert(numel(poses.frames) == numel(poses.t), ...
        'gs3dx:match_export:InvalidPoses', 'Pose frames (%d) != t (%d)', ...
        numel(poses.frames), numel(poses.t));

    assert(all(isfinite(poses.joint), 'all') && all(poses.status == 1), ...
        'gs3dx:match_export:InvalidPoses', 'Every pose must be finite and close the kinematic loop');
    assert(isequal(poses.frames, frames_to_render) && strcmp(poses.model, mdl), ...
        'gs3dx:match_export:InvalidPoses', 'Pose source identity differs from requested fit');
    t_vec = (poses.frames - 1) / rate_hz;
end

% -------------------------------------------------------------------------
% Helper: Map requested still capture frames to pose columns
% -------------------------------------------------------------------------
function [stills_cols, names_map] = local_resolve_stills(user_stills, pose_frames, cap, jc, views, cap_alias, export_stills)
    stills_cols = [];
    names_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
    if ~export_stills
        return;
    end

    if ~isempty(user_stills)
        target_cap_frames = user_stills;
    else
        f_top = local_detect_top_frame(jc, cap.impact_frame);
        target_cap_frames = unique([1, f_top, cap.impact_frame, pose_frames(end)]);
    end

    % Stills index POSE columns (1 to numel(pose_frames))
    cols = zeros(1, numel(target_cap_frames));
    for i = 1:numel(target_cap_frames)
        [~, c_idx] = min(abs(pose_frames - target_cap_frames(i)));
        cols(i) = c_idx;
    end
    stills_cols = unique(cols);

    for v_idx = 1:numel(views)
        view_slug = strrep(lower(char(views(v_idx))), ' ', '_');
        s_names = string.empty;
        for s_i = 1:numel(stills_cols)
            col = stills_cols(s_i);
            cap_f = pose_frames(col);
            s_names(end+1) = sprintf('%s_ik_%s_f%04d.png', cap_alias, view_slug, cap_f); %#ok<AGROW>
        end
        names_map(char(views(v_idx))) = s_names;
    end
end

% -------------------------------------------------------------------------
% Helper: Detect top of backswing frame from joint centres
% -------------------------------------------------------------------------
function f_top = local_detect_top_frame(jc, f_impact)
    if isfield(jc, 'pelvis_R')
        yaw = squeeze(atan2(jc.pelvis_R(2, 1, :), jc.pelvis_R(1, 1, :)));
        search_range = 1:min(f_impact, numel(yaw));
        [~, rel_top] = min(yaw(search_range));
        f_top = search_range(rel_top);
    else
        f_top = max(1, round(0.7 * f_impact));
    end
end

% -------------------------------------------------------------------------
% Helper: Build marker overlay (transformed to waist origin and aligned)
% -------------------------------------------------------------------------
function pts_subset = local_build_overlay_markers(cap, jc, pose_frames, overlay_markers)
    if ~overlay_markers
        pts_subset = [];
        return;
    end

    pts_subset = gs3dx_marker_overlay(jc, pose_frames);
end

% -------------------------------------------------------------------------
% Helper: Render views headlessly through gs3dx_render
% -------------------------------------------------------------------------
function [rendered_videos, rendered_stills, setup_info] = local_render_views(mdl, poses, ...
    views, cap_alias, output_dir, export_video, export_stills, stills_cols, ...
    still_names_map, markers_overlay, t_vec, fps, scene_title, personal_lengths)

    rendered_videos = containers.Map('KeyType', 'char', 'ValueType', 'char');
    rendered_stills = containers.Map('KeyType', 'char', 'ValueType', 'any');
    setup_info = struct();

    for v_idx = 1:numel(views)
        view_name = char(views(v_idx));
        view_slug = strrep(lower(view_name), ' ', '_');

        v_out_file = '';
        if export_video
            v_out_file = char(fullfile(output_dir, sprintf('%s_ik_%s.mp4', cap_alias, view_slug)));
        end

        s_names = string.empty;
        if export_stills && isKey(still_names_map, view_name)
            s_names = still_names_map(view_name);
        end

        mdl_char = char(mdl);
        out_dir_char = char(output_dir);
        title_char = char(scene_title);

        % DRY model setup and visual adaptation before every render (renderer closes model)
        setup_info = local_apply_model_setup(mdl_char, personal_lengths);

        % Call gs3dx_render (headless only; closes model internally or via caller)
        gs3dx_render(mdl_char, poses, ...
            view=view_name, ...
            video=v_out_file, ...
            fps=fps, ...
            stills=stills_cols, ...
            still_files=s_names, ...
            markers=markers_overlay, ...
            time=t_vec, ...
            output_dir=out_dir_char, ...
            title=title_char, ...
            visible=false);

        % Close system cleanly without saving
        if bdIsLoaded(mdl_char)
            close_system(mdl_char, 0);
        end

        if export_video && isfile(v_out_file)
            rendered_videos(view_name) = v_out_file;
        end

        if export_stills
            actual_stills = {};
            for s_i = 1:numel(s_names)
                s_path = fullfile(output_dir, char(s_names(s_i)));
                if isfile(s_path)
                    actual_stills{end+1} = s_path; %#ok<AGROW>
                end
            end
            rendered_stills(view_name) = actual_stills;
        end
    end

    if isempty(fieldnames(setup_info))
        setup_info = local_apply_model_setup(mdl, personal_lengths);
        if bdIsLoaded(mdl)
            close_system(mdl, 0);
        end
    end
end

% -------------------------------------------------------------------------
% Helper: Build report and write provenance JSON
% -------------------------------------------------------------------------
function report = local_build_report(cap_alias, cap_sha256, mdl, poses, cap, ...
    stride, fps, t_vec, personal_lengths, rendered_videos, rendered_stills, ...
    output_dir, model_sha256, deps_hashes, setup_info, foot_calibration, ...
    deps, runtime)

    provenance = struct();
    provenance.capture_alias = char(cap_alias);
    provenance.capture_sha256 = char(cap_sha256);
    provenance.source_commit = char(local_git_commit());
    provenance.model = char(mdl);
    provenance.model_sha256 = model_sha256;
    provenance.code_sha256 = struct( ...
        'solve_dependencies', deps_hashes.solve, ...
        'render_dependencies', deps_hashes.render);
    if isstruct(deps)
        provenance.recorded_dependencies = struct( ...
            'solve_dependencies', local_relativize_struct(deps.solve), ...
            'render_dependencies', local_relativize_struct(deps.render));
    end
    if isstruct(runtime)
        provenance.runtime = runtime;
    end
    provenance.limitations = struct( ...
        'external_stl_assets', 'External STL graphics mesh assets are not recorded in dependency inventory or hashed; code parameter hashes do not record or guarantee actual mesh geometry assets.', ...
        'transitive_dependencies', 'Direct runtime execution components (MATLAB, Simulink, Simscape, Simscape Multibody, Python, numpy, ezc3d) are recorded explicitly; unrecorded transitive packages and libraries are not implied frozen.');

    for side = ["L", "R"]
        if isfield(foot_calibration, side) && isfield(foot_calibration.(side), 'F')
            foot_calibration.(side) = rmfield(foot_calibration.(side), 'F');
        end
    end
    provenance.foot_calibration = foot_calibration;
    if isfield(poses, 'foot_orientation_error_deg')
        provenance.foot_orientation_error_deg = poses.foot_orientation_error_deg;
    else
        provenance.foot_orientation_error_deg = struct('L', 0, 'R', 0);
    end
    provenance.mode = "ik";
    provenance.mode_label = "IK / DYNAMICS UNQUALIFIED";
    provenance.qualification_verdict = "IK_VISUALIZATION_UNQUALIFIED";
    if isfield(poses, 'joint')
        provenance.diagnostics = gs3dx_ik_diagnostics(poses);
        provenance.diagnostics.units = struct('residual', 'm', 'step_speed', 'm/s', 'time', 's');
    else
        provenance.diagnostics = struct('status', 'not_evaluated');
    end
    provenance.events = struct('peak_speed_frame', cap.impact_frame, ...
        'contact_frame', [], 'contact_status', 'No explicit contact annotation supplied');
    provenance.joint_inventory = struct('native_position_variable_count', numel(poses.joint_ids), ...
        'independent_fit_coordinate_count', poses.independent_coordinate_count);

    provenance.regularization = struct( ...
        'posture_weight', 0.04, ...
        'smooth_weight', 0.025, ...
        'gap_weight', 0.1, ...
        'rom_weight', 0.025, ...
        'foot_orientation_weight', 0.1, ...
        'foot_orientation_unit', 'm per normalized chordal SO(3) error', ...
        'foot_orientation_disclaimer', 'Marker triad calibrated at frame 1 assuming flat soles; dynamic contact unqualified', ...
        'rms_basis', '14 anatomical marker position residuals (orientation term excluded from measured-position RMS)');

    provenance.red_gates = { ...
        "DynamicsUnsimulated: Pure kinematic geometry fit (torques/forces unmodeled; dynamic qualification unestablished)", ...
        "ModelChanged: Human includes articulated neck and forefoot topology and inherited segment inertia parameters. Capture-specific mass and dynamic equivalence remain unqualified" ...
    };

    provenance.style = struct( ...
        'model', char(mdl), ...
        'type', setup_info.style_type, ...
        'torso_proportions', "Fixed artistic visual proportions; head and shoe widths fixed artistic values without complete anthropometric calibration", ...
        'notes', setup_info.notes, ...
        'visual_adapter_sha256', deps_hashes.solve.visual_adapter, ...
        'vector_scale_sha256', deps_hashes.solve.vector_scale, ...
        'visual_geometry', setup_info.visual_geometry, ...
        'external_stl_assets', 'External STL mesh assets are not recorded or hashed; parameter hashes do not verify mesh files');

    provenance.coverage = struct( ...
        'start_frame', poses.frames(1), ...
        'end_frame', poses.frames(end), ...
        'total_capture_frames', cap.n_frames, ...
        'sample_rate_hz', cap.rate_hz, ...
        'stride', stride, ...
        'playback_fps', fps, ...
        'start_time_s', t_vec(1), ...
        'end_time_s', t_vec(end), ...
        'duration_s', t_vec(end) - t_vec(1));

    provenance.personal_geometry = personal_lengths.vars;

    if isfield(poses, 'rms')
        provenance.metrics = struct('mean_rms_mm', mean(poses.rms) * 1000, ...
            'max_rms_mm', max(poses.rms) * 1000);
    else
        provenance.metrics = struct('type', 'kinematic_ik_fit');
    end

    provenance.video_exports = struct();
    for v_k = keys(rendered_videos)
        vk = v_k{1};
        v_path = rendered_videos(vk);
        v_info = dir(v_path);
        provenance.video_exports.(matlab.lang.makeValidName(vk)) = struct( ...
            'file', v_path, 'bytes', v_info.bytes, 'fps', fps, 'codec_requested', 'MPEG-4 (decode verification required)');
    end

    provenance.still_exports = struct();
    for s_k = keys(rendered_stills)
        sk = s_k{1};
        provenance.still_exports.(matlab.lang.makeValidName(sk)) = rendered_stills(sk);
    end

    provenance.timestamp = datestr(now, 'yyyy-mm-ddTHH:MM:SS');

    prov_file = fullfile(output_dir, sprintf('provenance_%s_ik.json', cap_alias));
    fid = fopen(prov_file, 'w');
    assert(fid ~= -1, 'gs3dx:match_export', 'Could not open provenance file: %s', prov_file);
    cleanup_fid = onCleanup(@() fclose(fid));
    fwrite(fid, jsonencode(provenance, 'PrettyPrint', true));
    clear cleanup_fid;

    report = struct();
    report.capture_alias = cap_alias;
    report.mode = "ik";
    report.model = mdl;
    report.provenance_file = string(prov_file);
    report.provenance = provenance;
    report.videos = rendered_videos;
    report.stills = rendered_stills;
    report.verdict = "IK_VISUALIZATION_UNQUALIFIED";
    report.red_gates = provenance.red_gates;
    report.status = "success";
end

% -------------------------------------------------------------------------
% Helper: Consistent dependency inventory for identity hashing and provenance
% -------------------------------------------------------------------------
function dep = local_dependency_inventory()
    dep = struct();

    % 1. Core solve dependencies (invalidation triggers for IK checkpoint)
    dep.solve = struct();
    dep.solve.export = local_resolve_file([mfilename('fullpath') '.m']);
    dep.solve.runtime_fingerprint = local_resolve_which('gs3dx_runtime_fingerprint');
    dep.solve.capture_markers = local_resolve_which('gs3dx_capture_markers');
    dep.solve.capture_points = local_resolve_which('gs3dx_capture_points');
    dep.solve.resolve_capture = local_resolve_which('resolve_capture');
    dep.solve.ik = local_resolve_which('gs3dx_whole_body_ik');
    dep.solve.ik_joint_roles = local_resolve_which('gs3dx_ik_joint_roles');
    dep.solve.joint_keys = local_resolve_which('gs3dx_joint_keys');
    dep.solve.joint_rom = local_resolve_which('gs3dx_joint_rom');
    dep.solve.names = local_resolve_which('gs3dx_names');
    dep.solve.foot_orientation_residual = local_resolve_which('gs3dx_foot_orientation_residual');
    dep.solve.foot_marker_frame = local_resolve_which('gs3dx_foot_marker_frame');
    dep.solve.fit_lengths = local_resolve_which('gs3dx_fit_lengths');
    dep.solve.capture_joint_centres = local_resolve_which('gs3dx_capture_joint_centres');
    dep.solve.capture_address_transform = local_resolve_which('gs3dx_capture_address_transform');
    dep.solve.visual_adapter = local_resolve_which('gs3dx_fit_human_visuals');
    dep.solve.vector_scale = local_resolve_which('gs3dx_scale_vector');

    % 2. Render-only dependencies (recorded in provenance; do NOT invalidate IK checkpoint)
    dep.render = struct();
    dep.render.render = local_resolve_which('gs3dx_render');
    dep.render.scene_bounds = local_resolve_which('gs3dx_scene_bounds');
    dep.render.marker_overlay = local_resolve_which('gs3dx_marker_overlay');
    dep.render.sampling = local_resolve_which('gs3dx_video_sampling');
    dep.render.diagnostics = local_resolve_which('gs3dx_ik_diagnostics');
    dep.render.joint_keys = local_resolve_which('gs3dx_joint_keys');
end

function f = local_resolve_which(name)
    w = which(name);
    if isempty(w) && ~endsWith(name, '.m')
        w = which([name '.m']);
    end
    if isempty(w)
        tools_dir = fileparts(mfilename('fullpath'));
        cand = fullfile(tools_dir, [name '.m']);
        if isfile(cand)
            w = cand;
        end
    end
    f = local_resolve_file(w);
end

function f = local_resolve_file(p)
    if isempty(p)
        f = '';
    else
        f = char(p);
    end
end

function h = local_hash_inventory(files_struct)
    h = struct();
    for fld = fieldnames(files_struct).'
        fn = fld{1};
        p = files_struct.(fn);
        if strcmp(fn, 'resolve_capture')
            if isempty(p) || ~isfile(p)
                h.(fn) = 'absent';
            else
                h.(fn) = local_sha256(p);
            end
        else
            assert(~isempty(p) && isfile(p), ...
                'gs3dx:match_export:MissingRequiredDependency', ...
                'Required dependency "%s" has missing or unresolvable file path.', fn);
            h_val = local_sha256(p);
            assert(~isempty(h_val) && ~strcmp(h_val, 'sha256_unavailable'), ...
                'gs3dx:match_export:HashFailure', ...
                'Failed to compute valid SHA-256 for required dependency "%s".', fn);
            h.(fn) = h_val;
        end
    end
end

% -------------------------------------------------------------------------
% Helper: Validate required dependencies fail-closed (resolve_capture optional)
% -------------------------------------------------------------------------
function local_verify_required_dependencies(deps_struct)
    for fld = fieldnames(deps_struct).'
        fn = fld{1};
        if strcmp(fn, 'resolve_capture')
            continue; % Optional dependency: absent is valid direct-file or public tour fallback
        end
        p = deps_struct.(fn);
        if isempty(p) || ~isfile(p)
            error('gs3dx:match_export:MissingRequiredDependency', ...
                'Required dependency "%s" was not found on path.', fn);
        end
    end
end

function py_ver = local_get_python_version()
    try
        sys = py.importlib.import_module('sys');
        vi = sys.version_info;
        py_ver = sprintf('%d.%d.%d', int64(vi.major), int64(vi.minor), int64(vi.micro));
    catch err
        error('gs3dx:match_export:MissingRuntimeDependency', ...
            'Failed to query Python version from sys.version_info: %s', err.message);
    end
    if isempty(py_ver)
        error('gs3dx:match_export:MissingRuntimeDependency', ...
            'Python version could not be determined from sys.version_info.');
    end
end

function mod_ver = local_get_python_module_version(mod_name)
    try
        m = py.importlib.import_module(mod_name);
        mod_ver = char(py.getattr(m, '__version__'));
    catch err
        error('gs3dx:match_export:MissingRuntimeDependency', ...
            'Required Python package "%s" is not loaded or missing: %s', mod_name, err.message);
    end
    if isempty(mod_ver)
        error('gs3dx:match_export:MissingRuntimeDependency', ...
            'Version for Python package "%s" is empty or unknown.', mod_name);
    end
end

function reuse = local_check_cache(cached_identity, current_identity)
    reuse = isstruct(cached_identity) && isstruct(current_identity) && ...
        isfield(cached_identity, 'runtime') && isfield(current_identity, 'runtime') && ...
        isequal(cached_identity, current_identity);
end

function s_rel = local_relativize_struct(s_abs)
    s_rel = struct();
    for fld = fieldnames(s_abs).'
        fn = fld{1};
        p = s_abs.(fn);
        if isempty(p)
            s_rel.(fn) = '';
        elseif strcmp(p, 'absent')
            s_rel.(fn) = 'absent';
        else
            s_rel.(fn) = local_relative_path(p);
        end
    end
end

function rel = local_relative_path(abs_path)
    if isempty(abs_path)
        rel = '';
        return;
    end
    repo_root = local_find_repo_root();
    if ~isempty(repo_root) && startsWith(abs_path, repo_root)
        rel = extractAfter(abs_path, strlength(repo_root));
        if startsWith(rel, filesep) || startsWith(rel, '/') || startsWith(rel, '\')
            rel = extractAfter(rel, 1);
        end
        rel = strrep(rel, '\', '/');
    else
        [~, n, e] = fileparts(abs_path);
        rel = [n e];
    end
end

% -------------------------------------------------------------------------
% Helper: Compute SHA-256 hash using Java MessageDigest (fail-closed)
% -------------------------------------------------------------------------
function hex = local_sha256(filepath)
    assert(~isempty(filepath) && isfile(filepath), ...
        'gs3dx:match_export:MissingFile', 'File does not exist: %s', filepath);
    try
        md = java.security.MessageDigest.getInstance('SHA-256');
        fid = fopen(filepath, 'r');
        assert(fid ~= -1, 'Cannot open file for hashing');
        clean_fid = onCleanup(@() fclose(fid));
        bytes = fread(fid, Inf, '*uint8');
        md.update(bytes);
        hashBytes = typecast(md.digest(), 'uint8');
        hex = lower(reshape(dec2hex(hashBytes)', 1, []));
    catch err
        error('gs3dx:match_export:HashFailure', ...
            'Failed to compute SHA-256 for file "%s": %s', filepath, err.message);
    end
    assert(~isempty(hex), 'gs3dx:match_export:HashFailure', ...
        'SHA-256 computation produced empty string for file: %s', filepath);
end

% -------------------------------------------------------------------------
% Helper: Query current git commit
% -------------------------------------------------------------------------
function commit = local_git_commit()
    commit = 'unknown';
    try
        [st, out] = system('git rev-parse HEAD');
        if st == 0
            commit = strtrim(out);
        end
    catch
    end
end

function local_close_model(mdl)
    if bdIsLoaded(mdl)
        close_system(mdl, 0);
    end
end
