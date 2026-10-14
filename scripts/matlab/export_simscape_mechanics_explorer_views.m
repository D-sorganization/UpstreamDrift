function receipt = export_simscape_mechanics_explorer_views(model, out_dir, options)
%EXPORT_SIMSCAPE_MECHANICS_EXPLORER_VIEWS  Export 4 NV-1 views from Mechanics Explorer (NV-8, #11681).
%
%   RECEIPT = EXPORT_SIMSCAPE_MECHANICS_EXPLORER_VIEWS(MODEL, OUT_DIR)
%   RECEIPT = EXPORT_SIMSCAPE_MECHANICS_EXPLORER_VIEWS(MODEL, OUT_DIR, Name, Value)
%
%   Exports the four standard NV-1 golf camera views:
%     1. face_on
%     2. down_the_line
%     3. overhead
%     4. perspective
%   from Simscape Mechanics Explorer using smwritevideo.
%
%   Preconditions:
%     - MATLAB R2025b host required (asserted via version('-release') == '2025b').
%       Fails closed with a clear error on non-R2025b releases (e.g. R2026a).
%     - MODEL is a loaded Simscape/Simulink model name, file path, or simlog.
%     - OUT_DIR is a writable directory for exported MP4 files and JSON receipt.
%
%   Outputs:
%     - <prefix>_face_on.mp4
%     - <prefix>_down_the_line.mp4
%     - <prefix>_overhead.mp4
%     - <prefix>_perspective.mp4
%     - <prefix>_receipt.json (SHA-256 digests, camera poses, MATLAB metadata)
%
%   NV-1 Presets (World is Z-up, golfer faces -X, target line is -Y):
%     - face_on: azimuth 0 deg, elevation -6 deg, distance 3.4 m
%     - down_the_line: azimuth -90 deg, elevation -8 deg, distance 3.75 m
%     - overhead: azimuth 0 deg, elevation -89 deg, distance 2.28 m
%     - perspective: azimuth 135 deg, elevation -14 deg, distance 3.95 m
%     - vertical field of view: 0.7 rad (~40.1070 deg, VIEWER_FOV_Y_RAD)

    arguments
        model (1,1) string = ""
        out_dir (1,1) string = pwd
        options.Prefix (1,1) string = ""
        options.LookAt (1,3) double = [0.0, 0.0, 0.85]
        options.FovYRad (1,1) double = 0.7
        options.FrameRate (1,1) double = 60
        options.DryRun (1,1) logical = false
    end

    % 1. Release gate: strictly enforce MATLAB R2025b
    rel = version('-release');
    assert(strcmp(rel, '2025b'), ...
        'R2025bRequired: detected release %s; Simscape Mechanics Explorer video export requires MATLAB R2025b', rel);

    % Allow returning camera presets when called with no model
    if strlength(model) == 0
        receipt = get_simscape_mechanics_explorer_presets(options.LookAt, options.FovYRad);
        return;
    end

    if ~isfolder(out_dir)
        mkdir(out_dir);
    end

    % Load model if an SLX path was supplied
    mdl = model;
    if isfile(model)
        [~, model_name, ~] = fileparts(model);
        if ~bdIsLoaded(model_name)
            load_system(model);
        end
        mdl = string(model_name);
    end

    prefix = options.Prefix;
    if strlength(prefix) == 0
        prefix = mdl;
    end

    % 2. Retrieve the four NV-1 camera view presets
    presets = get_simscape_mechanics_explorer_presets(options.LookAt, options.FovYRad);
    view_names = ["face_on", "down_the_line", "overhead", "perspective"];

    view_records = struct();
    out_files = struct();

    % 3. Export each view with smwritevideo
    for k = 1:numel(view_names)
        vname = char(view_names(k));
        preset = presets.(vname);

        mp4_filename = sprintf('%s_%s.mp4', prefix, vname);
        mp4_path = fullfile(out_dir, mp4_filename);

        pos = preset.position;
        target = preset.target;
        up = preset.up_vector;
        fov_deg = preset.fov_y_deg;

        if ~options.DryRun
            smwritevideo(char(mdl), mp4_path, ...
                'CameraPosition', pos, ...
                'CameraTarget', target, ...
                'CameraUpVector', up, ...
                'CameraViewAngle', fov_deg);
        end

        sha = "";
        if isfile(mp4_path)
            sha = local_file_sha256(mp4_path);
        end

        view_records.(vname) = struct(...
            'preset_name', string(vname), ...
            'mp4_file', string(mp4_filename), ...
            'mp4_path', string(mp4_path), ...
            'sha256', string(sha), ...
            'azimuth_deg', preset.azimuth_deg, ...
            'elevation_deg', preset.elevation_deg, ...
            'distance_m', preset.distance_m, ...
            'fov_y_rad', preset.fov_y_rad, ...
            'fov_y_deg', preset.fov_y_deg, ...
            'camera_position', pos, ...
            'camera_target', target, ...
            'camera_up_vector', up);
        out_files.(vname) = mp4_path;
    end

    % 4. Build and write JSON receipt
    receipt = struct();
    receipt.schema = "simscape-mechanics-explorer-export/v1";
    receipt.issue = 11681;
    receipt.epic_issue = 11673;
    receipt.matlab_release = string(rel);
    receipt.matlab_version = string(version);
    receipt.model = string(model);
    receipt.prefix = string(prefix);
    receipt.out_dir = string(out_dir);
    receipt.views = view_records;

    receipt_filename = sprintf('%s_receipt.json', prefix);
    receipt_path = fullfile(out_dir, receipt_filename);
    fid = fopen(receipt_path, 'w');
    assert(fid > 0, 'OpenFailed: cannot open %s for writing', receipt_path);
    cleanup_fid = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fwrite(fid, jsonencode(receipt, 'PrettyPrint', true), 'char');

    fprintf('Exported %d Simscape Mechanics Explorer views for %s (prefix: %s)\n', ...
        numel(view_names), mdl, prefix);
    fprintf('Receipt written to %s\n', receipt_path);
end

function presets = get_simscape_mechanics_explorer_presets(lookat_m, fov_y_rad)
%GET_SIMSCAPE_MECHANICS_EXPLORER_PRESETS Build the four NV-1 view presets.
    arguments
        lookat_m (1,3) double = [0.0, 0.0, 0.85]
        fov_y_rad (1,1) double = 0.7
    end

    fov_deg = fov_y_rad * 180.0 / pi;

    defs = struct();
    defs.face_on = struct('az', 0.0, 'el', -6.0, 'dist', 3.4);
    defs.down_the_line = struct('az', -90.0, 'el', -8.0, 'dist', 3.75);
    defs.overhead = struct('az', 0.0, 'el', -89.0, 'dist', 2.28);
    defs.perspective = struct('az', 135.0, 'el', -14.0, 'dist', 3.95);

    names = ["face_on", "down_the_line", "overhead", "perspective"];
    presets = struct();

    for k = 1:numel(names)
        nm = char(names(k));
        d = defs.(nm);

        a = d.az * pi / 180.0;
        e = d.el * pi / 180.0;
        view_dir = [cos(e) * cos(a), cos(e) * sin(a), sin(e)];

        pos = lookat_m - d.dist * view_dir;

        world_up = [0.0, 0.0, 1.0];
        right = cross(view_dir, world_up);
        right = right / norm(right);
        up = cross(right, view_dir);
        up = up / norm(up);

        presets.(nm) = struct(...
            'name', string(nm), ...
            'azimuth_deg', d.az, ...
            'elevation_deg', d.el, ...
            'distance_m', d.dist, ...
            'fov_y_rad', fov_y_rad, ...
            'fov_y_deg', fov_deg, ...
            'target', lookat_m, ...
            'position', pos, ...
            'up_vector', up, ...
            'view_direction', view_dir);
    end
end

function sha = local_file_sha256(file_path)
%LOCAL_FILE_SHA256 Compute SHA-256 digest of a file.
    fid = fopen(file_path, 'r');
    assert(fid > 0, 'OpenFailed: cannot open %s for reading', file_path);
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    bytes = fread(fid, Inf, '*uint8');
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(bytes);
    sha = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2).', 1, []));
end
