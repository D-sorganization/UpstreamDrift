function out = gs3dx_render(mdl, q, opts)
%GS3DX_RENDER  Headless 3D rendering of the GS3DX golfer (stills and swing video).
%
%   OUT = GS3DX_RENDER(MDL, Q) renders the Simscape Multibody golfer model MDL
%   at the joint poses Q in an invisible figure (works under 'matlab -batch')
%   and returns rendered solid geometry, poses, and written file paths.
%
%   Poses Q:
%     - KinematicsSolver target matrix or IK struct (such as produced by
%       GS3DX_WHOLE_BODY_IK: struct with .joint_ids and .joint, and optional
%       .frames and .t).  An IK solved on another variant (.model) is
%       matched to MDL's joints by block path (GS3DX_JOINT_KEYS), as are
%       rows named by an optional .joint_keys; joints it lacks are at 0.
%     - Numeric matrix of joint positions (joint_ids x frames).
%
%   Geometry:
%     Every solid block in MDL ('sm_lib/Body Elements/* Solid', excluding
%     GraphicType 'None') is drawn in its pose:
%     - Cylindrical Solid: CylinderRadius, CylinderLength (axis z, centred)
%     - Spherical Solid: SphereRadius
%     - Brick Solid: BrickDimensions
%     - Ellipsoidal Solid: EllipsoidRadii
%     - File Solid: an STL (ExtGeomFileName, found on the path;
%       ExtGeomFileUnits), shared vertices merged so it shades smoothly
%     Expressions are evaluated in the model workspace with slResolve and
%     converted to SI metres. Diffuse color and opacity are respected.
%
%   Options:
%     stills        (1,:) double frame indices to export as PNG stills
%     still_files   (1,:) string filenames for the stills
%     video         (1,:) char output video file (.mp4)
%     fps           (1,1) double video playback frame rate (default 30)
%     view          (1,:) char, string, or 1x2 double camera view
%                   ("face-on": camera on +X, the facing axis; "down-the-line":
%                   camera on -Y, behind the golfer looking at the target;
%                   "top"; or [azimuth, elevation] as for VIEW)
%     markers       (:,:,:) double capture markers or joint centres (Nx3xF or
%                   3xNxF, World frame, m) for overlay dots
%     time          (1,:) double timestamps (s) per frame for labels
%     output_dir    (1,:) char output directory for files (default pwd)
%     ground        (1,1) logical whether to draw a ground plane (default true)
%     title         (1,:) char title annotation on frames (default '')
%     visible       (1,1) logical figure visibility (default false)
%     focus         close-up: a solid's name suffix (string, e.g. "Driver
%                   Head") the view follows, or a fixed centre [x y z] (m);
%                   empty (default) frames the whole golfer
%     focus_width   (1,1) double half-width of the close-up (m, default 0.2)
%
%   Output OUT:
%     .files        string array of written image and video file paths
%     .solids       struct array of parsed solids with local and World geometry
%     .frames       frame indices rendered
%     .view         [azimuth elevation] of the camera (degrees)
%     .focus        close-up centre per frame (3 x frames, m); [] without
%                   FOCUS
%     .status       "success"
%
%   See also GS3DX_WHOLE_BODY_IK, GS3DX_TRACK_LEARN.

    arguments
        mdl (1,:) char
        q = []
        opts.stills (1,:) double {mustBeInteger, mustBeNonnegative} = []
        opts.still_files (1,:) string = string.empty
        opts.video (1,:) char = ''
        opts.fps (1,1) double {mustBePositive} = 30
        opts.view = "face-on"
        opts.markers double = []
        opts.time (1,:) double = []
        opts.output_dir (1,:) char = pwd
        opts.ground (1,1) logical = true
        opts.title (1,:) char = ''
        opts.visible (1,1) logical = false
        opts.focus = []
        opts.focus_width (1,1) double {mustBePositive} = 0.2
    end

    % Preconditions
    assert(~isempty(mdl), 'gs3dx:render', 'Model name cannot be empty');
    if ~bdIsLoaded(mdl)
        load_system(mdl);
    end
    cleanup = onCleanup(@() close_system(mdl, 0));

    % Parse solids and extract geometry parameters
    solids = local_discover_solids(mdl);
    assert(~isempty(solids), 'gs3dx:render', 'No graphical Solid blocks found in model %s', mdl);

    % Expose reference frame port on all solids in memory
    for i = 1:numel(solids)
        set_param(solids(i).block, 'DoExposeReferenceFrame', 'on');
    end

    % Set up KinematicsSolver with frame variables
    [ks, closed, tv, ids, keys] = local_build_ks(mdl, solids);

    % Parse poses Q and resolve per-frame joint positions
    [q_mat, n_frames, t_vec] = local_parse_poses(q, ids, keys, mdl, closed, tv, opts.time);

    % Frame selection for rendering
    if isempty(opts.stills) && isempty(opts.video)
        frames_to_solve = 1;
    elseif ~isempty(opts.video)
        frames_to_solve = 1:n_frames;
    else
        frames_to_solve = opts.stills;
    end
    frames_to_solve = unique(frames_to_solve);
    frames_to_solve(frames_to_solve < 1 | frames_to_solve > n_frames) = [];
    assert(~isempty(frames_to_solve), 'gs3dx:render', 'No valid frame indices to render');

    % Solve forward kinematics for solid poses
    solids = local_solve_poses(ks, solids, q_mat, closed, tv, frames_to_solve);

    % Format camera view
    [az, el] = local_parse_view(opts.view);
    focus = local_focus(solids, opts.focus, opts.focus_width);

    % Render figures / export files
    written_files = string.empty;
    if ~exist(opts.output_dir, 'dir')
        mkdir(opts.output_dir);
    end

    % Handle video recording
    if ~isempty(opts.video)
        video_path = opts.video;
        if ~isstring(video_path) && ~isempty(video_path) && ~contains(video_path, filesep)
            video_path = fullfile(opts.output_dir, video_path);
        end
        v_written = local_render_video(solids, frames_to_solve, t_vec, az, el, ...
            opts.markers, opts.ground, opts.title, opts.fps, video_path, opts.visible, focus);
        if ~isempty(v_written)
            written_files(end+1) = string(v_written);
        end
    end

    % Handle still images
    if ~isempty(opts.stills)
        for s_idx = 1:numel(opts.stills)
            f_num = opts.stills(s_idx);
            if f_num < 1 || f_num > n_frames
                continue;
            end
            if s_idx <= numel(opts.still_files)
                s_name = opts.still_files(s_idx);
            else
                s_name = sprintf('%s_frame_%04d.png', mdl, f_num);
            end
            if ~contains(s_name, filesep)
                s_path = fullfile(opts.output_dir, char(s_name));
            else
                s_path = char(s_name);
            end

            t_val = NaN;
            if ~isempty(t_vec) && f_num <= numel(t_vec)
                t_val = t_vec(f_num);
            end

            local_render_still(solids, f_num, t_val, az, el, ...
                opts.markers, opts.ground, opts.title, s_path, opts.visible, focus);
            written_files(end+1) = string(s_path);
        end
    end

    % Postconditions and output assembly
    out = struct();
    out.files = written_files;
    out.solids = solids;
    out.frames = frames_to_solve;
    out.view = [az el];
    out.focus = [];
    if ~isempty(focus)
        out.focus = cell2mat(arrayfun(focus.centre, frames_to_solve(:).', 'UniformOutput', false));
    end
    out.status = "success";
end

% -------------------------------------------------------------------------
% Helper: Discover Solid blocks and extract their geometry & materials
% -------------------------------------------------------------------------
function solids = local_discover_solids(mdl)
    ref_types = {'sm_lib/Body Elements/Brick Solid', ...
                 'sm_lib/Body Elements/Cylindrical Solid', ...
                 'sm_lib/Body Elements/Spherical Solid', ...
                 'sm_lib/Body Elements/Ellipsoidal Solid', ...
                 'sm_lib/Body Elements/File Solid'};
    all_blocks = {};
    for r = 1:numel(ref_types)
        blks = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
            'ReferenceBlock', ref_types{r});
        all_blocks = [all_blocks; blks]; %#ok<AGROW>
    end

    solids = struct('name', {}, 'block', {}, 'ref', {}, 'shape', {}, ...
        'color', {}, 'opacity', {}, 'params', {}, 'vertices_local', {}, ...
        'faces_local', {}, 'pose', {}, 'vertices_world', {});

    for i = 1:numel(all_blocks)
        b = all_blocks{i};
        gt = get_param(b, 'GraphicType');
        if strcmp(gt, 'None')
            continue;
        end
        ref = get_param(b, 'ReferenceBlock');

        % Resolve color and opacity
        color = [0.7 0.7 0.7];
        try
            c_val = slResolve(get_param(b, 'GraphicDiffuseColor'), b);
            if numel(c_val) == 3
                color = double(c_val(:)');
            end
        catch
        end

        opacity = 1.0;
        try
            op_val = slResolve(get_param(b, 'GraphicOpacity'), b);
            if isscalar(op_val)
                opacity = double(op_val);
            end
        catch
        end

        p = struct();
        switch ref
            case 'sm_lib/Body Elements/Cylindrical Solid'
                shape = "Cylinder";
                r_val = slResolve(get_param(b, 'CylinderRadius'), b);
                r_u = get_param(b, 'CylinderRadiusUnits');
                l_val = slResolve(get_param(b, 'CylinderLength'), b);
                l_u = get_param(b, 'CylinderLengthUnits');
                p.radius = local_to_meters(r_val, r_u);
                p.length = local_to_meters(l_val, l_u);
                [V_loc, F_loc] = local_cylinder_geometry(p.radius, p.length);

            case 'sm_lib/Body Elements/Spherical Solid'
                shape = "Sphere";
                r_val = slResolve(get_param(b, 'SphereRadius'), b);
                r_u = get_param(b, 'SphereRadiusUnits');
                p.radius = local_to_meters(r_val, r_u);
                [V_loc, F_loc] = local_sphere_geometry(p.radius);

            case 'sm_lib/Body Elements/Brick Solid'
                shape = "Brick";
                d_val = slResolve(get_param(b, 'BrickDimensions'), b);
                d_u = get_param(b, 'BrickDimensionsUnits');
                p.dimensions = local_to_meters(d_val, d_u);
                [V_loc, F_loc] = local_brick_geometry(p.dimensions);

            case 'sm_lib/Body Elements/Ellipsoidal Solid'
                shape = "Ellipsoid";
                r_val = slResolve(get_param(b, 'EllipsoidRadii'), b);
                r_u = get_param(b, 'EllipsoidRadiiUnits');
                p.radii = local_to_meters(r_val, r_u);
                [V_loc, F_loc] = local_ellipsoid_geometry(p.radii);

            case 'sm_lib/Body Elements/File Solid'
                shape = "Mesh";
                p.file = string(which(get_param(b, 'ExtGeomFileName')));
                assert(p.file ~= "", 'gs3dx:render', '%s: %s is not on the path', b, get_param(b, 'ExtGeomFileName'));
                [V_loc, F_loc] = local_mesh_geometry(p.file, local_to_meters(1, get_param(b, 'ExtGeomFileUnits')));

            otherwise
                continue;
        end

        s_entry = struct();
        s_entry.name = string(b);
        s_entry.block = b;
        s_entry.ref = string(ref);
        s_entry.shape = shape;
        s_entry.color = color;
        s_entry.opacity = opacity;
        s_entry.params = p;
        s_entry.vertices_local = V_loc;
        s_entry.faces_local = F_loc;
        s_entry.pose = struct('P', [], 'R', []);
        s_entry.vertices_world = {};

        solids(end+1) = s_entry; %#ok<AGROW>
    end
end

% -------------------------------------------------------------------------
% Helper: Unit conversion to SI metres
% -------------------------------------------------------------------------
function val_m = local_to_meters(val, unit_str)
    val = double(val);
    u = lower(strtrim(char(unit_str)));
    switch u
        case {'m', 'meter', 'meters'}
            scale = 1.0;
        case {'cm', 'centimeter', 'centimeters'}
            scale = 0.01;
        case {'mm', 'millimeter', 'millimeters'}
            scale = 0.001;
        case {'in', 'inch', 'inches'}
            scale = 0.0254;
        case {'ft', 'foot', 'feet'}
            scale = 0.3048;
        otherwise
            scale = 1.0;
    end
    val_m = val * scale;
end

% -------------------------------------------------------------------------
% Helper: Build KinematicsSolver with solid frame variables
% -------------------------------------------------------------------------
function [ks, closed, tv, ids, keys] = local_build_ks(mdl, solids)
    wf = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'ReferenceBlock', 'sm_lib/Frames and Transforms/World Frame');
    assert(~isempty(wf), 'gs3dx:render', 'World Frame not found in %s', mdl);
    world = [wf{1} '/W'];

    ks = simscape.multibody.KinematicsSolver(mdl);
    jp = ks.jointPositionVariables;
    [keys, ids] = gs3dx_joint_keys(mdl, jp);

    for i = 1:numel(solids)
        s_port = [solids(i).block '/R'];
        addFrameVariables(ks, sprintf('p%d', i), 'Translation', world, s_port);
    end
    for i = 1:numel(solids)
        s_port = [solids(i).block '/R'];
        addFrameVariables(ks, sprintf('r%d', i), 'Rotation', world, s_port);
    end

    % Grip loop closed joints (right elbow, shoulder, wrist), by block path:
    % a joint added to a variant renumbers the IDs after it
    closed = startsWith(keys, ["Right Elbow Joint/" "Right Shoulder Joint/" "Right Wrist and Hand/"]);
    tv = ids(~closed);

    addTargetVariables(ks, tv);
    if any(closed)
        addInitialGuessVariables(ks, ids(closed));
        addOutputVariables(ks, [string(ks.frameVariables.ID); ids(closed)]);
    else
        addOutputVariables(ks, string(ks.frameVariables.ID));
    end
end

% -------------------------------------------------------------------------
% Helper: Parse pose inputs (ik struct or matrix)
% -------------------------------------------------------------------------
function [q_mat, n_frames, t_vec] = local_parse_poses(q, ids, keys, mdl, closed, tv, user_time)
    t_vec = user_time;
    if isempty(q)
        % Target-free identity/zero pose
        q_mat = zeros(numel(ids), 1);
        n_frames = 1;
        if isempty(t_vec)
            t_vec = 0;
        end
        return;
    end

    if isstruct(q) && isfield(q, 'joint') && isfield(q, 'joint_ids')
        % IK struct
        ik_ids = string(q.joint_ids);
        if isfield(q, 'joint_keys') || (isfield(q, 'model') && ~strcmp(q.model, mdl) && ~isequal(ik_ids, ids))
            % Rows named by GS3DX_JOINT_KEYS, or solved on another variant:
            % match joints by block path; joints the IK does not have stay at 0
            if isfield(q, 'joint_keys')
                ik_keys = string(q.joint_keys);
            else
                [src_keys, src_ids] = gs3dx_joint_keys(char(q.model));
                [ok, r] = ismember(ik_ids, src_ids);
                assert(all(ok), 'gs3dx:render', 'IK joint IDs are not those of %s', q.model);
                ik_keys = src_keys(r);
            end
            [found, at] = ismember(keys, ik_keys);
            q_mat = zeros(numel(ids), size(q.joint, 2));
            q_mat(found, :) = q.joint(at(found), :);
        elseif isequal(ik_ids, ids)
            q_mat = q.joint;
        else
            % Reorder to match model IDs
            q_mat = zeros(numel(ids), size(q.joint, 2));
            for i = 1:numel(ids)
                idx = find(ik_ids == ids(i), 1);
                if ~isempty(idx)
                    q_mat(i, :) = q.joint(idx, :);
                end
            end
        end
        n_frames = size(q_mat, 2);
        if isfield(q, 't') && isempty(t_vec)
            t_vec = q.t;
        end
        return;
    end

    if isnumeric(q)
        if size(q, 1) == numel(ids)
            q_mat = double(q);
        elseif size(q, 1) == numel(tv)
            % Only target variables passed; zero fill closed joints
            q_mat = zeros(numel(ids), size(q, 2));
            q_mat(~closed, :) = double(q);
        else
            assert(size(q, 2) == numel(ids), 'gs3dx:render', ...
                'Dimension mismatch: expected %d joint position rows', numel(ids));
            q_mat = double(q.');
        end
        n_frames = size(q_mat, 2);
        return;
    end

    error('gs3dx:render:InvalidInput', 'Unrecognized pose input format');
end

% -------------------------------------------------------------------------
% Helper: Solve poses and build World geometry
% -------------------------------------------------------------------------
function solids = local_solve_poses(ks, solids, q_mat, closed, tv, frames_to_solve)
    ns = numel(solids);
    n_total_frames = size(q_mat, 2);

    for s_idx = 1:ns
        solids(s_idx).pose.P = zeros(3, n_total_frames);
        solids(s_idx).pose.R = zeros(3, 3, n_total_frames);
        solids(s_idx).vertices_world = cell(1, n_total_frames);
    end

    for f = frames_to_solve
        targets = q_mat(~closed, f);
        if any(closed)
            guess = q_mat(closed, f);
            [sol, st] = solve(ks, targets, guess);
        else
            [sol, st] = solve(ks, targets, []);
        end
        assert(st == 1, 'gs3dx:render', 'KinematicsSolver solve failed on frame %d (status %d)', f, st);

        for s_idx = 1:ns
            p_world = sol(3*(s_idx - 1) + 1 : 3*s_idx);
            rot_deg = sol(3*ns + 3*(s_idx - 1) + 1 : 3*ns + 3*s_idx);
            rot_rad = rot_deg * (pi / 180);
            R_world = local_rx(rot_rad(1)) * local_ry(rot_rad(2)) * local_rz(rot_rad(3));

            solids(s_idx).pose.P(:, f) = p_world;
            solids(s_idx).pose.R(:, :, f) = R_world;

            % Transform local vertices to World coordinates
            V_loc = solids(s_idx).vertices_local;
            solids(s_idx).vertices_world{f} = R_world * V_loc + p_world;
        end
    end
end

% -------------------------------------------------------------------------
% Geometry generators: Local vertex and face definitions
% -------------------------------------------------------------------------
function [V, F] = local_cylinder_geometry(R, L)
    % Cylindrical Solid: radius R, length L along local Z, centred at origin
    n = 24;
    theta = linspace(0, 2*pi, n + 1);
    theta(end) = [];
    x = R * cos(theta);
    y = R * sin(theta);
    
    % Vertices: top rim, bottom rim, top center, bottom center
    V_top = [x; y; (L/2) * ones(1, n)];
    V_bot = [x; y; (-L/2) * ones(1, n)];
    V_tc = [0; 0; L/2];
    V_bc = [0; 0; -L/2];
    V = [V_top, V_bot, V_tc, V_bc];

    % Side faces (quads converted to triangles)
    F = [];
    for i = 1:n
        next_i = mod(i, n) + 1;
        % Top vertices: 1..n, Bottom vertices: n+1..2n
        t1 = i; t2 = next_i;
        b1 = i + n; b2 = next_i + n;
        F = [F; t1 t2 b2; t1 b2 b1]; %#ok<AGROW>
    end
    % Top cap (center is 2*n + 1)
    tc_idx = 2*n + 1;
    for i = 1:n
        next_i = mod(i, n) + 1;
        F = [F; tc_idx i next_i]; %#ok<AGROW>
    end
    % Bottom cap (center is 2*n + 2)
    bc_idx = 2*n + 2;
    for i = 1:n
        next_i = mod(i, n) + 1;
        F = [F; bc_idx (next_i + n) (i + n)]; %#ok<AGROW>
    end
end

function [V, F] = local_sphere_geometry(R)
    % Spherical Solid: radius R, centred at origin
    [Xs, Ys, Zs] = sphere(20);
    [F, V] = surf2patch(R * Xs, R * Ys, R * Zs, 'triangles');
    V = V.';
end

function [V, F] = local_brick_geometry(dims)
    % Brick Solid: dimensions [dx, dy, dz] centered at origin
    dx = dims(1); dy = dims(2); dz = dims(3);
    x = dx/2 * [-1 1 1 -1 -1 1 1 -1];
    y = dy/2 * [-1 -1 1 1 -1 -1 1 1];
    z = dz/2 * [-1 -1 -1 -1 1 1 1 1];
    V = [x; y; z];

    % 12 triangular faces
    F = [
        1 2 6; 1 6 5; % front (-y)
        2 3 7; 2 7 6; % right (+x)
        3 4 8; 3 8 7; % back (+y)
        4 1 5; 4 5 8; % left (-x)
        5 6 7; 5 7 8; % top (+z)
        1 4 3; 1 3 2  % bottom (-z)
    ];
end

function [V, F] = local_mesh_geometry(file, scale)
% STL triangles (file units times SCALE, m), with coincident vertices
% merged so neighbouring facets share normals.
    tr = stlread(file);
    [V, ~, k] = unique(round(tr.Points * scale, 9), 'rows');
    F = reshape(k(tr.ConnectivityList), [], 3);
    V = V.';
end

function [V, F] = local_ellipsoid_geometry(radii)
    % Ellipsoidal Solid: radii [rx, ry, rz] centered at origin
    rx = radii(1); ry = radii(2); rz = radii(3);
    [Xe, Ye, Ze] = ellipsoid(0, 0, 0, rx, ry, rz, 20);
    [F, V] = surf2patch(Xe, Ye, Ze, 'triangles');
    V = V.';
end

% -------------------------------------------------------------------------
% Helper: Camera view parser
% -------------------------------------------------------------------------
function [az, el] = local_parse_view(view_opt)
    if ischar(view_opt) || isstring(view_opt)
        v = lower(strtrim(string(view_opt)));
        switch v
            case {"face-on", "fo"}
                az = 90; el = 5;    % camera on +X, in front of the golfer
            case {"down-the-line", "dtl"}
                az = 0; el = 5;     % camera on -Y, behind the golfer looking at the target
            case {"top", "overhead"}
                az = 0; el = 90;
            otherwise
                error('gs3dx:render', 'Unknown view "%s"', v);
        end
    elseif isnumeric(view_opt) && numel(view_opt) == 2
        az = double(view_opt(1));
        el = double(view_opt(2));
    else
        error('gs3dx:render', 'VIEW must be a view name or [azimuth elevation]');
    end
end

% -------------------------------------------------------------------------
% Helper: close-up centre per frame (a solid's position, or a fixed point)
% -------------------------------------------------------------------------
function focus = local_focus(solids, spec, width)
    focus = [];
    if isempty(spec)
        return
    end
    if isnumeric(spec)
        assert(numel(spec) == 3, 'gs3dx:render', 'FOCUS must be a solid name or [x y z]');
        p = double(spec(:));
        focus = struct('centre', @(f) p, 'width', width);
        return
    end
    k = find(endsWith([solids.name], "/" + string(spec)), 1);
    assert(~isempty(k), 'gs3dx:render', 'No drawn solid named %s to focus on', spec);
    P = solids(k).pose.P;
    focus = struct('centre', @(f) P(:, f), 'width', width);
end

% -------------------------------------------------------------------------
% Helper: Draw scene for frame f into given axes
% -------------------------------------------------------------------------
function local_draw_scene(ax, solids, f, t_val, az, el, markers, draw_ground, scene_title, focus)
    cla(ax);
    hold(ax, 'on');

    % Set viewpoint and lighting
    view(ax, az, el);
    camproj(ax, 'perspective');

    % Ground plane at shoe sole level (~ -1.02 m)
    if draw_ground
        [Xg, Yg] = meshgrid(-0.8:0.25:1.6, -1.6:0.25:1.6);
        Zg = -1.02 * ones(size(Xg));
        surf(ax, Xg, Yg, Zg, 'FaceColor', [0.88 0.90 0.88], 'EdgeColor', [0.80 0.82 0.80], ...
            'FaceAlpha', 0.6, 'AmbientStrength', 0.5);
    end

    % Draw every solid
    for i = 1:numel(solids)
        V_w = solids(i).vertices_world{f};
        F = solids(i).faces_local;
        color = solids(i).color;
        opacity = solids(i).opacity;

        patch(ax, 'Faces', F, 'Vertices', V_w.', ...
            'FaceColor', color, 'FaceAlpha', opacity, ...
            'EdgeColor', 'none', 'SpecularStrength', 0.15, ...
            'DiffuseStrength', 0.85, 'AmbientStrength', 0.35);
    end

    % Overlay markers if available
    if ~isempty(markers)
        if ndims(markers) == 3 && size(markers, 1) == 3 && size(markers, 2) ~= 3
            markers = permute(markers, [2 1 3]);
        end
        if size(markers, 3) >= f
            m_f = markers(:, :, f);
            valid_m = ~isnan(m_f(:, 1));
            scatter3(ax, m_f(valid_m, 1), m_f(valid_m, 2), m_f(valid_m, 3), ...
                28, [0.85 0.325 0.098], 'filled', 'MarkerEdgeColor', [0.2 0.2 0.2]);
        end
    end

    % Scene bounds and appearance
    axis(ax, 'equal');
    if isempty(focus)
        xlim(ax, [-0.6 1.4]);
        ylim(ax, [-1.2 1.2]);
        zlim(ax, [-1.15 1.15]);
    else
        c = focus.centre(f);
        w = focus.width;
        xlim(ax, c(1) + [-w w]);
        ylim(ax, c(2) + [-w w]);
        zlim(ax, c(3) + [-w w]);
    end
    grid(ax, 'on');
    set(ax, 'GridColor', [0.7 0.7 0.7], 'GridAlpha', 0.3);
    set(ax, 'Color', [0.96 0.97 0.98]);
    xlabel(ax, 'X (m, Facing)');
    ylabel(ax, 'Y (m, Target)');
    zlabel(ax, 'Z (m, Up)');

    % Lights
    delete(findall(ax, 'Type', 'light'));
    camlight(ax, 'headlight');
    light(ax, 'Position', [2 -3 4], 'Style', 'infinite', 'Color', [0.9 0.9 0.9]);
    lighting(ax, 'gouraud');

    % Text overlay
    header_str = scene_title;
    if ~isnan(t_val)
        time_str = sprintf('t = %.3f s | Frame %d', t_val, f);
    else
        time_str = sprintf('Frame %d', f);
    end
    if isempty(header_str)
        annot_str = time_str;
    else
        annot_str = sprintf('%s | %s', header_str, time_str);
    end
    text(ax, 0.03, 0.95, annot_str, 'Units', 'normalized', 'FontSize', 11, ...
        'FontWeight', 'bold', 'Color', [0.15 0.15 0.15], 'BackgroundColor', [1 1 1 0.8], ...
        'Margin', 4);
end

% -------------------------------------------------------------------------
% Helper: Render single still image
% -------------------------------------------------------------------------
function local_render_still(solids, f, t_val, az, el, markers, draw_ground, scene_title, outfile, is_vis, focus)
    vis_str = 'off';
    if is_vis
        vis_str = 'on';
    end
    fig = figure('Visible', vis_str, 'Color', 'w', 'Position', [100 100 1024 768]);
    ax = axes('Parent', fig);
    
    local_draw_scene(ax, solids, f, t_val, az, el, markers, draw_ground, scene_title, focus);
    
    exportgraphics(fig, outfile, 'Resolution', 120);
    close(fig);
end

% -------------------------------------------------------------------------
% Helper: Render swing video (MP4 or GIF fallback)
% -------------------------------------------------------------------------
function video_file = local_render_video(solids, frames, t_vec, az, el, ...
    markers, draw_ground, scene_title, fps, outfile, is_vis, focus)

    vis_str = 'off';
    if is_vis
        vis_str = 'on';
    end
    fig = figure('Visible', vis_str, 'Color', 'w', 'Position', [100 100 800 600]);
    ax = axes('Parent', fig);

    video_file = outfile;
    [out_dir, base, ext] = fileparts(outfile);
    if isempty(ext)
        ext = '.mp4';
        video_file = fullfile(out_dir, [base ext]);
    end

    use_mp4 = strcmpi(ext, '.mp4');
    vw = [];
    if use_mp4
        try
            vw = VideoWriter(video_file, 'MPEG-4');
            vw.FrameRate = fps;
            vw.Quality = 85;
            open(vw);
        catch
            use_mp4 = false;
            video_file = fullfile(out_dir, [base '.gif']);
        end
    end

    for i = 1:numel(frames)
        f = frames(i);
        t_val = NaN;
        if ~isempty(t_vec) && f <= numel(t_vec)
            t_val = t_vec(f);
        end
        local_draw_scene(ax, solids, f, t_val, az, el, markers, draw_ground, scene_title, focus);
        drawnow;
        frame_data = getframe(fig);

        if use_mp4
            writeVideo(vw, frame_data);
        else
            % Write animated GIF
            [A, map] = rgb2ind(frame_data.cdata, 256);
            if i == 1
                imwrite(A, map, video_file, 'gif', 'LoopCount', Inf, 'DelayTime', 1/fps);
            else
                imwrite(A, map, video_file, 'gif', 'WriteMode', 'append', 'DelayTime', 1/fps);
            end
        end
    end

    if use_mp4 && ~isempty(vw)
        close(vw);
    end
    close(fig);
end

% -------------------------------------------------------------------------
% Rotation matrix helpers
% -------------------------------------------------------------------------
function R = local_rx(a)
    R = [1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)];
end

function R = local_ry(a)
    R = [cos(a) 0 sin(a); 0 1 0; -sin(a) 0 cos(a)];
end

function R = local_rz(a)
    R = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];
end
