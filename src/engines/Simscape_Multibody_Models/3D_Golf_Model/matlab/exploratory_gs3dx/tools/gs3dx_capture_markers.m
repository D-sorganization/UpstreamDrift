function cap = gs3dx_capture_markers(file)
%GS3DX_CAPTURE_MARKERS  Read a C3D capture's markers, converted to Z-up (#10985, #11011).
%
%   CAP = GS3DX_CAPTURE_MARKERS() reads the canonical tour-average driver
%   capture (data/C3D_TA_Driver.c3d in the nearest ancestor folder that has
%   one); CAP = GS3DX_CAPTURE_MARKERS(FILE) reads another C3D ('' = the
%   default).  The file is
%   read through Python ezc3d (MATLAB pyenv) and never written; nothing is
%   filtered, gap-filled or retimed.
%
%   CAP fields:
%     .file .rate_hz .n_frames .force_plates_used .n_analog
%     .source_units  units declared in C3D POINT:UNITS ('m', 'mm', or 'cm')
%     .units         SI units of .points ('m')
%     .missing_mask  1 x markers x frames logical mask; true where sample
%                    is missing or invalid
%     .event_qualification  explicit event qualification semantics
%                    ('PROXY_ONLY_NO_EXPLICIT_CONTACT'); computed timings
%                    are proxies. This importer does not qualify EVENT data.
%     .labels   marker labels (string row)
%     .points   3 x markers x frames, m, Z-up: the capture's Y-up axes
%               converted as (x, -z, y)
%     .marker   function handle: .marker(name) is the 3 x frames track of
%               the one marker called NAME (error if absent or repeated)
%     .club_head, .club_grip  3 x frames centroids of the two club marker
%               clusters (head = the one farther from the wrists at
%               address); NaN where a cluster marker is missing
%     .impact_frame  frame (1-based) of peak club-head cluster speed proxy;
%               the head peaks just before it reaches the ball
%     .ball_time    capture frame (fractional, 1-based) address-line crossing
%               proxy: frame at which the club head cluster centroid, moving
%               toward the target, returns to its address position along the
%               target line (proxy only, not measured contact)
%     .ball_frame   round(.ball_time)
%     .target_frame  3x3 [facing, lateral, up] columns at address (frame 1):
%               lateral is the horizontal RAnkleOut -> LAnkleOut direction
%               (toward the target for a right-handed golfer), up is +Z and
%               facing = lateral x up, checked to point from the back waist
%               markers to the front ones

    arguments
        file (1,:) char = ''
    end
    if isempty(file)
        file = local_default_file();
    end
    assert(isfile(file), 'gs3dx:capture', 'Capture not found: %s', file);
    try
        ez = py.importlib.import_module('ezc3d');
    catch err
        error('gs3dx:capture', 'Python ezc3d is required (pyenv %s): %s', pyenv().Executable, err.message);
    end
    c = ez.c3d(file);
    get = @(obj, key) py.operator.getitem(obj, key);
    params = get(c, 'parameters');
    data = get(c, 'data');

    % Fail-closed extraction of POINT:UNITS metadata
    try
        point_params = get(params, 'POINT');
        units_param = get(point_params, 'UNITS');
        units_val = cell(get(units_param, 'value'));
        assert(numel(units_val)==1, 'POINT:UNITS must contain exactly one unit');
        source_units = strtrim(char(units_val{1}));
        assert(~isempty(source_units), 'POINT:UNITS value string is empty');
    catch err
        error('gs3dx:capture', 'Missing or invalid POINT:UNITS metadata in %s: %s', file, err.message);
    end

    % Fail-closed extraction of meta_points residuals metadata
    try
        meta_points = get(data, 'meta_points');
        res_py = get(meta_points, 'residuals');
        residuals = double(py.numpy.asarray(res_py));
    catch err
        error('gs3dx:capture', 'Missing or invalid data.meta_points.residuals metadata in %s: %s', file, err.message);
    end

    pts = double(py.numpy.asarray(get(data, 'points')));   % 4 x markers x frames
    labels = string(cell(get(get(point_params, 'LABELS'), 'value')));
    rate = double(py.numpy.asarray(get(get(point_params, 'RATE'), 'value')));
    analogs = double(py.numpy.asarray(get(data, 'analogs')));

    % Convert homogeneous points to SI Z-up via pure helper with residual masking
    [zup, missing_mask] = gs3dx_capture_points(pts, source_units, residuals);

    cap.file = file;
    cap.source_units = source_units;
    cap.units = 'm';
    cap.rate_hz = rate(1);
    cap.n_frames = size(pts, 3);
    cap.force_plates_used = double(py.numpy.asarray(get(get(get(params, 'FORCE_PLATFORM'), 'USED'), 'value')));
    cap.n_analog = size(analogs, 2);
    cap.labels = labels;
    cap.points = zup;
    cap.missing_mask = missing_mask;
    cap.event_qualification = 'PROXY_ONLY_NO_EXPLICIT_CONTACT';
    cap.marker = @(name) reshape(zup(:, local_index(labels, name), :), 3, []);
    [cap.club_head, cap.club_grip] = local_club(zup, labels, cap.marker);
    [~, cap.impact_frame] = max(vecnorm(diff(cap.club_head, 1, 2)));
    cap.target_frame = local_target_frame(cap.marker);
    [cap.ball_time, cap.ball_frame] = local_ball(cap.club_head, cap.target_frame(:, 2), cap.impact_frame);
end

function [t, f] = local_ball(head, lateral, impact)
% First crossing, from the trail side, of the head's address position along
% the target line, searched from 30 frames before peak speed.
% Address-line crossing proxy: proxy only, not measured contact.
    y = lateral.' * (head - head(:, 1));
    k = find(y(1:end - 1) < 0 & y(2:end) >= 0);
    k = k(k >= impact - 30);
    assert(~isempty(k), 'gs3dx:capture', 'The club head never returns to its address position');
    k = k(1);
    t = k + -y(k) / (y(k + 1) - y(k));
    f = round(t);
end

function S = local_target_frame(marker)
% [facing, lateral, up] at address (frame 1); lateral from the ankles.
    up = [0; 0; 1];
    lateral = marker("LAnkleOut") - marker("RAnkleOut");
    lateral = [lateral(1:2, 1); 0];
    lateral = lateral / norm(lateral);
    facing = cross(lateral, up);
    front = (marker("WaistLeft") + marker("WaistRight")) / 2;
    back = (marker("WaistLBack") + marker("WaistRBack")) / 2;
    assert(dot(facing, front(:, 1) - back(:, 1)) > 0, 'gs3dx:capture', ...
        'Postcondition: facing does not point from the back waist markers to the front');
    S = [facing, lateral, up];
end

function [head, grip] = local_club(zup, labels, marker)
% Club marker cluster centroids: the head is the cluster farther from the
% wrists at address, the grip the nearer one.
    wrists = (marker("LWristTop") + marker("RWristTop")) / 2;
    c = cell(1, 2);
    away = zeros(1, 2);
    clusters = ["Marker_2:2:", "Marker_3:3:"];
    for k = 1:2
        idx = find(startsWith(labels, clusters(k)));
        assert(~isempty(idx), 'gs3dx:capture', 'Club cluster %s not found', clusters(k));
        c{k} = reshape(mean(zup(:, idx, :), 2), 3, []);
        away(k) = norm(c{k}(:, 1) - wrists(:, 1));
    end
    [~, h] = max(away);
    head = c{h};
    grip = c{3 - h};
end

function file = local_default_file()
% data/C3D_TA_Driver.c3d in the nearest ancestor folder that has one.
    d = fileparts(mfilename('fullpath'));
    while true
        file = fullfile(d, 'data', 'C3D_TA_Driver.c3d');
        parent = fileparts(d);
        if isfile(file) || strcmp(parent, d)
            return;
        end
        d = parent;
    end
end

function k = local_index(labels, name)
    k = find(labels == name);
    assert(isscalar(k), 'gs3dx:capture', 'Marker %s not found exactly once', name);
end
