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
%     .labels   marker labels (string row)
%     .points   3 x markers x frames, m, Z-up: the capture's Y-up axes
%               converted as (x, -z, y)
%     .marker   function handle: .marker(name) is the 3 x frames track of
%               the one marker called NAME (error if absent or repeated)
%     .club_head, .club_grip  3 x frames centroids of the two club marker
%               clusters (head = the one farther from the wrists at
%               address); NaN where a cluster marker is missing
%     .impact_frame  frame (1-based) of peak club-head cluster speed
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
    pts = double(py.numpy.asarray(get(get(c, 'data'), 'points')));   % 4 x markers x frames
    labels = string(cell(get(get(get(params, 'POINT'), 'LABELS'), 'value')));
    rate = double(py.numpy.asarray(get(get(get(params, 'POINT'), 'RATE'), 'value')));
    analogs = double(py.numpy.asarray(get(get(c, 'data'), 'analogs')));

    zup = cat(1, pts(1, :, :), -pts(3, :, :), pts(2, :, :));   % 3 x markers x frames
    cap.file = file;
    cap.rate_hz = rate(1);
    cap.n_frames = size(pts, 3);
    cap.force_plates_used = double(py.numpy.asarray(get(get(get(params, 'FORCE_PLATFORM'), 'USED'), 'value')));
    cap.n_analog = size(analogs, 2);
    cap.labels = labels;
    cap.points = zup;
    cap.marker = @(name) reshape(zup(:, local_index(labels, name), :), 3, []);
    [cap.club_head, cap.club_grip] = local_club(zup, labels, cap.marker);
    [~, cap.impact_frame] = max(vecnorm(diff(cap.club_head, 1, 2)));
    cap.target_frame = local_target_frame(cap.marker);
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
