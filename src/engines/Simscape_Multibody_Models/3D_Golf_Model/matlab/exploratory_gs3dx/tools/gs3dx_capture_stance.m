function cap = gs3dx_capture_stance(file)
%GS3DX_CAPTURE_STANCE  Stance and leg measurements from the tour-average capture (#10985).
%
%   CAP = GS3DX_CAPTURE_STANCE() reads the canonical tour-average driver
%   capture (data/C3D_TA_Driver.c3d at the repository root) and
%   CAP = GS3DX_CAPTURE_STANCE(FILE) reads another C3D with the same marker
%   set.  The file is read through Python ezc3d (MATLAB pyenv) and never
%   written; nothing is filtered, gap-filled or retimed.
%
%   Axes.  The capture is Y-up; it is converted to Z-up as (x, -z, y).
%   The golfer's target frame at address (first frame) is
%     lateral  horizontal LAnkleOut - RAnkleOut (toward the target for a
%              right-handed golfer),
%     up       +Z,
%     facing   lateral x up (checked to point from the back waist markers
%              to the front ones),
%   the same axis order as the GS3DX leg frame (x facing, y target, z up).
%   Positions are relative to the pelvis centre (mean of the four waist
%   markers) at address.
%
%   CAP fields (lengths in m, angles in deg):
%     .file .sha256 .rate_hz .n_frames .force_plates_used .n_analog
%     .missing          struct: frames with no sample per leg/waist marker
%     .impact_frame     peak speed of the club-head marker cluster
%     .ankle_L/.ankle_R, .toe_L/.toe_R   3x1 address positions (toe = mean
%                       of ToeIn and ToeOut) in the target frame
%     .stance_width     horizontal LAnkleOut-RAnkleOut distance at address
%     .foot_yaw_L/_R    toe direction from facing, positive toward target
%     .foot_width_L/_R  mean ToeIn-ToeOut distance
%     .ankle_to_toe_L/_R, .shank_proxy_L/_R, .waist_width   mean distances
%     .knee_angle_L/_R  angle at KneeOut between the waist marker (lowered
%                       0.08 m) and AnkleOut at address; 180 = straight.
%                       A marker proxy, not a joint angle.
%     .excursion        struct: max horizontal and vertical travel (m) of
%                       each foot marker over the trial, [horizontal vertical]
%
%   Markers are skin/shoe markers, not joint centres: every length here is
%   a proxy.  See docs/DATA_AUDIT.md.

    arguments
        file (1,:) char = local_default_file()
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

    cap.file = file;
    cap.sha256 = local_sha256(file);
    cap.rate_hz = rate(1);
    cap.n_frames = size(pts, 3);
    cap.force_plates_used = double(py.numpy.asarray(get(get(get(params, 'FORCE_PLATFORM'), 'USED'), 'value')));
    cap.n_analog = size(analogs, 2);

    zup = cat(1, pts(1, :, :), -pts(3, :, :), pts(2, :, :));   % 3 x markers x frames
    marker = @(name) squeeze(zup(:, local_index(labels, name), :));   % 3 x frames
    feet = ["LAnkleOut", "RAnkleOut", "LToeIn", "LToeOut", "RToeIn", "RToeOut"];
    legs = ["LKneeOut", "RKneeOut", feet];
    waist = ["WaistLeft", "WaistRight", "WaistLBack", "WaistRBack"];
    cap.missing = struct();
    for name = [waist, legs]
        cap.missing.(name) = nnz(any(isnan(marker(name)), 1));
    end

    a = 1;   % address frame
    heights = cellfun(@(n) marker(n), cellstr([feet, "LKneeOut", waist]), 'UniformOutput', false);
    lowest = cellfun(@(m) m(3, a), heights);
    assert(max(lowest(1:numel(feet))) < min(lowest(numel(feet) + 1:end)), 'gs3dx:capture', ...
        'Postcondition: foot markers are not the lowest at address; the axis conversion is wrong');

    up = [0; 0; 1];
    lateral = marker("LAnkleOut") - marker("RAnkleOut");
    lateral = [lateral(1:2, a); 0];
    lateral = lateral / norm(lateral);
    facing = cross(lateral, up);
    front = (marker("WaistLeft") + marker("WaistRight")) / 2;
    back = (marker("WaistLBack") + marker("WaistRBack")) / 2;
    assert(dot(facing, front(:, a) - back(:, a)) > 0, 'gs3dx:capture', ...
        'Postcondition: facing does not point from the back waist markers to the front');
    S = [facing, lateral, up];
    centre = (front(:, a) + back(:, a)) / 2;
    local = @(p) S.' * (p(:, a) - centre);

    toe = @(s) (marker(s + "ToeIn") + marker(s + "ToeOut")) / 2;
    dist = @(m1, m2) mean(vecnorm(m1 - m2), 'omitnan');
    for s = ["L", "R"]
        cap.("ankle_" + s) = local(marker(s + "AnkleOut"));
        cap.("toe_" + s) = local(toe(s));
        d = cap.("toe_" + s) - cap.("ankle_" + s);
        cap.("foot_yaw_" + s) = atan2d(d(2), d(1));
        cap.("foot_width_" + s) = dist(marker(s + "ToeIn"), marker(s + "ToeOut"));
        cap.("ankle_to_toe_" + s) = dist(marker(s + "AnkleOut"), toe(s));
        cap.("shank_proxy_" + s) = dist(marker(s + "KneeOut"), marker(s + "AnkleOut"));
        hip = marker(ternary(s == "L", "WaistLeft", "WaistRight")) - [0; 0; 0.08];
        knee = marker(s + "KneeOut");
        u = hip(:, a) - knee(:, a);
        v = marker(s + "AnkleOut");
        v = v(:, a) - knee(:, a);
        cap.("knee_angle_" + s) = acosd(dot(u, v) / (norm(u) * norm(v)));
    end
    cap.stance_width = norm(cap.ankle_L(1:2) - cap.ankle_R(1:2));
    cap.waist_width = dist(marker("WaistLeft"), marker("WaistRight"));
    cap.excursion = struct();
    for name = feet
        m = marker(name);
        h = vecnorm(m(1:2, :) - m(1:2, a));
        cap.excursion.(name) = [max(h, [], 'omitnan'), max(m(3, :), [], 'omitnan') - min(m(3, :), [], 'omitnan')];
    end
    cap.impact_frame = local_impact_frame(zup, labels, marker);
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

function f = local_impact_frame(zup, labels, marker)
% Peak speed of the club marker cluster farther from the wrists at address.
    wrists = (marker("LWristTop") + marker("RWristTop")) / 2;
    best = -Inf;
    for cluster = ["Marker_2:2:", "Marker_3:3:"]
        idx = find(startsWith(labels, cluster));
        centroid = squeeze(mean(zup(:, idx, :), 2));
        away = norm(centroid(:, 1) - wrists(:, 1));
        if away > best
            best = away;
            head = centroid;
        end
    end
    speed = vecnorm(diff(head, 1, 2));
    [~, f] = max(speed);
end

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end

function hex = local_sha256(file)
    fid = fopen(file, 'r');
    bytes = fread(fid, Inf, '*uint8');
    fclose(fid);
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(bytes);
    hex = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2).', 1, []));
end
