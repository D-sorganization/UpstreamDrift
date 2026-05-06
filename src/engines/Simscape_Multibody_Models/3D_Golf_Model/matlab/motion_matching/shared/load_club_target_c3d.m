function target = load_club_target_c3d(c3d_path, opts)
%LOAD_CLUB_TARGET_C3D  Read club 6-DOF trajectory from a C3D file.
%
%   TARGET = LOAD_CLUB_TARGET_C3D(C3D_PATH, OPTS) returns a target struct as
%   specified in CLUB_IK_SPEC.md.
%
%   Implementation choice: call the existing Python reader at
%   src/engines/.../python/src/c3d_reader.py via `pyrunfile`, per
%   CLUB_IK_SPEC.md §"Implementation notes".  This avoids duplicating the
%   C3D parser in MATLAB.
%
%   Marker mapping: the Optical C3D file in
%   src/engines/.../matlab/Data/Optical C3D Files/ has not previously been
%   parsed.  This loader uses a heuristic to identify butt and clubhead:
%     - Preferred: explicit names (case-insensitive substring match) for
%       "butt", "grip" (-> butt) and "head", "club" (-> clubhead).
%     - Fallback: take the marker pair with the largest mean separation that
%       is plausible as a shaft length (0.7-1.4 m).  This captures the
%       butt<->head endpoints without assuming naming conventions.
%
%   Whatever mapping is chosen is written to opts.verbosity-gated log so the
%   companion test (test_marker_mapping_documented_in_log) can record it.
%
%   Orientation: the club_quat is derived from the unit shaft vector using a
%   minimal-rotation fix (z-axis of body frame == shaft direction, x-axis is
%   the world-x projected component).  Two-marker data does not constrain
%   roll about the shaft; downstream callers must accept the residual gauge.
%
%   Preconditions: enforced via arguments block.
%   Postconditions: per CLUB_IK_SPEC.md §"Validation rules".
    arguments
        c3d_path (1,1) string {mustBeFile}
        opts     (1,1) struct = default_align_options()
    end

    log_info(opts, "Loading C3D club target from %s", c3d_path);

    % --- Call Python c3d_reader via pyrunfile -----------------------------
    [points_df, metadata] = read_c3d_via_python(c3d_path);

    marker_labels = string(metadata.marker_labels);
    frame_rate    = double(metadata.frame_rate);
    if frame_rate <= 0
        error("load_club_target_c3d:badFrameRate", ...
              "C3D file reports frame_rate <= 0 (%g)", frame_rate);
    end

    % --- Pivot tidy points dataframe into Mx3 per marker ------------------
    [marker_xyz, present_markers] = pivot_marker_array(points_df, marker_labels);
    log_info(opts, "C3D markers present: %s", strjoin(present_markers, ", "));

    [butt_idx, head_idx, mapping_method] = pick_butt_and_head(present_markers, ...
                                                              marker_xyz);
    log_info(opts, "Marker mapping (%s): butt=%s, clubhead=%s", ...
        mapping_method, present_markers(butt_idx), present_markers(head_idx));

    butt_m     = marker_xyz(:, :, butt_idx);
    clubhead_m = marker_xyz(:, :, head_idx);

    % Drop frames where either endpoint is NaN
    keep = all(isfinite(butt_m), 2) & all(isfinite(clubhead_m), 2);
    if nnz(keep) < 5
        error("load_club_target_c3d:tooFewValidFrames", ...
              "Only %d non-NaN frames after marker selection", nnz(keep));
    end
    butt_m     = butt_m(keep, :);
    clubhead_m = clubhead_m(keep, :);
    M          = size(butt_m, 1);

    t_raw = (0:M-1).' / frame_rate;

    club_quat = shaft_vector_to_quaternion(butt_m, clubhead_m);

    raw = struct( ...
        "time",      t_raw, ...
        "butt",      butt_m, ...
        "clubhead",  clubhead_m, ...
        "club_quat", club_quat);

    aligned = align_to_simulation_grid(raw, opts);

    [~, fname, fext] = fileparts(c3d_path);
    source = struct( ...
        "filename",   string(strcat(fname, fext)), ...
        "format",     "c3d", ...
        "subject_id", string(opts.subject_id), ...
        "trial_id",   string(opts.trial_id), ...
        "sha256",     string(sha256_of_file(c3d_path)));

    target = struct( ...
        "time",       aligned.time, ...
        "butt",       aligned.butt, ...
        "clubhead",   aligned.clubhead, ...
        "club_quat",  aligned.club_quat, ...
        "impact_idx", aligned.impact_idx, ...
        "source",     source);

    % --- Postconditions ----------------------------------------------------
    N = numel(target.time);
    assert(all(diff(target.time) > 0), "Postcondition: time strictly increasing");
    assert(abs(target.time(1)) < eps, "Postcondition: time(1) must be 0");
    assert(size(target.butt,1) == N && size(target.clubhead,1) == N && ...
           size(target.club_quat,1) == N, ...
        "Postcondition: trajectory rows must equal numel(time)");
    assert(all(isfinite(target.butt(:))) && all(isfinite(target.clubhead(:))), ...
        "Postcondition: positions must be finite");
    assert(all(vecnorm(target.butt,2,2) < 5) && ...
           all(vecnorm(target.clubhead,2,2) < 5), ...
        "Postcondition: ||r|| < 5 m");
    qn = sqrt(sum(target.club_quat.^2, 2));
    assert(all(abs(qn - 1) < 1e-6), ...
        "Postcondition: club_quat unit-norm to 1e-6");
    assert(target.impact_idx >= 1 && target.impact_idx <= N, ...
        "Postcondition: 1 <= impact_idx <= N");
    assert(strlength(target.source.sha256) == 64, ...
        "Postcondition: source.sha256 must be 64 hex chars");

    log_info(opts, "Loaded %d frames; impact at idx=%d (t=%.3fs)", ...
        N, target.impact_idx, target.time(target.impact_idx));
end


function [points_df, metadata] = read_c3d_via_python(c3d_path)
    % Locate c3d_reader.py relative to this file.
    here = fileparts(mfilename("fullpath"));
    repo_python_src = fullfile(here, "..", "..", "..", "python", "src");
    repo_python_src = char(java.io.File(repo_python_src).getCanonicalPath());

    if exist(repo_python_src, "dir") ~= 7
        error("load_club_target_c3d:pythonSrcMissing", ...
              "Cannot locate %s — required for c3d_reader.py", repo_python_src);
    end

    % pyrun a small wrapper to ingest one file and return frame, marker,
    % x, y, z arrays plus metadata.
    cmd = [
        "import sys, pathlib"
        "p = pathlib.Path(repo_src)"
        "sys.path.insert(0, str(p))"
        "from c3d_reader import C3DDataReader"
        "rdr = C3DDataReader(file_path)"
        "df = rdr.points_dataframe(include_time=True, target_units='m')"
        "md = rdr.get_metadata()"
        "frames = df['frame'].to_numpy().astype('int64')"
        "markers = df['marker'].to_numpy().astype('U64')"
        "xs = df['x'].to_numpy().astype('float64')"
        "ys = df['y'].to_numpy().astype('float64')"
        "zs = df['z'].to_numpy().astype('float64')"
        "labels = list(md.marker_labels)"
        "frame_rate = float(md.frame_rate)"
        "frame_count = int(md.frame_count)"
        "units = str(md.units)"
    ];
    pyOut = pyrun(cmd, ...
        ["frames" "markers" "xs" "ys" "zs" "labels" "frame_rate" "frame_count" "units"], ...
        repo_src=string(repo_python_src), ...
        file_path=string(c3d_path));

    points_df = struct( ...
        "frame",  double(pyOut.frames), ...
        "marker", string(pyOut.markers), ...
        "x",      double(pyOut.xs), ...
        "y",      double(pyOut.ys), ...
        "z",      double(pyOut.zs));

    metadata = struct( ...
        "marker_labels", string(cell(pyOut.labels)), ...
        "frame_rate",    double(pyOut.frame_rate), ...
        "frame_count",   double(pyOut.frame_count), ...
        "units",         string(pyOut.units));
end


function [marker_xyz, present_markers] = pivot_marker_array(points_df, marker_labels)
    % Returns Mx3xK where K = number of markers actually present.
    unique_present = unique(points_df.marker, "stable");
    [~, order] = ismember(unique_present, marker_labels);
    order = order(order > 0);
    present_markers = marker_labels(order);

    n_frames = max(points_df.frame) + 1;
    K = numel(present_markers);
    marker_xyz = nan(n_frames, 3, K);

    for k = 1:K
        mask = points_df.marker == present_markers(k);
        f = points_df.frame(mask) + 1;
        marker_xyz(f, 1, k) = points_df.x(mask);
        marker_xyz(f, 2, k) = points_df.y(mask);
        marker_xyz(f, 3, k) = points_df.z(mask);
    end
end


function [butt_idx, head_idx, method] = pick_butt_and_head(markers, xyz)
    butt_keys = ["butt", "grip", "hand"];
    head_keys = ["head", "club", "face", "tip"];

    [butt_idx, butt_hit] = find_first_match(markers, butt_keys);
    [head_idx, head_hit] = find_first_match(markers, head_keys);

    if butt_hit && head_hit && butt_idx ~= head_idx
        method = "name-based";
        return;
    end

    % Fallback: find marker pair with mean separation in [0.7, 1.4] m.
    K = size(xyz, 3);
    best_pair  = [NaN NaN];
    best_score = -Inf;
    for a = 1:K
        for b = a+1:K
            d = squeeze(xyz(:, :, a) - xyz(:, :, b));
            sep = sqrt(sum(d.^2, 2));
            mean_sep = mean(sep, "omitnan");
            if mean_sep >= 0.7 && mean_sep <= 1.4
                if mean_sep > best_score
                    best_score = mean_sep;
                    best_pair  = [a, b];
                end
            end
        end
    end
    if any(isnan(best_pair))
        error("load_club_target_c3d:noPlausibleShaft", ...
              "No marker pair with mean separation in [0.7, 1.4] m");
    end
    % Convention: butt is the higher-of-mean-y or lower-z marker (closer to
    % hands).  Without semantics we just pick the one with smaller mean speed
    % (the butt swings on a smaller radius).
    speed_a = mean_speed(xyz(:, :, best_pair(1)));
    speed_b = mean_speed(xyz(:, :, best_pair(2)));
    if speed_a <= speed_b
        butt_idx = best_pair(1);
        head_idx = best_pair(2);
    else
        butt_idx = best_pair(2);
        head_idx = best_pair(1);
    end
    method = "geometric-fallback";
end


function s = mean_speed(xyz_marker)
    d = diff(xyz_marker);
    s = mean(sqrt(sum(d.^2, 2)), "omitnan");
end


function [idx, hit] = find_first_match(markers, keys)
    idx = 0;
    hit = false;
    for k = 1:numel(keys)
        m = find(contains(lower(markers), lower(keys(k))), 1, "first");
        if ~isempty(m)
            idx = m;
            hit = true;
            return;
        end
    end
    if idx == 0
        idx = 1;  % avoid invalid index when caller checks `hit`
    end
end


function quats = shaft_vector_to_quaternion(butt, head)
    M = size(butt, 1);
    R = zeros(3, 3, M);
    for k = 1:M
        z_axis = head(k, :) - butt(k, :);
        z_axis = z_axis / max(norm(z_axis), eps);
        % Construct a stable basis: take world X projected orthogonal to z.
        x_world = [1, 0, 0];
        if abs(dot(x_world, z_axis)) > 0.95
            x_world = [0, 1, 0];
        end
        x_axis = x_world - dot(x_world, z_axis) * z_axis;
        x_axis = x_axis / max(norm(x_axis), eps);
        y_axis = cross(z_axis, x_axis);
        R(:, :, k) = [x_axis.', y_axis.', z_axis.'];
    end
    quats = rotmat_to_quaternion(R);
end


function log_info(opts, fmt, varargin)
    if ismember(string(opts.verbosity), ["Normal", "Verbose", "Debug"])
        fprintf("[load_club_target_c3d] " + fmt + "\n", varargin{:});
    end
end
