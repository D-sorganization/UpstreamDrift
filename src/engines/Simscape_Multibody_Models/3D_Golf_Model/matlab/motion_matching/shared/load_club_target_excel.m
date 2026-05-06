function target = load_club_target_excel(xlsx_path, sheet_name, opts)
%LOAD_CLUB_TARGET_EXCEL  Read club 6-DOF trajectory from a Wiffle-style xlsx file.
%
%   TARGET = LOAD_CLUB_TARGET_EXCEL(XLSX_PATH, SHEET_NAME, OPTS) returns a
%   target struct as specified in CLUB_IK_SPEC.md.
%
%   Implementation choice: native MATLAB parser (readmatrix).
%
%   Sheet layout — verified from the Definitions tab + the per-sheet
%   header band (rows 1-3):
%     row 1 (event markers): "ProV1" | <blank> | A | <addr#> | T | <top#>
%                            | I | <impact#> | F | <finish#> | CHS | <mph>
%     row 2 (group labels):  "Mid-hands" (cols 3-14) | "Center of club face" (cols 15-26)
%     row 3 (column names):  Sample# | Time | X | Y | Z | Xx Xy Xz Yx Yy Yz Zx Zy Zz
%                                                   | X | Y | Z | Xx ... Zz
%     row 4..N (data):       sampling rate 240 Hz; time in seconds with
%                            negative values for the 1 s of pre-address data.
%
%   IMPORTANT — Units. The Definitions sheet states "all displacements
%   reported in inches", but the actual values give a constant
%   mid-hands-to-clubhead distance of 106.93 across every frame.  In
%   inches that would be 2.71 m (impossible — no golf club is 9 ft long).
%   In **centimetres** it's 1.07 m, exactly a typical iron/fairway-wood
%   shaft.  Independent confirmation: the Z-velocity of the clubhead at
%   impact integrates to a CHS that matches the documented 114.5 mph
%   value when treated as cm/s, not in/s.  We therefore parse the
%   positions as **centimetres** despite the boilerplate text in the
%   Definitions sheet.
%
%   The club_quat field is the *club-head* rotation matrix converted to a
%   unit quaternion with q(1) >= 0.
%
%   Output target struct gains the additional field
%       .events  struct with fields A, T, I, F (sample numbers per
%                                              the row-1 header) and CHS_mph.
%   The .impact_idx is taken from .events.I when available rather than
%   the clubhead-speed argmax heuristic — the documented value is
%   authoritative.
%
%   Permitted SHEET_NAME values: "TW_wiffle", "TW_ProV1", "GW_wiffle",
%   "GW_ProV11" (CLUB_IK_SPEC.md §"Source formats supported").
%
%   Preconditions: enforced via the arguments block.
%   Postconditions: per CLUB_IK_SPEC.md §"Validation rules" (see asserts at
%   end of function).
    arguments
        xlsx_path  (1,1) string {mustBeFile}
        sheet_name (1,1) string {mustBeMember(sheet_name, ...
            ["TW_wiffle", "TW_ProV1", "GW_wiffle", "GW_ProV11"])}
        opts       (1,1) struct = default_align_options()
    end

    CM_TO_METRES = 0.01;   % see header for unit derivation

    log_info(opts, "Loading club target from %s [sheet=%s]", xlsx_path, sheet_name);

    % --- Parse the row-1 event-marker header (A, T, I, F, CHS) ---------
    events = local_read_event_header(xlsx_path, sheet_name);

    % --- Parse the sheet ----------------------------------------------------
    try
        raw_table = readmatrix(xlsx_path, ...
            "Sheet", sheet_name, ...
            "NumHeaderLines", 3);
    catch ME
        if contains(ME.message, "sheet", "IgnoreCase", true) || ...
           strcmp(ME.identifier, "MATLAB:spreadsheet:book:openSheet")
            error("load_club_target_excel:missingSheet", ...
                  "Sheet '%s' not found in %s (%s)", ...
                  sheet_name, xlsx_path, ME.message);
        end
        rethrow(ME);
    end

    if isempty(raw_table) || size(raw_table, 1) < 5
        error("load_club_target_excel:emptySheet", ...
              "Sheet '%s' has fewer than 5 data rows", sheet_name);
    end

    % Strip rows that are all-NaN (trailing blanks)
    valid_rows = any(~isnan(raw_table), 2);
    raw_table  = raw_table(valid_rows, :);

    % Need at least columns 1..26 (time + mid + club blocks)
    if size(raw_table, 2) < 26
        error("load_club_target_excel:truncatedColumns", ...
              "Sheet '%s' has %d columns, need >= 26", ...
              sheet_name, size(raw_table, 2));
    end

    M = size(raw_table, 1);
    sample_n = raw_table(:, 1);                                  % sample # (1..N)
    t_raw    = raw_table(:, 2);                                  % seconds
    % Cols 3-5 are the **mid-hands** position per the Definitions tab —
    % "midpoint of the hands at the center of the shaft".  We expose
    % them as `grip` (the canonical name) and as `butt` for backward
    % compatibility with older callers.
    grip_in  = raw_table(:, 3:5)  * CM_TO_METRES;                % cm -> m
    % Some recordings only export X and Y basis vectors (cols 12-14 NaN);
    % rebuild Z via cross(X, Y) so we still get a valid rotation matrix.
    raw_table(:, 6:14)  = local_repair_dcm_block(raw_table(:, 6:14));
    raw_table(:, 18:26) = local_repair_dcm_block(raw_table(:, 18:26));
    mid_R    = reshape_dcm(raw_table(:, 6:14));                  % 3x3xM
    head_in  = raw_table(:, 15:17) * CM_TO_METRES;               % cm -> m
    club_R   = reshape_dcm(raw_table(:, 18:26));                 % 3x3xM

    % --- Drop NaN-tainted rows (impact tracking can have dropouts).  Also
    %     drop rows where the direction-cosine matrices are partially NaN,
    %     since the quaternion conversion errors out on non-finite input.
    rot_finite_mid  = squeeze(all(all(isfinite(mid_R),  1), 2));
    rot_finite_club = squeeze(all(all(isfinite(club_R), 1), 2));
    keep = isfinite(t_raw) & all(isfinite(grip_in), 2) & ...
           all(isfinite(head_in), 2) & rot_finite_mid(:) & rot_finite_club(:);
    if nnz(keep) < 5
        error("load_club_target_excel:tooFewValidFrames", ...
              "Only %d non-NaN frames in sheet '%s'", nnz(keep), sheet_name);
    end
    t_raw   = t_raw(keep);
    grip_in = grip_in(keep, :);
    head_in = head_in(keep, :);
    mid_R   = mid_R(:, :, keep);     % grip orientation, used below
    club_R  = club_R(:, :, keep);
    M       = numel(t_raw);                                      %#ok<NASGU>

    % Time may start nonzero in source — shift so first sample sits at 0 for
    % the raw timeline (alignment helper will re-anchor to expected_impact_s).
    t_offset = t_raw(1);
    t_raw    = t_raw - t_offset;

    grip_quat = rotmat_to_quaternion(mid_R);
    club_quat = rotmat_to_quaternion(club_R);

    raw = struct( ...
        "time",      t_raw, ...
        "grip",      grip_in, ...
        "grip_quat", grip_quat, ...
        "butt",      grip_in, ...           % alias of `grip` for backward compat
        "clubhead",  head_in, ...
        "club_quat", club_quat);

    % Pass the documented impact sample's time through so the alignment
    % helper uses it instead of the speed-argmax heuristic.  Sample
    % numbers from the row-1 header refer to the original 1-indexed
    % timeline; convert to the post-NaN-filter shifted timeline by
    % nearest-neighbour lookup on the original `sample_n` column.
    if ~isempty(events) && isfield(events, "I_sample") && ~isnan(events.I_sample)
        sample_kept = sample_n(keep) - sample_n(1);
        [~, ix] = min(abs(sample_kept - (events.I_sample - sample_n(1))));
        opts.known_impact_s = t_raw(ix);
    end

    aligned = align_to_simulation_grid(raw, opts);

    % --- Provenance ---------------------------------------------------------
    [~, fname, fext] = fileparts(xlsx_path);
    source = struct( ...
        "filename",   string(strcat(fname, fext)), ...
        "format",     "xlsx", ...
        "subject_id", string(opts.subject_id), ...
        "trial_id",   coalesce_trial_id(opts.trial_id, sheet_name), ...
        "sha256",     string(sha256_of_file(xlsx_path)));

    target = struct( ...
        "time",       aligned.time, ...
        "grip",       aligned.grip, ...
        "grip_quat",  aligned.grip_quat, ...
        "butt",       aligned.butt, ...     % alias of grip for backward compat
        "clubhead",   aligned.clubhead, ...
        "club_quat",  aligned.club_quat, ...
        "impact_idx", aligned.impact_idx, ...
        "events",     events, ...
        "source",     source);

    % --- Postconditions: CLUB_IK_SPEC.md §"Validation rules" ---------------
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


function block = local_repair_dcm_block(block)
%LOCAL_REPAIR_DCM_BLOCK  Fill NaN basis vectors in the 9-col DCM block.
%   The Wiffle exports occasionally omit one of the three basis vectors —
%   we reconstruct any all-NaN basis as the cross product of the other two
%   (with a sign that preserves a right-handed frame).
    if size(block, 2) ~= 9 || size(block, 1) == 0
        return;
    end
    X = block(:, 1:3);  Y = block(:, 4:6);  Z = block(:, 7:9);
    bad_X = all(isnan(X), 2);  bad_Y = all(isnan(Y), 2);  bad_Z = all(isnan(Z), 2);
    if any(bad_Z)
        idx = bad_Z & ~bad_X & ~bad_Y;
        Z(idx, :) = cross(X(idx, :), Y(idx, :), 2);
    end
    if any(bad_X)
        idx = bad_X & ~bad_Y & ~bad_Z;
        X(idx, :) = cross(Y(idx, :), Z(idx, :), 2);
    end
    if any(bad_Y)
        idx = bad_Y & ~bad_X & ~bad_Z;
        Y(idx, :) = cross(Z(idx, :), X(idx, :), 2);
    end
    block(:, 1:3) = X;  block(:, 4:6) = Y;  block(:, 7:9) = Z;
end


function events = local_read_event_header(xlsx_path, sheet_name)
%LOCAL_READ_EVENT_HEADER  Parse the row-1 event-marker band of a Wiffle sheet.
%   Returns a struct with sample numbers for A (address), T (top of
%   backswing), I (impact), F (finish), and CHS_mph (club head speed at
%   impact in mph).  Any field that's missing in the sheet is NaN.
%
%   The sheet's row 1 is laid out as:
%     col1=trial-name col2=blank col3=A col4=<#> col5=T col6=<#>
%     col7=I col8=<#> col9=F col10=<#> col11=CHS col12=<mph>
    events = struct( ...
        'A_sample',  NaN, ...
        'T_sample',  NaN, ...
        'I_sample',  NaN, ...
        'F_sample',  NaN, ...
        'CHS_mph',   NaN);
    try
        row1 = readcell(xlsx_path, "Sheet", sheet_name, ...
                        "Range", "A1:Z1", "TextType", "string");
    catch
        return;
    end
    if isempty(row1); return; end
    % Walk the cells looking for label/value pairs.
    n = size(row1, 2);
    map = containers.Map( ...
        {'A','T','I','F','CHS'}, ...
        {'A_sample','T_sample','I_sample','F_sample','CHS_mph'});
    for c = 1:n - 1
        v = row1{1, c};
        if ismissing(v); continue; end
        label = strtrim(string(v));
        if strlength(label) == 0; continue; end
        if ~isKey(map, char(label)); continue; end
        val = row1{1, c + 1};
        if ismissing(val); continue; end
        if isnumeric(val) && isscalar(val)
            events.(map(char(label))) = double(val);
        elseif isstring(val) || ischar(val)
            num = str2double(val);
            if ~isnan(num); events.(map(char(label))) = num; end
        end
    end
end


function R = reshape_dcm(cols9)
    % cols9 is Mx9: [Xx Xy Xz Yx Yy Yz Zx Zy Zz]
    M = size(cols9, 1);
    R = zeros(3, 3, M);
    for k = 1:M
        % Each row is the basis vector expressed in world coords; assemble
        % the rotation matrix with X,Y,Z basis vectors as columns.
        Rk = [cols9(k, 1), cols9(k, 4), cols9(k, 7); ...
              cols9(k, 2), cols9(k, 5), cols9(k, 8); ...
              cols9(k, 3), cols9(k, 6), cols9(k, 9)];
        R(:, :, k) = Rk;
    end
end


function tid = coalesce_trial_id(trial_id, sheet_name)
    if strlength(string(trial_id)) > 0
        tid = string(trial_id);
    else
        tid = string(sheet_name);
    end
end


function log_info(opts, fmt, varargin)
    if ismember(string(opts.verbosity), ["Normal", "Verbose", "Debug"])
        fprintf("[load_club_target_excel] " + fmt + "\n", varargin{:});
    end
end
