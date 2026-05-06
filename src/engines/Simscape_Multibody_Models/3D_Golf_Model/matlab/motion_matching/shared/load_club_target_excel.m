function target = load_club_target_excel(xlsx_path, sheet_name, opts)
%LOAD_CLUB_TARGET_EXCEL  Read club 6-DOF trajectory from a Wiffle-style xlsx file.
%
%   TARGET = LOAD_CLUB_TARGET_EXCEL(XLSX_PATH, SHEET_NAME, OPTS) returns a
%   target struct as specified in CLUB_IK_SPEC.md.
%
%   Implementation choice: native MATLAB parser (readmatrix) rather than
%   pyrunfile, because the Excel layout is simple (column ranges hard-coded
%   by mocap_data_loader.py) and avoiding a Python dependency makes the
%   loader runnable without configured `pyenv`.  The native parser mirrors
%   the column convention from
%   src/apps/golf_gui/Motion Capture Plotter/mocap_data_loader.py:
%     col 2  : time (seconds)
%     cols 3-5  : mid-hands position (inches)        -> butt position (m)
%     cols 6-14 : mid-hands rotation (direction cosines, X/Y/Z basis vectors
%                 stored row-major: Xx Xy Xz Yx Yy Yz Zx Zy Zz)
%     cols 15-17: club-head position (inches)        -> clubhead position (m)
%     cols 18-26: club-head rotation (direction cosines)
%
%   The club_quat field is the *club-head* rotation matrix converted to a
%   unit quaternion with q(1) >= 0.
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

    INCHES_TO_METRES = 0.0254;

    log_info(opts, "Loading club target from %s [sheet=%s]", xlsx_path, sheet_name);

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
    t_raw    = raw_table(:, 2);                                  % seconds
    butt_in  = raw_table(:, 3:5)  * INCHES_TO_METRES;            % metres
    mid_R    = reshape_dcm(raw_table(:, 6:14));                  % 3x3xM
    head_in  = raw_table(:, 15:17) * INCHES_TO_METRES;           % metres
    club_R   = reshape_dcm(raw_table(:, 18:26));                 % 3x3xM

    % --- Drop NaN-tainted rows (impact tracking can have dropouts) ---------
    keep = isfinite(t_raw) & all(isfinite(butt_in), 2) & ...
           all(isfinite(head_in), 2);
    if nnz(keep) < 5
        error("load_club_target_excel:tooFewValidFrames", ...
              "Only %d non-NaN frames in sheet '%s'", nnz(keep), sheet_name);
    end
    t_raw   = t_raw(keep);
    butt_in = butt_in(keep, :);
    head_in = head_in(keep, :);
    club_R  = club_R(:, :, keep);
    M       = numel(t_raw);                                      %#ok<NASGU>

    % Time may start nonzero in source — shift so first sample sits at 0 for
    % the raw timeline (alignment helper will re-anchor to expected_impact_s).
    t_raw = t_raw - t_raw(1);

    club_quat = rotmat_to_quaternion(club_R);

    raw = struct( ...
        "time",      t_raw, ...
        "butt",      butt_in, ...
        "clubhead",  head_in, ...
        "club_quat", club_quat);

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
        "butt",       aligned.butt, ...
        "clubhead",   aligned.clubhead, ...
        "club_quat",  aligned.club_quat, ...
        "impact_idx", aligned.impact_idx, ...
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
