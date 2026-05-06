function row = result_to_table_row(result, source_path)
%RESULT_TO_TABLE_ROW  Project a result struct into a leaderboard row.
%
%   ROW = RESULT_TO_TABLE_ROW(RESULT, SOURCE_PATH) returns a 1-row table
%   whose schema matches the leaderboard contract (issue #3992 +
%   VISUALIZATION_SPEC.md §"Comparison across options"):
%
%       swing_id  option  solver  rmse_mm  work_J  wall_s  commit  timestamp
%
%   Mapping rules:
%     swing_id   <- result.swing_id, else derived from SOURCE_PATH parent
%                   directory (e.g. ".../results/TW_ProV1/option1.mat" ->
%                   "TW_ProV1"), else "(unknown)".
%     option     <- result.option (numeric or "1".."4"), else inferred
%                   from SOURCE_PATH ("optionN" segment), else 0.
%     solver     <- result.solver (required by provenance check upstream).
%     rmse_mm    <- 1000 * result.final_rmse_m.
%     work_J     <- result.final_total_work_J.
%     wall_s     <- result.duration_s.
%     commit     <- short (7 char) prefix of result.git_commit.
%     timestamp  <- result.timestamp_utc.
%
%   Preconditions:
%     - RESULT has already passed verify_result_provenance, so all
%       required provenance fields exist.
%     - SOURCE_PATH is a 1x1 string. May or may not exist on disk by the
%       time this is called; only the lexical structure is consulted.
%
%   Postconditions:
%     - ROW is a 1x8 table with the documented schema and column types.
    arguments
        result      (1,1) struct
        source_path (1,1) string
    end

    swing_id  = local_swing_id(result, source_path);
    option    = local_option(result, source_path);
    solver    = string(result.solver);
    rmse_mm   = 1000 * double(result.final_rmse_m);
    work_J    = double(result.final_total_work_J);
    wall_s    = double(result.duration_s);
    commit    = local_short_commit(result.git_commit);
    timestamp = string(result.timestamp_utc);

    row = table( ...
        swing_id, option, solver, rmse_mm, work_J, wall_s, commit, timestamp, ...
        'VariableNames', ...
        {'swing_id','option','solver','rmse_mm','work_J','wall_s','commit','timestamp'});

    % Postcondition: schema matches the leaderboard contract.
    expected = {'swing_id','option','solver','rmse_mm','work_J','wall_s','commit','timestamp'};
    assert(isequal(row.Properties.VariableNames, expected), ...
        "result_to_table_row:postcondition", ...
        "Row schema does not match leaderboard contract.");
    assert(height(row) == 1, ...
        "result_to_table_row:postcondition", ...
        "Row must have exactly one entry.");
end

% --- helpers --------------------------------------------------------------

function s = local_swing_id(result, source_path)
    if isfield(result, "swing_id") && strlength(string(result.swing_id)) > 0
        s = string(result.swing_id);
        return;
    end
    [parent, ~, ~] = fileparts(char(source_path));
    [~, parent_name, ~] = fileparts(parent);
    if ~isempty(parent_name) && ~strcmpi(parent_name, "results")
        s = string(parent_name);
        return;
    end
    s = "(unknown)";
end

function n = local_option(result, source_path)
    if isfield(result, "option") && ~isempty(result.option)
        n = double(result.option);
        return;
    end
    tokens = regexp(char(source_path), 'option(\d+)', 'tokens', 'once');
    if ~isempty(tokens)
        n = str2double(tokens{1});
        return;
    end
    n = 0;
end

function short = local_short_commit(commit)
    s = string(commit);
    if strlength(s) >= 7
        short = extractBefore(s, 8);
    else
        short = s;
    end
end
