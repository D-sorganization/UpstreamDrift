function [tbl, fig_or_str] = leaderboard(results_dir, opts)
%LEADERBOARD  Cross-option comparison table for motion_matching results.
%
%   TBL = LEADERBOARD() scans the default
%   motion_matching/results/ directory for *.mat files containing per-fit
%   result structs (CODING_STANDARDS.md §"Provenance and reproducibility")
%   and returns a sorted MATLAB table with columns:
%
%       swing_id  option  solver  rmse_mm  work_J  wall_s  commit  timestamp
%
%   Sort order is ascending by OPTS.sort_by (default "rmse_mm"), per
%   VISUALIZATION_SPEC.md §"Comparison across options".
%
%   TBL = LEADERBOARD(RESULTS_DIR, OPTS) overrides the directory and
%   options. Use DEFAULT_LEADERBOARD_OPTIONS() for a starter struct.
%
%   [TBL, FIG_OR_STR] = LEADERBOARD(...) additionally returns:
%     - a figure handle (uitable + grouped bar chart) when
%       OPTS.build_figure is true or when OPTS.format == "table" (default);
%     - the CSV string when OPTS.format == "csv";
%     - the markdown pipe-table string when OPTS.format == "markdown".
%
%   Behaviour notes:
%     - When RESULTS_DIR has no .mat files (or does not exist), an empty
%       table with the documented schema is returned, no error.
%     - Result structs missing provenance fields are skipped with a
%       warning ("leaderboard:missingProvenance"); the scan continues.
%     - When OPTS.recheck is true, the saved coefficients are re-fed
%       through compute_cost and rmse_mm verified to within 1e-6.
%     - When OPTS.write_csv is true the table is written to OPTS.csv_path
%       via writetable, regardless of OPTS.format.
%
%   Preconditions:
%     - RESULTS_DIR is a 1x1 string. Existence is preferred but not
%       required (an empty schema-correct table is returned otherwise).
%     - OPTS is a 1x1 struct merged with DEFAULT_LEADERBOARD_OPTIONS().
%
%   Postconditions:
%     - TBL is a MATLAB table whose VariableNames are exactly:
%         {'swing_id','option','solver','rmse_mm','work_J','wall_s', ...
%          'commit','timestamp'}.
%     - When height(TBL) > 1, TBL is sorted ascending on OPTS.sort_by.
%     - When OPTS.filter_swing_id is non-empty, every row's swing_id
%       equals it.
%
%   GitHub issue: #3992.
    arguments
        results_dir (1,1) string = local_default_results_dir()
        opts        (1,1) struct = default_leaderboard_options()
    end

    opts = local_merge_defaults(opts);

    % --- discover and load --------------------------------------------------
    files = scan_results_directory(results_dir);
    rows  = cell(numel(files), 1);
    keep  = false(numel(files), 1);

    for k = 1:numel(files)
        try
            payload = load(files(k));
        catch err
            warning("leaderboard:loadFailed", ...
                "Could not load %s: %s", files(k), err.message);
            continue;
        end
        result = local_extract_result(payload);
        if isempty(result)
            warning("leaderboard:noResultStruct", ...
                "%s does not contain a 'result' struct; skipping.", files(k));
            continue;
        end
        [ok, missing] = verify_result_provenance(result);
        if ~ok
            warning("leaderboard:missingProvenance", ...
                "%s missing provenance fields: %s; skipping.", ...
                files(k), strjoin(missing, ", "));
            continue;
        end
        if opts.recheck
            local_recheck_rmse(result, files(k));
        end
        rows{k} = result_to_table_row(result, files(k));
        keep(k) = true;
    end

    % --- assemble table -----------------------------------------------------
    tbl = local_assemble_table(rows(keep));

    % --- filter -------------------------------------------------------------
    if strlength(opts.filter_swing_id) > 0 && height(tbl) > 0
        tbl = tbl(tbl.swing_id == opts.filter_swing_id, :);
    end

    % --- sort ---------------------------------------------------------------
    if height(tbl) > 1
        sort_col = char(opts.sort_by);
        assert(any(strcmp(sort_col, tbl.Properties.VariableNames)), ...
            "leaderboard:badSortBy", ...
            "opts.sort_by '%s' is not a table column", sort_col);
        tbl = sortrows(tbl, sort_col, 'ascend');
    end

    % --- side-effect: csv ---------------------------------------------------
    if opts.write_csv && height(tbl) > 0
        writetable(tbl, char(opts.csv_path));
    end

    % --- second return ------------------------------------------------------
    if nargout >= 2
        fig_or_str = local_secondary_output(tbl, opts);
    end

    % --- postconditions -----------------------------------------------------
    expected_cols = {'swing_id','option','solver','rmse_mm', ...
                     'work_J','wall_s','commit','timestamp'};
    assert(isequal(tbl.Properties.VariableNames, expected_cols), ...
        "leaderboard:postcondition", "Output schema mismatch.");
    if strlength(opts.filter_swing_id) > 0 && height(tbl) > 0
        assert(all(tbl.swing_id == opts.filter_swing_id), ...
            "leaderboard:postcondition", "Filter postcondition violated.");
    end
end

% --- helpers --------------------------------------------------------------

function d = local_default_results_dir()
    here = fileparts(mfilename('fullpath'));      % .../shared
    parent = fileparts(here);                      % .../motion_matching
    d = string(fullfile(parent, "results"));
end

function opts = local_merge_defaults(opts)
%LOCAL_MERGE_DEFAULTS  Patch user opts with default fields they omitted.
    defaults = default_leaderboard_options();
    fns = string(fieldnames(defaults));
    for k = 1:numel(fns)
        if ~isfield(opts, fns(k))
            opts.(fns(k)) = defaults.(fns(k));
        end
    end
end

function result = local_extract_result(payload)
%LOCAL_EXTRACT_RESULT  Pull a result struct out of a loaded .mat payload.
    result = [];
    if isfield(payload, 'result') && isstruct(payload.result) && isscalar(payload.result)
        result = payload.result;
        return;
    end
    % Fall back: any single struct field that looks like a result.
    fns = fieldnames(payload);
    for k = 1:numel(fns)
        v = payload.(fns{k});
        if isstruct(v) && isscalar(v) && isfield(v, 'final_rmse_m')
            result = v;
            return;
        end
    end
end

function local_recheck_rmse(result, source_path)
%LOCAL_RECHECK_RMSE  Optional re-run of compute_cost (DbC postcondition).
    if ~isfield(result, 'coefficients') || ~isfield(result, 'target')
        warning("leaderboard:recheckSkipped", ...
            "Cannot recheck %s: missing coefficients or target.", source_path);
        return;
    end
    try
        cost = compute_cost(result.coefficients, result.target);
        if abs(cost - result.final_rmse_m) > 1e-6
            warning("leaderboard:recheckMismatch", ...
                "%s: stored final_rmse_m=%.9g != recomputed=%.9g", ...
                source_path, result.final_rmse_m, cost);
        end
    catch err
        warning("leaderboard:recheckFailed", ...
            "Recheck failed for %s: %s", source_path, err.message);
    end
end

function tbl = local_assemble_table(row_cells)
%LOCAL_ASSEMBLE_TABLE  Vertically concat row cells, or build empty schema.
    if isempty(row_cells)
        tbl = local_empty_table();
        return;
    end
    tbl = vertcat(row_cells{:});
end

function tbl = local_empty_table()
%LOCAL_EMPTY_TABLE  Construct the schema-correct empty leaderboard table.
    tbl = table( ...
        strings(0,1), zeros(0,1), strings(0,1), zeros(0,1), ...
        zeros(0,1), zeros(0,1), strings(0,1), strings(0,1), ...
        'VariableNames', ...
        {'swing_id','option','solver','rmse_mm','work_J','wall_s','commit','timestamp'});
end

function out = local_secondary_output(tbl, opts)
%LOCAL_SECONDARY_OUTPUT  Build the requested second return value.
    fmt = lower(string(opts.format));
    switch fmt
        case "csv"
            out = local_table_to_csv_string(tbl);
        case "markdown"
            out = local_table_to_markdown(tbl);
        case "table"
            if opts.build_figure
                out = local_build_figure(tbl);
            else
                out = tbl;  % echo of primary
            end
        otherwise
            error("leaderboard:badFormat", ...
                "opts.format must be 'table'|'csv'|'markdown', got '%s'", fmt);
    end
end

function s = local_table_to_csv_string(tbl)
    tmp = [tempname, '.csv'];
    cleaner = onCleanup(@() local_safe_delete(tmp));  %#ok<NASGU>
    writetable(tbl, tmp);
    s = string(fileread(tmp));
end

function local_safe_delete(p)
    if isfile(p)
        delete(p);
    end
end

function s = local_table_to_markdown(tbl)
    cols = string(tbl.Properties.VariableNames);
    header = "| " + strjoin(cols, " | ") + " |";
    sep    = "| " + strjoin(repmat("---", 1, numel(cols)), " | ") + " |";
    lines  = [header; sep];
    for r = 1:height(tbl)
        cells = strings(1, numel(cols));
        for c = 1:numel(cols)
            v = tbl{r, c};
            if iscell(v); v = v{1}; end
            cells(c) = local_md_cell(v);
        end
        lines(end+1, 1) = "| " + strjoin(cells, " | ") + " |"; %#ok<AGROW>
    end
    s = strjoin(lines, newline);
end

function s = local_md_cell(v)
    if isstring(v) || ischar(v)
        s = string(v);
    elseif isnumeric(v) && isscalar(v)
        if v == floor(v)
            s = string(sprintf("%d", v));
        else
            s = string(sprintf("%.3f", v));
        end
    else
        s = string(v);
    end
end

function fig = local_build_figure(tbl)
%LOCAL_BUILD_FIGURE  Comparison figure: uitable on top, bar chart below.
    fig = figure('Visible', 'off', 'Color', 'w', ...
                 'Name', 'Leaderboard (cross-option comparison)', ...
                 'NumberTitle', 'off');
    if height(tbl) == 0
        annotation(fig, 'textbox', [0.1 0.45 0.8 0.1], ...
            'String', 'No results to display.', ...
            'EdgeColor', 'none', 'HorizontalAlignment', 'center');
        return;
    end
    cell_data = local_table_to_cell(tbl);
    uitable(fig, ...
        'Data', cell_data, ...
        'ColumnName', tbl.Properties.VariableNames, ...
        'Units', 'normalized', ...
        'Position', [0.02 0.55 0.96 0.43], ...
        'Tag', 'leaderboard_uitable');
    ax = axes(fig, 'Units', 'normalized', 'Position', [0.1 0.08 0.85 0.4], ...
        'Tag', 'leaderboard_bar_axes');
    [groups, ~, gidx] = unique(tbl.swing_id);
    rmse = tbl.rmse_mm;
    bar(ax, gidx, rmse);
    ax.XTick = 1:numel(groups);
    ax.XTickLabel = cellstr(groups);
    ylabel(ax, 'RMSE (mm)');
    title(ax, 'Clubhead RMSE by swing');
end

function c = local_table_to_cell(tbl)
    n = height(tbl);
    m = width(tbl);
    c = cell(n, m);
    for r = 1:n
        for k = 1:m
            v = tbl{r, k};
            if iscell(v); v = v{1}; end
            if isstring(v); v = char(v); end
            c{r, k} = v;
        end
    end
end
