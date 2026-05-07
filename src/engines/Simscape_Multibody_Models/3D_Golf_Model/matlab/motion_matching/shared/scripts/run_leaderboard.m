function summary = run_leaderboard(varargin)
%RUN_LEADERBOARD  Cross-option motion-matching fit comparison driver (#4080).
%
%   SUMMARY = RUN_LEADERBOARD() runs every available motion-matching option
%   on every canonical test trial, saves per-pair result .mat files under
%   motion_matching/results/<trial_id>_<option>.mat, regenerates
%   motion_matching/results/LEADERBOARD.md, and returns a summary struct.
%
%   SUMMARY = RUN_LEADERBOARD(NAME, VALUE, ...) accepts:
%
%     'Trials'     string array of trial sheet names to evaluate.
%                  Default = ["TW_ProV1"]; expand as more land.
%     'Options'    string array selecting which optimizers to run.
%                  Default = ["fmincon", "surrogate", "inverse", "hybrid"].
%                  Each entry must be one of the listed values; unknown
%                  values raise run_leaderboard:unknownOption.
%     'XlsxPath'   path to the canonical Wiffle xlsx. Default resolves
%                  relative to the engine root via local_default_xlsx().
%     'ResultsDir' destination for per-pair .mat files and LEADERBOARD.md.
%                  Default = motion_matching/results next to this script.
%     'DryRun'     logical (default false). When true, the driver does not
%                  actually invoke any optimizer; instead it stamps a
%                  placeholder row marking each cell as "not run". Used by
%                  tests that cannot stand up the Simscape model.
%     'SkipMissing' logical (default true). When true, an option whose
%                  optimizer entry point is not on path is logged and
%                  marked "not yet available" rather than aborting the run.
%     'Verbose'    logical (default true).
%     'GitCommit'  optional commit SHA to embed in LEADERBOARD.md. Default
%                  is read from `git rev-parse HEAD` if available, else "".
%
%   Returns SUMMARY with fields:
%     .results_dir    string  - absolute path to results directory
%     .leaderboard_md string  - absolute path to the regenerated MD file
%     .rows           struct array, one entry per (trial, option) cell
%     .skipped        cell array of {trial, option, reason} triplets
%
%   Preconditions (DbC):
%     - Every entry in Options is one of {"fmincon","surrogate","inverse",
%       "hybrid"}; an unknown value triggers run_leaderboard:unknownOption.
%     - Trials is a non-empty string array.
%
%   Postconditions:
%     - SUMMARY.leaderboard_md points at an existing .md file.
%     - The MD file contains a header row with columns swing_id, option,
%       solver, rmse_mm, work_J, wall_s, commit, timestamp.
%
%   Robust to missing optimizers: if Option 2 (surrogate) or Option 3
%   (inverse) entry points are not yet on path, those columns render as
%   "not yet available" rather than crashing the run.
%
%   See also: FIT_SWING_FULL_PIPELINE, LEADERBOARD,
%             VISUALIZATION_SPEC.md.

    p = inputParser();
    p.addParameter('Trials',      "TW_ProV1", @(x) isstring(x) || ischar(x) || iscell(x));
    p.addParameter('Options',     ["fmincon", "surrogate", "inverse", "hybrid"], ...
                                  @(x) isstring(x) || ischar(x) || iscell(x));
    p.addParameter('XlsxPath',    "", @(x) ischar(x) || isstring(x));
    p.addParameter('ResultsDir',  "", @(x) ischar(x) || isstring(x));
    p.addParameter('DryRun',      false, @(x) islogical(x) && isscalar(x));
    p.addParameter('SkipMissing', true,  @(x) islogical(x) && isscalar(x));
    p.addParameter('Verbose',     true,  @(x) islogical(x) && isscalar(x));
    p.addParameter('GitCommit',   "",    @(x) ischar(x) || isstring(x));
    p.parse(varargin{:});
    args = p.Results;

    trials  = local_to_string_array(args.Trials);
    options = local_to_string_array(args.Options);
    if isempty(trials)
        error("run_leaderboard:noTrials", ...
              "Precondition: at least one trial must be supplied.");
    end
    local_validate_options(options);

    results_dir = local_resolve_results_dir(args.ResultsDir);
    if ~isfolder(results_dir); mkdir(results_dir); end

    xlsx_path = local_resolve_xlsx(args.XlsxPath);
    git_commit = local_resolve_commit(args.GitCommit);

    rows    = struct([]);
    skipped = cell(0, 3);

    for i = 1:numel(trials)
        trial_id = trials(i);
        for j = 1:numel(options)
            opt_name = options(j);
            [row, skip_reason] = local_run_one(trial_id, opt_name, ...
                xlsx_path, results_dir, args.DryRun, ...
                args.SkipMissing, args.Verbose, git_commit);
            if isempty(skip_reason)
                rows = local_append_row(rows, row);
            else
                skipped(end + 1, :) = {char(trial_id), char(opt_name), char(skip_reason)}; %#ok<AGROW>
                rows = local_append_row(rows, ...
                    local_pending_row(trial_id, opt_name, skip_reason, git_commit));
            end
        end
    end

    md_path = fullfile(results_dir, "LEADERBOARD.md");
    local_write_markdown(md_path, rows, trials, options);

    summary = struct();
    summary.results_dir    = string(results_dir);
    summary.leaderboard_md = string(md_path);
    summary.rows           = rows;
    summary.skipped        = skipped;

    % Postconditions ----------------------------------------------------
    assert(isfile(char(md_path)), ...
        "run_leaderboard:postMd", ...
        "Postcondition: LEADERBOARD.md must exist at %s", md_path);
end


%% =====================================================================
function local_validate_options(options)
    allowed = ["fmincon", "surrogate", "inverse", "hybrid"];
    bad = setdiff(options, allowed);
    if ~isempty(bad)
        error("run_leaderboard:unknownOption", ...
              "Unknown option(s): %s. Allowed: %s", ...
              strjoin(bad, ", "), strjoin(allowed, ", "));
    end
end


%% =====================================================================
function s = local_to_string_array(v)
    if iscell(v)
        s = string(v);
    else
        s = string(v);
    end
    s = s(:).';
end


%% =====================================================================
function p = local_resolve_results_dir(arg)
    if strlength(string(arg)) > 0
        p = char(arg);
        return;
    end
    here = fileparts(mfilename("fullpath"));
    % shared/scripts -> shared -> motion_matching
    mm_root = fileparts(fileparts(here));
    p = fullfile(mm_root, "results");
end


%% =====================================================================
function p = local_resolve_xlsx(arg)
    if strlength(string(arg)) > 0
        p = char(arg);
        return;
    end
    here = fileparts(mfilename("fullpath"));
    % shared/scripts -> shared -> motion_matching -> matlab -> 3D_Golf_Model
    engine_root = fileparts(fileparts(fileparts(fileparts(here))));
    candidate = fullfile(engine_root, "matlab", "src", "apps", "golf_gui", ...
                         "Motion Capture Plotter", ...
                         "Wiffle_ProV1_club_3D_data.xlsx");
    if isfile(candidate)
        p = char(candidate);
    else
        % Fall back to the bare filename — MATLAB's path search may pick
        % it up; when it doesn't, the per-cell run is marked skipped.
        p = "Wiffle_ProV1_club_3D_data.xlsx";
    end
end


%% =====================================================================
function commit = local_resolve_commit(arg)
    if strlength(string(arg)) > 0
        commit = char(arg);
        return;
    end
    commit = "";
    try
        [status, out] = system("git rev-parse HEAD");
        if status == 0
            commit = strtrim(out);
            if numel(commit) > 12
                commit = commit(1:12);
            end
        end
    catch
        % swallow — commit stays ""
    end
end


%% =====================================================================
function [row, skip_reason] = local_run_one(trial_id, opt_name, ...
        xlsx_path, results_dir, dry_run, skip_missing, verbose, commit)
%LOCAL_RUN_ONE  Run one (trial, option) pair, returning its leaderboard row.

    skip_reason = "";
    row = struct();

    solver_fn_name = local_solver_fn_for(opt_name);
    if ~dry_run && skip_missing && ~local_is_on_path(solver_fn_name)
        skip_reason = sprintf("optimizer %s not on path", solver_fn_name);
        if verbose
            fprintf("[run_leaderboard] %s/%s -> SKIP (%s)\n", ...
                trial_id, opt_name, skip_reason);
        end
        return;
    end

    if dry_run
        if verbose
            fprintf("[run_leaderboard] %s/%s -> DRY-RUN\n", trial_id, opt_name);
        end
        skip_reason = "dry-run";
        return;
    end

    if ~isfile(xlsx_path)
        skip_reason = sprintf("xlsx not found: %s", xlsx_path);
        if verbose
            fprintf("[run_leaderboard] %s/%s -> SKIP (%s)\n", ...
                trial_id, opt_name, skip_reason);
        end
        return;
    end

    save_dir = fullfile(results_dir, sprintf("%s_%s", trial_id, opt_name));
    if ~isfolder(save_dir); mkdir(save_dir); end

    fit_opts = struct();
    fit_opts.sheet    = char(trial_id);
    fit_opts.option   = char(local_pipeline_option_for(opt_name));
    fit_opts.save_dir = char(save_dir);
    fit_opts.verbose  = verbose;
    fit_opts.render_figures = false;

    t0 = tic;
    try
        result = fit_swing_full_pipeline(xlsx_path, fit_opts);
        wall_s = toc(t0);
    catch ME
        skip_reason = sprintf("fit raised: %s", ME.identifier);
        if verbose
            fprintf("[run_leaderboard] %s/%s -> FAIL (%s: %s)\n", ...
                trial_id, opt_name, ME.identifier, ME.message);
        end
        return;
    end

    % Persist per-pair result .mat next to LEADERBOARD.md.
    pair_mat = fullfile(results_dir, sprintf("%s_%s.mat", trial_id, opt_name));
    try
        result_to_save = result; %#ok<NASGU>
        save(char(pair_mat), "result_to_save", "-v7");
    catch ME
        if verbose
            fprintf("[run_leaderboard] WARNING: could not save %s: %s\n", ...
                pair_mat, ME.message);
        end
    end

    row = struct();
    row.swing_id  = string(trial_id);
    row.option    = string(opt_name);
    row.solver    = string(local_get(result, "solver", opt_name));
    row.rmse_mm   = 1000 * double(local_get(result, "final_rmse_m", NaN));
    row.work_J    = double(local_get(result, "final_total_work_J", NaN));
    row.wall_s    = wall_s;
    row.commit    = string(commit);
    row.timestamp = string(datetime("now", "TimeZone", "UTC", ...
                                    "Format", "yyyy-MM-dd'T'HH:mm:ss'Z'"));
    row.status    = "ok";
end


%% =====================================================================
function row = local_pending_row(trial_id, opt_name, reason, commit)
    row = struct();
    row.swing_id  = string(trial_id);
    row.option    = string(opt_name);
    row.solver    = "n/a";
    row.rmse_mm   = NaN;
    row.work_J    = NaN;
    row.wall_s    = NaN;
    row.commit    = string(commit);
    row.timestamp = string(datetime("now", "TimeZone", "UTC", ...
                                    "Format", "yyyy-MM-dd'T'HH:mm:ss'Z'"));
    row.status    = string(reason);
end


%% =====================================================================
function rows = local_append_row(rows, row)
    if isempty(rows)
        rows = row;
    else
        rows(end + 1) = row;
    end
end


%% =====================================================================
function tf = local_is_on_path(fn_name)
%LOCAL_IS_ON_PATH  True iff a function with this name is resolvable.
    tf = exist(char(fn_name), "file") == 2 || ...
         exist(char(fn_name), "builtin") == 5;
end


%% =====================================================================
function fn = local_solver_fn_for(opt_name)
%LOCAL_SOLVER_FN_FOR  Map option label -> expected MATLAB function name.
    switch lower(string(opt_name))
        case "fmincon",   fn = "fit_swing_fmincon";
        case "surrogate", fn = "fit_swing_surrogateopt";
        case "inverse",   fn = "fit_swing_inverse";
        case "hybrid",    fn = "fit_swing_hybrid";
        otherwise
            error("run_leaderboard:unknownOption", ...
                  "Unknown option label: %s", opt_name);
    end
end


%% =====================================================================
function s = local_pipeline_option_for(opt_name)
%LOCAL_PIPELINE_OPTION_FOR  Map driver label -> fit_swing_full_pipeline opt.
    switch lower(string(opt_name))
        case "fmincon",   s = "fmincon";
        case "surrogate", s = "surrogateopt";
        case "inverse"
            % Option 3 (#4076) doesn't yet wire through the pipeline; the
            % skip-missing path catches this when fit_swing_inverse isn't
            % present, but if a future PR registers it as a dispatch we
            % map it here.
            s = "inverse";
        case "hybrid",    s = "hybrid";
        otherwise
            s = lower(string(opt_name));
    end
end


%% =====================================================================
function v = local_get(s, name, default)
    if isstruct(s) && isfield(s, name)
        v = s.(name);
    else
        v = default;
    end
end


%% =====================================================================
function local_write_markdown(md_path, rows, trials, options)
%LOCAL_WRITE_MARKDOWN  Emit motion_matching/results/LEADERBOARD.md.

    fid = fopen(char(md_path), "w");
    if fid < 0
        error("run_leaderboard:mdOpen", ...
              "Could not open %s for writing", md_path);
    end
    closer = onCleanup(@() fclose(fid));

    fprintf(fid, "# Motion-matching leaderboard\n\n");
    fprintf(fid, "_Auto-generated by `scripts/run_leaderboard.m` (#4080)._\n");
    fprintf(fid, "_Regenerated whenever a PR touches `motion_matching/`._\n\n");

    fprintf(fid, "## Per-pair fit results\n\n");
    fprintf(fid, "| swing_id | option | solver | rmse_mm | work_J | wall_s | commit | timestamp | status |\n");
    fprintf(fid, "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for k = 1:numel(rows)
        r = rows(k);
        fprintf(fid, "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", ...
            r.swing_id, ...
            r.option, ...
            r.solver, ...
            local_fmt_num(r.rmse_mm), ...
            local_fmt_num(r.work_J), ...
            local_fmt_num(r.wall_s), ...
            local_or_dash(r.commit), ...
            local_or_dash(r.timestamp), ...
            r.status);
    end
    fprintf(fid, "\n");

    fprintf(fid, "## Cross-option grid (rmse_mm)\n\n");
    fprintf(fid, "| trial \\ option |");
    for j = 1:numel(options)
        fprintf(fid, " %s |", options(j));
    end
    fprintf(fid, "\n| --- |");
    for j = 1:numel(options)
        fprintf(fid, " --- |"); %#ok<*NOPRT>
    end
    fprintf(fid, "\n");
    for i = 1:numel(trials)
        fprintf(fid, "| %s |", trials(i));
        for j = 1:numel(options)
            cell_str = local_grid_cell(rows, trials(i), options(j));
            fprintf(fid, " %s |", cell_str);
        end
        fprintf(fid, "\n");
    end
    fprintf(fid, "\n");

    fprintf(fid, "## Notes\n\n");
    fprintf(fid, "- `not yet available` cells indicate the optimizer entry point ");
    fprintf(fid, "is not yet on the MATLAB path; see the linked option PRs.\n");
    fprintf(fid, "- Per-pair `.mat` files are written next to this report as ");
    fprintf(fid, "`<trial_id>_<option>.mat`.\n");
    fprintf(fid, "- The grip-RMSE acceptance gate (#4080 AC3) is ");
    fprintf(fid, "`max(rmse_mm) - min(rmse_mm) <= 5 mm`; document divergence ");
    fprintf(fid, "in this file when the gate trips.\n");
end


%% =====================================================================
function s = local_grid_cell(rows, trial_id, opt_name)
%LOCAL_GRID_CELL  Pretty cell text for the cross-option grid.
    for k = 1:numel(rows)
        r = rows(k);
        if r.swing_id == trial_id && r.option == opt_name
            if r.status == "ok"
                s = local_fmt_num(r.rmse_mm);
            else
                s = sprintf("_%s_", local_short_status(r.status));
            end
            return;
        end
    end
    s = "_pending_";
end


%% =====================================================================
function s = local_short_status(status)
%LOCAL_SHORT_STATUS  Compress long status strings for grid cells.
    s = string(status);
    if startsWith(s, "optimizer ") && contains(s, "not on path")
        s = "not yet available";
    elseif startsWith(s, "xlsx not found")
        s = "xlsx missing";
    elseif startsWith(s, "fit raised")
        s = "fit error";
    elseif s == "dry-run"
        s = "not run";
    end
end


%% =====================================================================
function s = local_fmt_num(x)
    if isnumeric(x) && isscalar(x) && isfinite(x)
        s = sprintf("%.3f", double(x));
    else
        s = "n/a";
    end
end


%% =====================================================================
function s = local_or_dash(v)
    s = string(v);
    if strlength(s) == 0
        s = "-";
    end
end
