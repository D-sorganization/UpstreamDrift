function result = fit_swing_full_pipeline(target_or_xlsx, opts)
%FIT_SWING_FULL_PIPELINE  One-call grip-primary swing fit (Stage-1 + Stage-2).
%
%   RESULT = FIT_SWING_FULL_PIPELINE(TARGET_OR_XLSX, OPTS) codifies the
%   GRIP_FIT_PLAYBOOK.md §6.4 copy-paste recipe as a reusable function:
%
%     1. Resolve the target (load Wiffle xlsx if a path is given, else
%        accept a pre-loaded canonical target struct).
%     2. Run SOLVE_STARTING_POSE (Stage-1) to compute the small set of
%        ``*StartPosition*`` / ``*StartVelocity*`` overrides that put the
%        body at the address pose.
%     3. Inject those overrides into ``opts.stage2_opts.sim.input_overrides``.
%     4. Dispatch the requested optimizer
%        (FIT_SWING_FMINCON / FIT_SWING_MULTISTART / FIT_SWING_SURROGATEOPT
%        / FIT_SWING_HYBRID).
%     5. Render the canonical figures
%        (PLOT_TRAJECTORY_OVERLAY, PLOT_ERROR_TIMECOURSE,
%        PLOT_FIT_QUALITY_CARD) with `Visible='off'` and save them as PNGs.
%     6. Optionally render the swing animation.
%     7. Save the result struct + a plain-text human-readable report.
%     8. Return the augmented RESULT struct.
%
%   Inputs
%   ------
%   TARGET_OR_XLSX
%       Either a string/char path to a Wiffle xlsx file (the sheet name
%       comes from OPTS.sheet, default "TW_ProV1") OR a pre-loaded
%       canonical target struct (must satisfy CLUB_IK_SPEC.md — fields
%       time, grip, grip_quat, butt, clubhead, club_quat, impact_idx,
%       events, source).
%
%   OPTS  (1,1) struct.  Optional fields with defaults:
%     .sheet            "TW_ProV1"
%     .option           "fmincon" (default) | "multistart" | "surrogate"
%                       | "surrogateopt" | "hybrid"
%     .save_dir         default = output/fits/<yyyyMMdd_HHmmss>
%     .render_figures   default true
%     .save_animation   default false
%     .input_mat        default 3DModelInputs_Impact.mat (path resolved
%                       relative to the engine root if found, else used
%                       as-is)
%     .stage1_opts      passed straight to solve_starting_pose
%     .stage2_opts      seeded from default_option1_options(), then
%                       deep-merged with caller overrides; the Stage-1
%                       overrides are written into
%                       opts.stage2_opts.sim.input_overrides before
%                       dispatch.
%     .viz_opts         passed to the figure renderers (defaults from
%                       default_viz_options() with .visible="off").
%     .skip_stage1      logical; if true, Stage-1 is skipped and the
%                       returned overrides struct is empty.  Useful for
%                       smoke tests.
%     .verbose          default true
%
%   Output
%   ------
%   RESULT
%       The canonical fit-result struct returned by the dispatched
%       optimizer (see option1_direct_optimization/INTERFACES.md "Result
%       struct contract"), augmented with:
%         .stage1_overrides   struct of *StartPosition* / *StartVelocity*
%                             overrides from solve_starting_pose
%         .figure_paths       cell array of saved figure paths (PNG)
%         .report_path        full path to a plain-text summary report
%         .save_dir           save directory used
%         .target             the resolved canonical target (also nested
%                             inside the inner solver result)
%
%   Preconditions (DbC)
%   -------------------
%     * TARGET_OR_XLSX is a string/char path that exists, OR a 1x1 struct
%       with the required canonical fields.
%     * OPTS, when supplied, is a 1x1 struct.
%     * OPTS.option (when set) names a known optimizer dispatch.
%
%   Postconditions
%   --------------
%     * RESULT contains the fields .stage1_overrides, .figure_paths,
%       .report_path, .save_dir; figure_paths is a cellstr (possibly
%       empty when render_figures=false), and report_path is a non-empty
%       string pointing at an existing file.
%     * RESULT.solver matches the dispatched optimizer name.
%     * The save directory exists.
%
%   See also: SOLVE_STARTING_POSE, LOAD_CLUB_TARGET_EXCEL,
%             FIT_SWING_FMINCON, FIT_SWING_MULTISTART, FIT_SWING_SURROGATEOPT,
%             FIT_SWING_HYBRID, PLOT_TRAJECTORY_OVERLAY,
%             PLOT_ERROR_TIMECOURSE, PLOT_FIT_QUALITY_CARD,
%             ANIMATE_TRAJECTORY_OVERLAY, GRIP_FIT_PLAYBOOK.

    arguments
        target_or_xlsx
        opts (1,1) struct = struct()
    end

    % ---- 1. Fill option defaults --------------------------------------
    opts = local_fill_defaults(opts);

    if opts.verbose
        fprintf("[fit_swing_full_pipeline] option=%s sheet=%s save_dir=%s\n", ...
                opts.option, opts.sheet, opts.save_dir);
    end

    pipeline_t0 = tic;

    % ---- 2. Resolve target --------------------------------------------
    target = local_resolve_target(target_or_xlsx, opts);

    % ---- 3. Stage-1: starting pose ------------------------------------
    if opts.skip_stage1
        if opts.verbose
            fprintf("[fit_swing_full_pipeline] Stage-1 skipped (opts.skip_stage1=true)\n");
        end
        stage1_overrides = struct();
        stage1_duration_s = 0.0;
    else
        stage1_t0 = tic;
        stage1_overrides = solve_starting_pose(target, ...
                                               string(opts.input_mat), ...
                                               opts.stage1_opts);
        stage1_duration_s = toc(stage1_t0);
        if opts.verbose
            fprintf("[fit_swing_full_pipeline] Stage-1 done in %.2fs\n", ...
                    stage1_duration_s);
        end
    end

    % ---- 4. Inject overrides into Stage-2 sim opts --------------------
    stage2_opts = opts.stage2_opts;
    if ~isfield(stage2_opts, "sim") || ~isstruct(stage2_opts.sim)
        stage2_opts.sim = default_sim_options();
    end
    stage2_opts.sim.input_overrides = local_merge_overrides( ...
        local_get_field(stage2_opts.sim, "input_overrides", struct()), ...
        stage1_overrides);

    % ---- 5. Stage-2: dispatch the optimizer ---------------------------
    [solver_fn, solver_name] = local_dispatch(opts.option);
    if opts.verbose
        fprintf("[fit_swing_full_pipeline] dispatching Stage-2 -> %s\n", ...
                solver_name);
    end
    stage2_t0 = tic;
    inner_result = solver_fn(target, stage2_opts);
    stage2_duration_s = toc(stage2_t0);
    if opts.verbose
        fprintf("[fit_swing_full_pipeline] Stage-2 done in %.2fs\n", ...
                stage2_duration_s);
    end

    % ---- 6. Ensure save dir exists ------------------------------------
    if ~isfolder(opts.save_dir)
        mkdir(opts.save_dir);
    end

    % ---- 7. Augment result --------------------------------------------
    result = inner_result;
    result.stage1_overrides   = stage1_overrides;
    result.save_dir           = string(opts.save_dir);
    result.target             = target;
    result.stage1_duration_s  = stage1_duration_s;
    result.stage2_duration_s  = stage2_duration_s;
    result.pipeline_duration_s = toc(pipeline_t0);

    % ---- 8. Render and save figures -----------------------------------
    fig_paths = cell(0, 1);
    if opts.render_figures
        fig_paths = local_render_figures(result, target, opts);
    end
    result.figure_paths = fig_paths;

    % ---- 9. Optional animation ----------------------------------------
    if opts.save_animation && opts.render_figures
        anim_path = local_render_animation(result, target, opts);
        if strlength(string(anim_path)) > 0
            result.animation_path = anim_path;
        end
    end

    % ---- 10. Write report ---------------------------------------------
    report_path = local_write_report(result, target, opts, solver_name);
    result.report_path = string(report_path);

    % ---- 11. Save the result struct -----------------------------------
    result_mat = fullfile(opts.save_dir, "result.mat");
    try
        save(char(result_mat), "result", "-v7");
    catch ME
        if opts.verbose
            fprintf("[fit_swing_full_pipeline] WARNING: could not save result.mat: %s\n", ...
                    ME.message);
        end
    end

    % ---- 12. Postconditions -------------------------------------------
    assert(isfield(result, "stage1_overrides"), ...
        "fit_swing_full_pipeline:postOverrides", ...
        "Postcondition: result must have .stage1_overrides");
    assert(isfield(result, "figure_paths") && iscell(result.figure_paths), ...
        "fit_swing_full_pipeline:postFigPaths", ...
        "Postcondition: result.figure_paths must be a cell array");
    assert(isfield(result, "report_path") && ...
           strlength(string(result.report_path)) > 0, ...
        "fit_swing_full_pipeline:postReport", ...
        "Postcondition: result.report_path must be a non-empty string");
    assert(isfile(char(result.report_path)), ...
        "fit_swing_full_pipeline:postReportFile", ...
        "Postcondition: result.report_path must point at an existing file");
    assert(isfolder(char(result.save_dir)), ...
        "fit_swing_full_pipeline:postSaveDir", ...
        "Postcondition: result.save_dir must exist");
    assert(string(result.solver) == solver_name, ...
        "fit_swing_full_pipeline:postSolver", ...
        "Postcondition: result.solver must equal dispatched solver name");

    if opts.verbose
        fprintf("[fit_swing_full_pipeline] all done in %.2fs (RMSE=%.2f mm)\n", ...
                result.pipeline_duration_s, ...
                1000 * local_get_field(result, "final_rmse_m", NaN));
    end
end


%% =====================================================================
function opts = local_fill_defaults(opts)
%LOCAL_FILL_DEFAULTS  Apply default values for unset OPTS fields.

    if ~isfield(opts, "sheet"),           opts.sheet           = "TW_ProV1";    end
    if ~isfield(opts, "option"),          opts.option          = "fmincon";     end
    if ~isfield(opts, "render_figures"),  opts.render_figures  = true;          end
    if ~isfield(opts, "save_animation"),  opts.save_animation  = false;         end
    if ~isfield(opts, "skip_stage1"),     opts.skip_stage1     = false;         end
    if ~isfield(opts, "verbose"),         opts.verbose         = true;          end

    if ~isfield(opts, "save_dir") || strlength(string(opts.save_dir)) == 0
        ts = char(datetime("now", "Format", "yyyyMMdd_HHmmss"));
        opts.save_dir = fullfile("output", "fits", ts);
    end
    opts.save_dir = char(opts.save_dir);

    if ~isfield(opts, "input_mat") || strlength(string(opts.input_mat)) == 0
        opts.input_mat = local_default_input_mat();
    end
    opts.input_mat = char(opts.input_mat);

    if ~isfield(opts, "stage1_opts"), opts.stage1_opts = struct(); end
    if ~isstruct(opts.stage1_opts)
        error("fit_swing_full_pipeline:badStage1Opts", ...
              "Precondition: opts.stage1_opts must be a struct");
    end
    if ~isfield(opts.stage1_opts, "verbose")
        opts.stage1_opts.verbose = opts.verbose;
    end

    if ~isfield(opts, "stage2_opts"), opts.stage2_opts = struct(); end
    if ~isstruct(opts.stage2_opts)
        error("fit_swing_full_pipeline:badStage2Opts", ...
              "Precondition: opts.stage2_opts must be a struct");
    end
    % Seed Stage-2 from default_option1_options where available, then
    % overlay caller-supplied fields (caller wins).
    try
        base_stage2 = default_option1_options();
    catch
        base_stage2 = struct();
    end
    opts.stage2_opts = local_merge_structs(base_stage2, opts.stage2_opts);
    % Apply the grip-primary cost weights from the playbook (caller can
    % override any of these by setting opts.stage2_opts.cost.<field>).
    opts.stage2_opts.cost = local_apply_grip_primary_cost( ...
        local_get_field(opts.stage2_opts, "cost", struct()));

    if ~isfield(opts, "viz_opts") || ~isstruct(opts.viz_opts)
        opts.viz_opts = default_viz_options();
    else
        opts.viz_opts = local_merge_structs(default_viz_options(), opts.viz_opts);
    end
    % Force headless rendering by default for batch / CI use; the caller
    % can flip this back to "on" by setting opts.viz_opts.visible="on".
    if ~isfield(opts.viz_opts, "visible") || ...
            strlength(string(opts.viz_opts.visible)) == 0
        opts.viz_opts.visible = "off";
    end

    % Validate dispatch name early so we fail fast.
    local_validate_option(opts.option);
end


%% =====================================================================
function cost = local_apply_grip_primary_cost(cost)
%LOCAL_APPLY_GRIP_PRIMARY_COST  Seed grip-primary defaults; do not clobber
%   any field the caller has already set.
    try
        base = default_cost_options();
    catch
        base = struct();
    end
    cost = local_merge_structs(base, cost);
    if ~isfield(cost, "w_position_grip"),     cost.w_position_grip     = 1.0;          end
    if ~isfield(cost, "w_orientation_grip"),  cost.w_orientation_grip  = 0.5;          end
    if ~isfield(cost, "w_position_clubhead"), cost.w_position_clubhead = 0.0;          end
    if ~isfield(cost, "w_orientation_club"),  cost.w_orientation_club  = 0.0;          end
    if ~isfield(cost, "w_anchor_impact"),     cost.w_anchor_impact     = 10.0;         end
    if ~isfield(cost, "lambda"),              cost.lambda              = 1e-4;         end
    if ~isfield(cost, "regularizer"),         cost.regularizer         = "total_work"; end
end


%% =====================================================================
function p = local_default_input_mat()
%LOCAL_DEFAULT_INPUT_MAT  Best-effort path to 3DModelInputs_Impact.mat.
%   Resolves relative to this .m file when running inside the repo; falls
%   back to the bare filename so MATLAB's path search can find it.
    here = fileparts(mfilename("fullpath"));
    % shared/ -> motion_matching/ -> matlab/ -> 3D_Golf_Model/
    engine_root = fileparts(fileparts(fileparts(here)));
    candidate   = fullfile(engine_root, "src", "model", "inputs", ...
                           "3DModelInputs_Impact.mat");
    if isfile(candidate)
        p = char(candidate);
    else
        p = "3DModelInputs_Impact.mat";
    end
end


%% =====================================================================
function local_validate_option(name)
%LOCAL_VALIDATE_OPTION  Reject unknown optimizer names with a clear error.
    allowed = ["fmincon", "multistart", "surrogate", "surrogateopt", "hybrid"];
    if ~ismember(string(name), allowed)
        error("fit_swing_full_pipeline:unknownOption", ...
              "Unknown optimizer '%s'. Allowed: %s", ...
              string(name), strjoin(allowed, ", "));
    end
end


%% =====================================================================
function [fn, name] = local_dispatch(option)
%LOCAL_DISPATCH  Map option name -> {handle, canonical solver string}.
    switch lower(string(option))
        case "fmincon"
            fn   = @fit_swing_fmincon;
            name = "fmincon";
        case "multistart"
            fn   = @fit_swing_multistart;
            name = "multistart";
        case {"surrogate", "surrogateopt"}
            fn   = @fit_swing_surrogateopt;
            name = "surrogateopt";
        case "hybrid"
            fn   = @fit_swing_hybrid;
            name = "hybrid";
        otherwise
            % Caught earlier by local_validate_option, but defensive.
            error("fit_swing_full_pipeline:unknownOption", ...
                  "Unknown optimizer '%s'", string(option));
    end
end


%% =====================================================================
function target = local_resolve_target(target_or_xlsx, opts)
%LOCAL_RESOLVE_TARGET  Accept either an xlsx path or a pre-loaded struct.
    if isstruct(target_or_xlsx)
        target = target_or_xlsx;
        required = ["time", "grip", "grip_quat", "butt", "clubhead", ...
                    "club_quat", "impact_idx", "events"];
        missing = setdiff(required, string(fieldnames(target)));
        if ~isempty(missing)
            error("fit_swing_full_pipeline:badTarget", ...
                  "Precondition: target struct missing fields: %s", ...
                  strjoin(missing, ", "));
        end
        return;
    end
    if ~(ischar(target_or_xlsx) || isstring(target_or_xlsx))
        error("fit_swing_full_pipeline:badTarget", ...
              "Precondition: target_or_xlsx must be a path or a struct");
    end
    xlsx_path = string(target_or_xlsx);
    if ~isfile(xlsx_path)
        error("fit_swing_full_pipeline:targetNotFound", ...
              "Precondition: xlsx file not found: %s", xlsx_path);
    end
    target = load_club_target_excel(xlsx_path, string(opts.sheet));
end


%% =====================================================================
function paths = local_render_figures(result, target, opts)
%LOCAL_RENDER_FIGURES  Render the three canonical views and save as PNG.
    paths = cell(0, 1);
    save_dir = opts.save_dir;
    viz = opts.viz_opts;

    % Each renderer is wrapped in a try so a single crash doesn't abort
    % the whole pipeline (useful when running before the model is fully
    % wired but Stage-1/2 already ran).
    paths = local_try_render(paths, save_dir, "trajectory_overlay.png", ...
        @() plot_trajectory_overlay(result, target, viz), opts.verbose);
    paths = local_try_render(paths, save_dir, "error_timecourse.png", ...
        @() plot_error_timecourse(result, target, viz), opts.verbose);
    paths = local_try_render(paths, save_dir, "fit_quality_card.png", ...
        @() plot_fit_quality_card(result, target, viz), opts.verbose);
end


%% =====================================================================
function paths = local_try_render(paths, save_dir, filename, render_fn, verbose)
%LOCAL_TRY_RENDER  Best-effort render+save; never aborts the pipeline.
    full_path = fullfile(save_dir, filename);
    fig = [];
    try
        fig = render_fn();
        if ~isempty(fig) && isgraphics(fig)
            try
                exportgraphics(fig, char(full_path), "Resolution", 200);
            catch
                saveas(fig, char(full_path));
            end
            paths{end + 1, 1} = char(full_path); %#ok<AGROW>
        end
    catch ME
        if verbose
            fprintf("[fit_swing_full_pipeline] WARNING: %s render failed: %s\n", ...
                    filename, ME.message);
        end
    end
    if ~isempty(fig) && isgraphics(fig)
        try, close(fig); catch, end %#ok<NOCOM>
    end
end


%% =====================================================================
function anim_path = local_render_animation(result, target, opts)
%LOCAL_RENDER_ANIMATION  Best-effort animation render.
    anim_path = "";
    out_path = fullfile(opts.save_dir, "trajectory_animation.mp4");
    viz = opts.viz_opts;
    viz.save_path = out_path;
    try
        animate_trajectory_overlay(result, target, viz);
        anim_path = string(out_path);
    catch ME
        if opts.verbose
            fprintf("[fit_swing_full_pipeline] WARNING: animation render failed: %s\n", ...
                    ME.message);
        end
    end
end


%% =====================================================================
function report_path = local_write_report(result, target, opts, solver_name)
%LOCAL_WRITE_REPORT  Emit a plain-text human-readable summary.
    report_path = fullfile(opts.save_dir, "report.txt");

    rmse_mm = 1000 * local_get_field(result, "final_rmse_m", NaN);
    work_J  = local_get_field(result, "final_total_work_J", NaN);
    pipeline_s = local_get_field(result, "pipeline_duration_s", NaN);
    stage1_s   = local_get_field(result, "stage1_duration_s", NaN);
    stage2_s   = local_get_field(result, "stage2_duration_s", NaN);
    target_hash = string(local_get_field(result, "target_hash", ""));
    git_commit  = string(local_get_field(result, "git_commit", ""));
    timestamp   = string(local_get_field(result, "timestamp_utc", ""));
    exitflag    = local_get_field(result, "exitflag", NaN);

    % Provenance for the target.
    src = local_get_field(target, "source", struct());
    src_filename = string(local_get_field(src, "filename", ""));
    src_format   = string(local_get_field(src, "format", ""));
    src_subject  = string(local_get_field(src, "subject_id", ""));
    src_trial    = string(local_get_field(src, "trial_id", ""));
    src_sha      = string(local_get_field(src, "sha256", ""));

    % Solver options summary.
    solver_opts = local_get_field(result, "solver_options", struct());
    max_iter = local_get_field(solver_opts, "MaxIterations", ...
                local_get_field(opts.stage2_opts, "max_iter", NaN));
    if ~isnumeric(max_iter); max_iter = double(max_iter); end

    % Cost breakdown.
    terms = local_get_field(result, "final_cost_terms", struct());

    % Stage-1 overrides table.
    s1 = local_get_field(result, "stage1_overrides", struct());
    s1_lines = local_format_overrides(s1);

    fid = fopen(char(report_path), "w");
    if fid < 0
        error("fit_swing_full_pipeline:reportOpen", ...
              "Could not open report file for writing: %s", report_path);
    end
    cleanup_fid = onCleanup(@() fclose(fid));

    fprintf(fid, "Grip-Primary Swing Fit Report\n");
    fprintf(fid, "================================\n\n");
    fprintf(fid, "Timestamp (UTC) : %s\n", timestamp);
    fprintf(fid, "Git commit      : %s\n", git_commit);
    fprintf(fid, "Save dir        : %s\n", string(opts.save_dir));
    fprintf(fid, "Sheet           : %s\n", string(opts.sheet));
    fprintf(fid, "\n");

    fprintf(fid, "Target source\n");
    fprintf(fid, "-------------\n");
    fprintf(fid, "  filename   : %s\n", src_filename);
    fprintf(fid, "  format     : %s\n", src_format);
    fprintf(fid, "  subject_id : %s\n", src_subject);
    fprintf(fid, "  trial_id   : %s\n", src_trial);
    fprintf(fid, "  sha256     : %s\n", src_sha);
    fprintf(fid, "  hash       : %s\n", target_hash);
    fprintf(fid, "\n");

    fprintf(fid, "Pipeline timing\n");
    fprintf(fid, "---------------\n");
    fprintf(fid, "  total      : %s\n", local_fmt_seconds(pipeline_s));
    fprintf(fid, "  Stage-1    : %s\n", local_fmt_seconds(stage1_s));
    fprintf(fid, "  Stage-2    : %s\n", local_fmt_seconds(stage2_s));
    fprintf(fid, "\n");

    fprintf(fid, "Stage-2 optimizer\n");
    fprintf(fid, "-----------------\n");
    fprintf(fid, "  name           : %s\n", solver_name);
    fprintf(fid, "  MaxIterations  : %s\n", local_fmt_num(max_iter));
    fprintf(fid, "  exitflag       : %s\n", local_fmt_num(exitflag));
    fprintf(fid, "\n");

    fprintf(fid, "Fit quality\n");
    fprintf(fid, "-----------\n");
    fprintf(fid, "  grip RMSE      : %s mm\n", local_fmt_num(rmse_mm));
    fprintf(fid, "  total work     : %s J\n", local_fmt_num(work_J));
    fprintf(fid, "\n");

    fprintf(fid, "Terminal cost breakdown\n");
    fprintf(fid, "-----------------------\n");
    if isstruct(terms)
        f = fieldnames(terms);
        if isempty(f)
            fprintf(fid, "  (no cost terms recorded)\n");
        end
        for k = 1:numel(f)
            v = terms.(f{k});
            if isnumeric(v) && isscalar(v)
                fprintf(fid, "  %-15s: %s\n", f{k}, local_fmt_num(double(v)));
            end
        end
    else
        fprintf(fid, "  (cost-terms struct unavailable)\n");
    end
    fprintf(fid, "\n");

    fprintf(fid, "Stage-1 overrides\n");
    fprintf(fid, "-----------------\n");
    if isempty(s1_lines)
        fprintf(fid, "  (none — Stage-1 skipped or returned empty)\n");
    else
        for k = 1:numel(s1_lines)
            fprintf(fid, "  %s\n", s1_lines{k});
        end
    end
    fprintf(fid, "\n");

    figs = local_get_field(result, "figure_paths", {});
    fprintf(fid, "Figures\n");
    fprintf(fid, "-------\n");
    if isempty(figs)
        fprintf(fid, "  (none rendered)\n");
    else
        for k = 1:numel(figs)
            fprintf(fid, "  %s\n", figs{k});
        end
    end

    clear cleanup_fid; %#ok<CLEAR>

    if ~isfile(char(report_path))
        error("fit_swing_full_pipeline:reportMissing", ...
              "Postcondition: report file was not created at %s", report_path);
    end
end


%% =====================================================================
function lines = local_format_overrides(s)
%LOCAL_FORMAT_OVERRIDES  Pretty-print an overrides struct as a cell of lines.
    lines = {};
    if ~isstruct(s); return; end
    f = fieldnames(s);
    for k = 1:numel(f)
        v = s.(f{k});
        if isnumeric(v) && isscalar(v)
            lines{end + 1, 1} = sprintf("%-32s = %+.6f", f{k}, double(v)); %#ok<AGROW>
        end
    end
end


%% =====================================================================
function s_out = local_merge_structs(base, overlay)
%LOCAL_MERGE_STRUCTS  Shallow-merge: overlay fields win over base fields.
    s_out = base;
    if ~isstruct(overlay); return; end
    f = fieldnames(overlay);
    for k = 1:numel(f)
        s_out.(f{k}) = overlay.(f{k});
    end
end


%% =====================================================================
function s_out = local_merge_overrides(base_overrides, stage1)
%LOCAL_MERGE_OVERRIDES  Stage-1 fields are layered on top of any caller-
%   supplied input_overrides.  Stage-1 wins on collisions.
    s_out = base_overrides;
    if ~isstruct(s_out); s_out = struct(); end
    if ~isstruct(stage1); return; end
    f = fieldnames(stage1);
    for k = 1:numel(f)
        s_out.(f{k}) = stage1.(f{k});
    end
end


%% =====================================================================
function v = local_get_field(s, name, default)
    if isstruct(s) && isfield(s, name)
        v = s.(name);
    else
        v = default;
    end
end


%% =====================================================================
function s = local_fmt_num(x)
    if isnumeric(x) && isscalar(x) && isfinite(x)
        if abs(x) >= 1e6 || (abs(x) > 0 && abs(x) < 1e-3)
            s = sprintf("%.4e", double(x));
        else
            s = sprintf("%.4f", double(x));
        end
    else
        s = "n/a";
    end
end


%% =====================================================================
function s = local_fmt_seconds(x)
    if isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0
        s = sprintf("%.2fs", double(x));
    else
        s = "n/a";
    end
end
