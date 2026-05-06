function fig = plot_fit_quality_card(result, target, opts)
%PLOT_FIT_QUALITY_CARD  View 3 single-figure fit quality summary card.
%
%   FIG = PLOT_FIT_QUALITY_CARD(RESULT, TARGET, OPTS) renders the View 3
%   summary card defined in VISUALIZATION_SPEC.md as a single figure that
%   is safe to drop into a PR description:
%
%     - Header  : swing id, solver, iterations, wall clock.
%     - Metrics : final clubhead/butt RMSE (mm), mean orientation error
%                 (deg), clubhead speed at impact (mph, sim vs meas).
%     - Reg     : total work (J), peak joint power (kW + joint name).
%     - Insets  : View 1 trajectory-overlay thumbnail (top right) and
%                 View 2 error-timecourse thumbnail (bottom right) rendered
%                 by reusing plot_trajectory_overlay.m and
%                 plot_error_timecourse.m with 'Visible','off' (DRY).
%     - Footer  : short target hash, branch, short git commit.
%
%   When OPTS.save_to_disk is true the figure is exported as a PNG at
%   OPTS.dpi (default 200) and saved alongside as a MATLAB .fig file in
%   OPTS.output_dir (default tempdir).  The base filename is derived from
%   RESULT.swing_id, falling back to "quality_card" plus the short target
%   hash.
%
%   Preconditions (per CODING_STANDARDS.md "Provenance and reproducibility"):
%     - RESULT is a scalar struct.  Required fields: solver, target_hash,
%       git_commit, duration_s, timestamp_utc.  Optional but consumed when
%       present: swing_id, branch, iterations, solver_options.iterations,
%       final_rmse_m, final_total_work_J, peak_joint_power_W,
%       peak_joint_name, sim_out.
%     - TARGET conforms to CLUB_IK_SPEC.md (time, butt, clubhead,
%       club_quat, impact_idx).
%   Postconditions:
%     - FIG is a valid matlab.ui.Figure handle.
%     - When OPTS.save_to_disk is true, the .png and .fig files exist on
%       disk under OPTS.output_dir.
%
%   GitHub issue: #3991.
    arguments
        result (1,1) struct {validators.mustHaveFields(result, ...
            ["solver", "target_hash", "git_commit", "duration_s", "timestamp_utc"])}
        target (1,1) struct {validators.mustHaveFields(target, ...
            ["time", "butt", "clubhead", "club_quat", "impact_idx"])}
        opts   (1,1) struct = default_viz_options()
    end

    opts = local_apply_card_defaults(opts);

    metrics = format_quality_metrics(result, target);
    header  = local_header_lines(result);
    footer  = local_footer_line(result);

    fig = figure('Visible', char(opts.figure_visible), ...
                 'Color', 'w', ...
                 'Name', 'Fit Quality Card (View 3)', ...
                 'NumberTitle', 'off', ...
                 'Units', 'inches', ...
                 'Position', [1 1 11 8.5]);

    % --- Header (top) ----------------------------------------------------
    annotation(fig, 'textbox', [0.02 0.90 0.96 0.08], ...
        'String', header, ...
        'EdgeColor', 'none', ...
        'FontSize', opts.fontsize_title, ...
        'FontWeight', 'bold', ...
        'VerticalAlignment', 'top', ...
        'Tag', 'card_header');

    % --- Metrics block (left) -------------------------------------------
    metrics_lines = local_metrics_lines(metrics);
    annotation(fig, 'textbox', [0.04 0.50 0.45 0.36], ...
        'String', metrics_lines, ...
        'EdgeColor', [0.85 0.85 0.85], ...
        'BackgroundColor', [0.98 0.98 0.98], ...
        'FontSize', opts.fontsize_axes + 1, ...
        'FontName', 'FixedWidth', ...
        'VerticalAlignment', 'top', ...
        'Tag', 'card_metrics');

    % --- Regularizer block (left, lower) --------------------------------
    reg_lines = [ ...
        sprintf("Total work (regularized):  %s", metrics.total_work_J); ...
        sprintf("Peak joint power:          %s", metrics.peak_power_kW)];
    annotation(fig, 'textbox', [0.04 0.30 0.45 0.16], ...
        'String', reg_lines, ...
        'EdgeColor', [0.85 0.85 0.85], ...
        'BackgroundColor', [0.98 0.98 0.98], ...
        'FontSize', opts.fontsize_axes + 1, ...
        'FontName', 'FixedWidth', ...
        'VerticalAlignment', 'top', ...
        'Tag', 'card_regularizer');

    % --- View 1 thumbnail (top right) -----------------------------------
    render_thumbnail(fig, [0.52 0.50 0.46 0.38], ...
        @() plot_trajectory_overlay(result, target, local_thumb_opts(opts)), ...
        'card_thumb_view1');

    % --- View 2 thumbnail (bottom right) --------------------------------
    render_thumbnail(fig, [0.52 0.08 0.46 0.38], ...
        @() plot_error_timecourse(result, target, local_thumb_opts(opts)), ...
        'card_thumb_view2');

    % --- Footer (bottom) ------------------------------------------------
    annotation(fig, 'textbox', [0.02 0.02 0.96 0.05], ...
        'String', footer, ...
        'EdgeColor', 'none', ...
        'FontSize', opts.fontsize_axes - 1, ...
        'FontAngle', 'italic', ...
        'Color', [0.30 0.30 0.30], ...
        'VerticalAlignment', 'middle', ...
        'Tag', 'card_footer');

    % --- Optional save --------------------------------------------------
    if opts.save_to_disk
        local_save_card(fig, result, opts);
    end

    % Postcondition: figure handle is valid and contains the four tagged
    % regions plus two thumbnail axes.
    assert(isgraphics(fig, 'figure'), ...
        "Postcondition: plot_fit_quality_card must return a valid figure handle");
    thumbs = findall(fig, 'Type', 'axes', '-regexp', 'Tag', '^card_thumb_');
    assert(numel(thumbs) >= 2, ...
        "Postcondition: card must include two thumbnail axes");
end

% =====================================================================
% Local helpers (LOD <= 2)
% =====================================================================
function opts = local_apply_card_defaults(opts)
    if ~isfield(opts, "save_to_disk"),    opts.save_to_disk    = false; end
    if ~isfield(opts, "output_dir"),      opts.output_dir      = string(tempdir); end
    if ~isfield(opts, "figure_visible"),  opts.figure_visible  = opts.visible; end
    if ~isfield(opts, "dpi"),             opts.dpi             = 200; end
    if ~isfield(opts, "fontsize_axes"),   opts.fontsize_axes   = 11; end
    if ~isfield(opts, "fontsize_title"),  opts.fontsize_title  = 13; end
end

function lines = local_header_lines(result)
    swing_id = local_field_or(result, "swing_id", "(unnamed swing)");
    solver = string(result.solver);
    iters = local_iter_count(result);
    wall = local_format_duration(double(result.duration_s));
    lines = [ ...
        sprintf("Swing: %s", swing_id); ...
        sprintf("Solver: %s   Iterations: %s   Wall clock: %s", solver, iters, wall)];
end

function s = local_iter_count(result)
    if isfield(result, "iterations") && isscalar(result.iterations)
        s = sprintf("%d", round(double(result.iterations)));
        return;
    end
    if isfield(result, "solver_options") && isstruct(result.solver_options) ...
            && isfield(result.solver_options, "iterations")
        s = sprintf("%d", round(double(result.solver_options.iterations)));
        return;
    end
    s = "n/a";
end

function s = local_format_duration(seconds_val)
    if ~isfinite(seconds_val) || seconds_val < 0
        s = "n/a";
        return;
    end
    minutes = floor(seconds_val / 60);
    secs = round(seconds_val - minutes * 60);
    if minutes > 0
        s = sprintf("%dm %ds", minutes, secs);
    else
        s = sprintf("%ds", secs);
    end
end

function lines = local_metrics_lines(metrics)
    lines = [ ...
        sprintf("Final RMSE - clubhead position:  %s", metrics.clubhead_rmse_mm); ...
        sprintf("Final RMSE - butt position:      %s", metrics.butt_rmse_mm); ...
        sprintf("Final mean orientation error:    %s", metrics.orient_err_deg); ...
        sprintf("Final clubhead speed at impact:  %s", metrics.speed_at_impact)];
end

function txt = local_footer_line(result)
    hash_short = local_short(string(result.target_hash), 7);
    commit_short = local_short(string(result.git_commit), 7);
    branch = local_field_or(result, "branch", "(unknown)");
    txt = sprintf("Hash: %s   Branch: %s   Commit: %s   %s", ...
        hash_short, branch, commit_short, string(result.timestamp_utc));
end

function s = local_short(str, n)
    str = string(str);
    if strlength(str) <= n
        s = str;
    else
        s = extractBefore(str, n + 1);
    end
end

function v = local_field_or(s, name, default)
    if isfield(s, name) && ~isempty(s.(name))
        v = string(s.(name));
    else
        v = string(default);
    end
end

function thumb_opts = local_thumb_opts(opts)
%LOCAL_THUMB_OPTS  Force visibility off for inset rendering, keep styling.
%   The thumbnail figures are created off-screen and then their axes are
%   copied onto the parent card by render_thumbnail.m.  This keeps the
%   View 1 and View 2 plotting code unchanged (DRY).
    thumb_opts = opts;
    thumb_opts.visible = "off";
    thumb_opts.figure_visible = "off";
    thumb_opts.show_impact_marker = false;  % busy in a thumbnail
end

function local_save_card(fig, result, opts)
    out_dir = char(opts.output_dir);
    if ~isfolder(out_dir)
        mkdir(out_dir);
    end
    base = local_filename_base(result);
    png_path = fullfile(out_dir, base + ".png");
    fig_path = fullfile(out_dir, base + ".fig");
    if isprop(fig, 'PaperPositionMode')
        fig.PaperPositionMode = 'auto';
    end
    exportgraphics(fig, char(png_path), ...
        'Resolution', double(opts.dpi), ...
        'BackgroundColor', 'white');
    savefig(fig, char(fig_path));
    assert(isfile(png_path), "Postcondition: card PNG not written: %s", png_path);
    assert(isfile(fig_path), "Postcondition: card FIG not written: %s", fig_path);
end

function base = local_filename_base(result)
    if isfield(result, "swing_id") && strlength(string(result.swing_id)) > 0
        base = matlab.lang.makeValidName(string(result.swing_id));
        return;
    end
    base = "quality_card_" + local_short(string(result.target_hash), 7);
end
