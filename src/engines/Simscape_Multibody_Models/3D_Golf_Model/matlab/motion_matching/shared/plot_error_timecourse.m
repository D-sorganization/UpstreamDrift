function fig = plot_error_timecourse(result, target, opts)
%PLOT_ERROR_TIMECOURSE  Stacked 4-panel error timecourse (View 2).
%
%   FIG = PLOT_ERROR_TIMECOURSE(RESULT, TARGET, OPTS) renders the
%   View-2 stacked plot defined in VISUALIZATION_SPEC.md:
%
%     Panel 1: Position error (mm) - butt (blue) and clubhead (orange),
%              with shaded +/-1 sigma band from sample-rate noise.
%     Panel 2: Orientation error (deg) - geodesic distance per frame:
%                d_geo = 2 * acos(|q_sim . q_meas|), in degrees.
%     Panel 3: Clubhead speed (mph) - measured (solid) vs simulated (dashed).
%     Panel 4: Joint torques (N*m) - one trace per joint, lines colororder.
%
%   A single vertical line at the impact frame is drawn across all panels.
%
%   Preconditions:
%     - RESULT is a scalar struct with field .sim_out.
%     - RESULT.sim_out has fields {time, tau} plus butt/clubhead/quat
%       trajectories under either {butt,clubhead,club_quat} OR the canonical
%       extract_sim_out names {r_butt,r_clubhead,q_club}.
%     - TARGET conforms to CLUB_IK_SPEC.md: {time, butt, clubhead, club_quat,
%       impact_idx}.
%   Postconditions:
%     - FIG is a matlab.ui.Figure with exactly four axes (subplots).
    arguments
        result (1,1) struct {validators.mustHaveFields(result, "sim_out")}
        target (1,1) struct {validators.mustHaveFields(target, ...
            ["time", "butt", "clubhead", "club_quat", "impact_idx"])}
        opts   (1,1) struct = default_viz_options()
    end

    sim_out = local_normalize_sim_out(result.sim_out);
    validators.mustHaveFields(sim_out, ...
        ["time", "butt", "clubhead", "club_quat", "tau"]);

    % --- per-frame errors (delegate to private helpers) ------------------
    pos_err_butt_mm  = compute_pointwise_position_error( ...
                            sim_out.butt, target.butt) * 1000;
    pos_err_ch_mm    = compute_pointwise_position_error( ...
                            sim_out.clubhead, target.clubhead) * 1000;
    ori_err_deg      = compute_pointwise_orientation_error( ...
                            sim_out.club_quat, target.club_quat);
    speed_meas_mph   = compute_clubhead_speed_mph(target.time, target.clubhead);
    speed_sim_mph    = compute_clubhead_speed_mph(sim_out.time, sim_out.clubhead);

    sigma_mm = double(opts.sample_rate_noise_m) * 1000;  % +/-1 sigma band
    t_sim    = sim_out.time;
    t_meas   = target.time;
    impact_t = local_impact_time(target);

    % --- figure shell ----------------------------------------------------
    fig = figure('Visible', opts.figure_visible, ...
                 'Color', 'w', ...
                 'Units', 'inches', ...
                 'Position', [1 1 8 9]);
    ax = gobjects(4, 1);
    for k = 1:4
        ax(k) = subplot(4, 1, k, 'Parent', fig);
        hold(ax(k), 'on');
        set(ax(k), 'FontSize', 11);
    end

    % --- panel 1: position error (mm) ------------------------------------
    local_draw_sigma_band(ax(1), t_sim, sigma_mm);
    plot(ax(1), t_sim, pos_err_butt_mm, '-', ...
        'Color', '#1f77b4', 'LineWidth', 1.5, 'DisplayName', 'butt');
    plot(ax(1), t_sim, pos_err_ch_mm, '-', ...
        'Color', '#ff7f0e', 'LineWidth', 1.5, 'DisplayName', 'clubhead');
    ylabel(ax(1), 'Position error (mm)');
    title(ax(1), 'Error timecourse', 'FontSize', 13);
    legend(ax(1), 'Location', 'best');

    % --- panel 2: orientation error (deg) --------------------------------
    plot(ax(2), t_sim, ori_err_deg, '-', ...
        'Color', '#7f7f7f', 'LineWidth', 1.5);
    ylabel(ax(2), 'Orientation error (deg)');

    % --- panel 3: clubhead speed (mph) -----------------------------------
    plot(ax(3), t_meas, speed_meas_mph, '-',  ...
        'Color', '#1f77b4', 'LineWidth', 1.5, 'DisplayName', 'measured');
    plot(ax(3), t_sim,  speed_sim_mph,  '--', ...
        'Color', '#d62728', 'LineWidth', 1.5, 'DisplayName', 'simulated');
    ylabel(ax(3), 'Clubhead speed (mph)');
    legend(ax(3), 'Location', 'best');

    % --- panel 4: joint torques (N*m) ------------------------------------
    colororder(ax(4), 'lines');
    plot(ax(4), t_sim, sim_out.tau, 'LineWidth', 1.25);
    ylabel(ax(4), 'Joint torque (N\cdotm)');
    xlabel(ax(4), 'Simulation time (s)');

    % --- impact line across all panels -----------------------------------
    for k = 1:4
        xline(ax(k), impact_t, '-', ...
            'Color', [0 0 0 0.5], 'LineWidth', 1.0, ...
            'Label', '', 'HandleVisibility', 'off');
    end
end

% ---------- helpers (LOD <= 2) ----------

function s = local_normalize_sim_out(s)
%LOCAL_NORMALIZE_SIM_OUT  Accept either {butt,clubhead,club_quat} or the
%   extract_sim_out aliases {r_butt,r_clubhead,q_club}.  This keeps the
%   plot DRY across both naming conventions used elsewhere in the package.
    if ~isfield(s, "butt")     && isfield(s, "r_butt"),     s.butt     = s.r_butt;     end
    if ~isfield(s, "clubhead") && isfield(s, "r_clubhead"), s.clubhead = s.r_clubhead; end
    if ~isfield(s, "club_quat") && isfield(s, "q_club"),    s.club_quat = s.q_club;    end
end

function t = local_impact_time(target)
    k = double(target.impact_idx);
    if k < 1 || k > numel(target.time)
        error("plot_error_timecourse:badImpactIdx", ...
              "target.impact_idx out of range.");
    end
    t = target.time(k);
end

function local_draw_sigma_band(ax, t, sigma_mm)
%LOCAL_DRAW_SIGMA_BAND  Shaded +/-1 sigma band centred on zero (mm).
    if sigma_mm <= 0, return; end
    t = t(:);
    xfill = [t; flipud(t)];
    yfill = [ sigma_mm * ones(numel(t), 1); ...
             -sigma_mm * ones(numel(t), 1)];
    h = fill(ax, xfill, yfill, [0.85 0.85 0.85], ...
             'EdgeColor', 'none', 'FaceAlpha', 0.4);
    set(h, 'HandleVisibility', 'off');
end
