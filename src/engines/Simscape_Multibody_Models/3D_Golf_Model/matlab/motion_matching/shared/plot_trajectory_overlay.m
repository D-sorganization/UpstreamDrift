function fig = plot_trajectory_overlay(result, target, opts)
%PLOT_TRAJECTORY_OVERLAY  View 1 still: side-by-side measured/simulated club.
%
%   FIG = PLOT_TRAJECTORY_OVERLAY(RESULT, TARGET, OPTS) renders View 1 of
%   VISUALIZATION_SPEC.md as a static figure:
%
%     * Left  axes: measured club skeleton (butt -> clubhead) at the impact
%       frame, faint full-trajectory clubhead trace.
%     * Right axes: simulated skeleton from RESULT.sim_out at the same
%       frame, identical camera / axis limits / aspect ratio.
%     * Inset axes (lower right): per-frame error vector (sim -> meas) at
%       the impact frame in grey.
%
%   The measured colour is OPTS.color_measured (#1f77b4 by default), the
%   simulated colour is OPTS.color_simulated (#d62728), the error is
%   OPTS.color_error (#7f7f7f).
%
%   Inputs
%     RESULT  (1,1) struct with a SIM_OUT field per simulate_with_coefficients
%             postcondition; specifically requires r_butt (Nx3),
%             r_clubhead (Nx3), and time (Nx1).
%     TARGET  (1,1) struct conforming to CLUB_IK_SPEC.md (butt, clubhead,
%             time, club_quat, impact_idx).
%     OPTS    (1,1) struct from default_viz_options().
%
%   Output
%     FIG     valid figure handle. fig.Children contains at least the two
%             main 3D axes plus the inset.
%
%   Side-effect free w.r.t. inputs: RESULT and TARGET are not modified.
%
%   GitHub issue: #3989.
    arguments
        result (1,1) struct
        target (1,1) struct
        opts   (1,1) struct = default_viz_options()
    end

    validate_inputs(result, target);

    sim_out = result.sim_out;

    % Resolve impact frame: prefer target.impact_idx, fall back to argmax
    % clubhead speed if the optional field is missing on the sim side.
    n_meas = size(target.clubhead, 1);
    impact_idx = double(target.impact_idx);
    impact_idx = max(1, min(impact_idx, n_meas));
    sim_impact_idx = pick_sim_impact_frame(sim_out);

    % Figure: respect opts.visible so CI runs headless without a display.
    fig = figure('Visible', char(opts.visible), 'Color', 'w', ...
                 'Name', 'Trajectory Overlay (View 1)', ...
                 'NumberTitle', 'off', 'Position', [100 100 1200 540]);

    ax_left  = subplot(1, 2, 1, 'Parent', fig);
    ax_right = subplot(1, 2, 2, 'Parent', fig);

    title(ax_left,  'Measured', 'FontSize', opts.fontsize_title);
    title(ax_right, 'Simulated', 'FontSize', opts.fontsize_title);
    xlabel(ax_left,  'x (m)'); ylabel(ax_left,  'y (m)'); zlabel(ax_left,  'z (m)');
    xlabel(ax_right, 'x (m)'); ylabel(ax_right, 'y (m)'); zlabel(ax_right, 'z (m)');

    % --- Faint full-trajectory traces ---
    plot_clubhead_trace(ax_left,  target.clubhead, opts.color_measured,  opts.trace_alpha);
    plot_clubhead_trace(ax_right, sim_out.r_clubhead, opts.color_simulated, opts.trace_alpha);

    % --- Skeletons at impact frame ---
    draw_club_skeleton(ax_left, ...
        target.butt(impact_idx, :), target.clubhead(impact_idx, :), ...
        opts.color_measured, opts);
    draw_club_skeleton(ax_right, ...
        sim_out.r_butt(sim_impact_idx, :), sim_out.r_clubhead(sim_impact_idx, :), ...
        opts.color_simulated, opts);

    % --- Camera lock ---
    all_points = [target.butt; target.clubhead; sim_out.r_butt; sim_out.r_clubhead];
    setup_camera_lock(ax_left, ax_right, all_points);

    % --- Error-vector inset ---
    if opts.show_impact_marker
        meas_head = target.clubhead(impact_idx, :);
        sim_head  = sim_out.r_clubhead(sim_impact_idx, :);
        add_error_inset(fig, sim_head, meas_head, opts);
    end

    % Postcondition: figure handle valid and has expected children.
    assert(isgraphics(fig, 'figure'), ...
        'Postcondition: plot_trajectory_overlay must return a valid figure handle');
    main_axes = findall(fig, 'Type', 'axes');
    assert(numel(main_axes) >= 2, ...
        'Postcondition: figure must contain at least two axes');
end

% =====================================================================
% Local helpers (kept short to honour the LOD <= 2 rule)
% =====================================================================
function validate_inputs(result, target)
    target_fields = ["time", "butt", "clubhead", "club_quat", "impact_idx"];
    validators.mustHaveFields(target, target_fields);
    validators.mustHaveFields(result, "sim_out");
    sim_out = result.sim_out;
    if ~isstruct(sim_out) || ~isscalar(sim_out)
        error("plot_trajectory_overlay:badSimOut", ...
              "result.sim_out must be a scalar struct.");
    end
    validators.mustHaveFields(sim_out, ["time", "r_butt", "r_clubhead"]);

    if size(target.butt, 2) ~= 3 || size(target.clubhead, 2) ~= 3
        error("plot_trajectory_overlay:badShape", ...
              "target.butt and target.clubhead must be Nx3.");
    end
    if size(sim_out.r_butt, 2) ~= 3 || size(sim_out.r_clubhead, 2) ~= 3
        error("plot_trajectory_overlay:badShape", ...
              "sim_out.r_butt and sim_out.r_clubhead must be Nx3.");
    end
end

function plot_clubhead_trace(ax, traj, color, alpha)
    holdState = ishold(ax);
    hold(ax, "on");
    h = plot3(ax, traj(:, 1), traj(:, 2), traj(:, 3), ...
              '-', 'Color', color, 'LineWidth', 1.0);
    try
        h.Color(4) = alpha;  %#ok<NASGU> set alpha if supported
        set(h, 'Color', [hex2rgb(color), alpha]);
    catch
        % fallback: alpha not supported on this MATLAB; leave solid.
    end
    if ~holdState
        hold(ax, "off");
    end
end

function rgb = hex2rgb(hex)
    if isstring(hex) || ischar(hex)
        s = char(hex);
        if startsWith(s, '#')
            s = s(2:end);
        end
        rgb = [hex2dec(s(1:2)), hex2dec(s(3:4)), hex2dec(s(5:6))] / 255;
    else
        rgb = double(hex);
    end
end

function add_error_inset(fig, sim_head, meas_head, opts)
    % Inset axes inside fig, lower-right corner, normalised units.
    inset = axes('Parent', fig, 'Position', [0.78 0.08 0.18 0.22], ...
                 'Tag', 'error_inset');
    err = meas_head - sim_head;
    quiver3(inset, sim_head(1), sim_head(2), sim_head(3), ...
            err(1), err(2), err(3), 0, ...
            'Color', opts.color_error, 'LineWidth', 1.5, ...
            'MaxHeadSize', 0.6);
    hold(inset, "on");
    scatter3(inset, sim_head(1), sim_head(2), sim_head(3), ...
             40, 'MarkerEdgeColor', opts.color_simulated, ...
                 'MarkerFaceColor', opts.color_simulated);
    scatter3(inset, meas_head(1), meas_head(2), meas_head(3), ...
             40, 'MarkerEdgeColor', opts.color_measured, ...
                 'MarkerFaceColor', opts.color_measured);
    hold(inset, "off");
    title(inset, sprintf('Impact err: %.1f mm', 1000 * norm(err)), ...
          'FontSize', opts.fontsize_axes - 1);
    grid(inset, "on");
    daspect(inset, [1 1 1]);
    view(inset, 35, 25);
end

function idx = pick_sim_impact_frame(sim_out)
    if isfield(sim_out, "impact_idx")
        idx = double(sim_out.impact_idx);
    else
        v = sim_out.r_clubhead;
        if size(v, 1) >= 2
            d = diff(v, 1, 1);
            speed = sqrt(sum(d .^ 2, 2));
            [~, idx] = max(speed);
            idx = idx + 1;  % diff loses one row
        else
            idx = 1;
        end
    end
    idx = max(1, min(idx, size(sim_out.r_clubhead, 1)));
end
