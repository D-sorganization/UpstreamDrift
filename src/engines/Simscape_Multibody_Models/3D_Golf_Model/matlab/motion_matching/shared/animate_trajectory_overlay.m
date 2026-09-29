function out = animate_trajectory_overlay(result, target, opts)
%ANIMATE_TRAJECTORY_OVERLAY  View 1 animated: time-driven measured/simulated.
%
%   OUT = ANIMATE_TRAJECTORY_OVERLAY(RESULT, TARGET, OPTS) creates the
%   animated variant of plot_trajectory_overlay. Behaviour depends on
%   OPTS.save_path:
%
%     * If OPTS.save_path is non-empty, frames are written to a
%       VideoWriter using the OPTS.video_format profile at OPTS.video_fps,
%       and the (still open) writer handle is returned. The caller is
%       responsible for `close(writer)`.
%     * If OPTS.save_path is empty, an interactive figure with a time
%       slider and play button is created and the figure handle is
%       returned. Live redraw is capped at OPTS.live_refresh_hz Hz per the
%       VISUALIZATION_SPEC live-update rule.
%
%   The two 3D axes share camera and limits via setup_camera_lock so the
%   eye sees measured-vs-simulated drift directly.
%
%   GitHub issue: #3989.
    arguments
        result (1,1) struct
        target (1,1) struct
        opts   (1,1) struct = default_viz_options()
    end

    % Reuse the still-image entry as the validation + initial-render path.
    fig = plot_trajectory_overlay(result, target, opts);
    sim_out = result.sim_out;

    n_meas = size(target.clubhead, 1);
    n_sim  = size(sim_out.r_clubhead, 1);
    n_frames = min(n_meas, n_sim);

    do_save = strlength(string(opts.save_path)) > 0;

    if do_save
        out = run_save_path(fig, target, sim_out, n_frames, opts);
    else
        out = run_interactive(fig, target, sim_out, n_frames, opts);
    end
end

% =====================================================================
% Interactive path (slider + play button)
% =====================================================================
function fig = run_interactive(fig, target, sim_out, n_frames, opts)
    % Add a slider + play button as uicontrols on the figure. Skip on
    % headless figures (Visible off) — the controls are inert anyway.
    state = struct('frame', 1, 'last_redraw', 0);
    setappdata(fig, 'AnimState', state);

    slider = uicontrol(fig, 'Style', 'slider', ...
        'Units', 'normalized', 'Position', [0.10 0.02 0.65 0.04], ...
        'Min', 1, 'Max', max(2, n_frames), ...
        'Value', 1, 'SliderStep', [1 / max(1, n_frames - 1), 0.1]);
    slider.Callback = @(src, ~) on_slider(src, fig, target, sim_out, opts);

    uicontrol(fig, 'Style', 'pushbutton', ...
        'Units', 'normalized', 'Position', [0.78 0.02 0.08 0.04], ...
        'String', 'Play', ...
        'Callback', @(~, ~) play_loop(fig, slider, target, sim_out, n_frames, opts));
end

function on_slider(src, fig, target, sim_out, opts)
    frame = max(1, round(src.Value));
    redraw_throttled(fig, target, sim_out, frame, opts);
end

function play_loop(fig, slider, target, sim_out, n_frames, opts)
    period = 1 / max(1, opts.live_refresh_hz);
    for k = 1 : n_frames
        if ~isvalid(fig)
            return
        end
        slider.Value = k;
        redraw_throttled(fig, target, sim_out, k, opts);
        pause(period);
    end
end

function redraw_throttled(fig, target, sim_out, frame, opts)
    state = getappdata(fig, 'AnimState');
    now_s = posix_now();
    min_dt = 1 / max(1, opts.live_refresh_hz);
    if (now_s - state.last_redraw) < min_dt
        return
    end
    state.frame = frame;
    state.last_redraw = now_s;
    setappdata(fig, 'AnimState', state);
    redraw_frame(fig, target, sim_out, frame, opts);
end

function t = posix_now()
    t = posixtime(datetime('now', 'TimeZone', 'UTC'));
end

% =====================================================================
% Save path (VideoWriter)
% =====================================================================
function vw = run_save_path(fig, target, sim_out, n_frames, opts)
    vw = VideoWriter(char(opts.save_path), char(opts.video_format));
    vw.FrameRate = opts.video_fps;
    open(vw);
    for k = 1 : n_frames
        if ~isvalid(fig)
            break
        end
        redraw_frame(fig, target, sim_out, k, opts);
        drawnow limitrate;
        try
            frame_img = getframe(fig);
            writeVideo(vw, frame_img);
        catch err
            % Headless or off-screen rendering may fail getframe; abort
            % gracefully so the writer is returned in a closeable state.
            warning("animate_trajectory_overlay:getframe", ...
                    "getframe failed at frame %d: %s", k, err.message);
            break
        end
    end
    % Postcondition: writer is a valid VideoWriter handle the caller can close().
    assert(isa(vw, 'VideoWriter'), ...
        'Postcondition: animate_trajectory_overlay save mode must return VideoWriter');
end

% =====================================================================
% Frame redraw — clears axes contents and redraws skeleton at frame K.
% =====================================================================
function redraw_frame(fig, target, sim_out, frame, opts)
    axs = findall(fig, 'Type', 'axes', '-not', 'Tag', 'error_inset');
    if numel(axs) < 2
        return
    end
    % Order is creation order reversed in findall; sort by left-edge.
    [~, ord] = sort(arrayfun(@(a) a.Position(1), axs));
    ax_left  = axs(ord(1));
    ax_right = axs(ord(2));

    n_meas = size(target.clubhead, 1);
    n_sim  = size(sim_out.r_clubhead, 1);
    fm = max(1, min(frame, n_meas));
    fs = max(1, min(frame, n_sim));

    refresh_axes(ax_left,  target.butt(fm,:),  target.clubhead(fm,:),  ...
                 target.clubhead, opts.color_measured, opts);
    refresh_axes(ax_right, sim_out.r_butt(fs,:), sim_out.r_clubhead(fs,:), ...
                 sim_out.r_clubhead, opts.color_simulated, opts);
end

function refresh_axes(ax, butt_xyz, head_xyz, traj, color, opts)
    cla(ax);
    hold(ax, "on");
    plot3(ax, traj(:,1), traj(:,2), traj(:,3), ...
          '-', 'Color', color, 'LineWidth', 1.0);
    plot3(ax, [butt_xyz(1), head_xyz(1)], ...
              [butt_xyz(2), head_xyz(2)], ...
              [butt_xyz(3), head_xyz(3)], ...
          '-', 'Color', color, 'LineWidth', 2.0);
    scatter3(ax, butt_xyz(1), butt_xyz(2), butt_xyz(3), 36, ...
             'MarkerFaceColor', color, 'MarkerEdgeColor', color);
    scatter3(ax, head_xyz(1), head_xyz(2), head_xyz(3), 72, ...
             'MarkerFaceColor', color, 'MarkerEdgeColor', color);
    hold(ax, "off");
    if isfield(opts, 'fontsize_axes')
        ax.FontSize = opts.fontsize_axes;
    end
end
