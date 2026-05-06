function fig = plot_starting_position_match(skel, target, opts)
%PLOT_STARTING_POSITION_MATCH  Skeleton vs measured club path with live offset sliders.
%
%   FIG = PLOT_STARTING_POSITION_MATCH(SKEL, TARGET) renders the model
%   skeleton (all joint centres from LOAD_IMPACT_STARTING_POSITION) on the
%   same 3D axes as the measured club path (butt and clubhead samples from
%   a club-IK target struct), so the user can see how well the model's
%   starting pose matches reality before running training.
%
%   FIG = PLOT_STARTING_POSITION_MATCH(SKEL, TARGET, OPTS) accepts:
%     .visible       'on' | 'off'  (default 'on'; set 'off' for batch tests)
%     .impact_idx    target frame index whose butt/clubhead the model is
%                    being aligned against (default target.impact_idx)
%     .save_png      optional PNG path to save before returning
%
%   Inputs
%     SKEL   struct from LOAD_IMPACT_STARTING_POSITION with fields butt, mp,
%            ch, lw, le, ls, rw, re, rs, hub (each 1x3 metres).
%     TARGET struct conforming to CLUB_IK_SPEC.md (time, butt Nx3,
%            clubhead Nx3, club_quat Nx4, impact_idx).
%
%   The figure includes:
%     - 3D skeleton (cylinders + joint markers) using the model joints.
%     - Faint trace of the measured clubhead path through the swing.
%     - Markers for the measured butt + clubhead at the impact frame.
%     - Sliders that translate (X/Y/Z, ±0.5 m) and yaw-rotate (±45°) the
%       measured data; the figure updates live and reports butt/clubhead
%       RMS error in millimetres.
%     - "Apply offset" / "Reset" / "Save offset to .mat" buttons.
%
%   The transform applied to the measured data is
%       p' = R_z(yaw) * p + [dx, dy, dz]
%   so callers can later use that same transform to align measured data
%   into the model frame for cost-function evaluation.
%
%   See also: LOAD_IMPACT_STARTING_POSITION, LOAD_CLUB_TARGET_EXCEL.

    arguments
        skel   (1,1) struct
        target (1,1) struct
        opts   (1,1) struct = struct()
    end
    opts = local_fill_defaults(opts, target);
    local_validate(skel, target);

    state = struct();
    state.skel        = skel;
    state.target      = target;
    state.impact_idx  = opts.impact_idx;
    state.dx = 0; state.dy = 0; state.dz = 0;
    state.yaw_deg = 0;

    fig = figure('Visible', char(opts.visible), 'Color', 'w', ...
                 'Name', 'Starting Position Match (Model vs Measured)', ...
                 'NumberTitle', 'off', 'Position', [80 80 1280 760]);

    state.ax = axes('Parent', fig, 'Position', [0.06 0.12 0.66 0.82], ...
                    'Color', 'w', 'GridColor', [0.7 0.7 0.7], ...
                    'XColor', 'k', 'YColor', 'k', 'ZColor', 'k', ...
                    'Box', 'on');
    hold(state.ax, 'on'); grid(state.ax, 'on');
    axis(state.ax, 'equal');
    xlabel(state.ax, 'X (m)'); ylabel(state.ax, 'Y (m)'); zlabel(state.ax, 'Z (m)');
    view(state.ax, [-45 22]);

    state.handles = local_init_graphics(state.ax, state.skel, state.target);
    local_lock_camera(state.ax, state.skel, state.target);

    % --- Side panel with sliders, buttons, and RMSE readout. -----------
    panel = uipanel('Parent', fig, 'Title', 'Measured-frame offset', ...
                    'Units', 'normalized', 'Position', [0.74 0.12 0.24 0.82], ...
                    'BackgroundColor', [0.96 0.96 0.96], 'FontSize', 10);
    state.controls = local_build_controls(panel, fig);
    state.rmse_text = uicontrol('Parent', fig, 'Style', 'text', ...
        'Units', 'normalized', 'Position', [0.06 0.02 0.66 0.07], ...
        'BackgroundColor', 'w', 'FontSize', 10, ...
        'HorizontalAlignment', 'left', 'String', '');

    setappdata(fig, 'state', state);

    % Wire callbacks now that the state lives on the figure.
    set(state.controls.dx, 'Callback', @(s,~) local_on_slider(fig, 'dx', s.Value));
    set(state.controls.dy, 'Callback', @(s,~) local_on_slider(fig, 'dy', s.Value));
    set(state.controls.dz, 'Callback', @(s,~) local_on_slider(fig, 'dz', s.Value));
    set(state.controls.yaw,'Callback', @(s,~) local_on_slider(fig, 'yaw_deg', s.Value));
    set(state.controls.btn_reset, 'Callback', @(~,~) local_reset(fig));
    set(state.controls.btn_save,  'Callback', @(~,~) local_save_offset(fig));
    set(state.controls.btn_view_face, 'Callback', @(~,~) view(state.ax, [0 0]));
    set(state.controls.btn_view_dtl,  'Callback', @(~,~) view(state.ax, [270 0]));
    set(state.controls.btn_view_top,  'Callback', @(~,~) view(state.ax, [0 90]));
    set(state.controls.btn_view_iso,  'Callback', @(~,~) view(state.ax, [-45 22]));

    local_redraw(fig);

    if isfield(opts, 'save_png') && ~isempty(opts.save_png)
        try
            exportgraphics(fig, opts.save_png, 'Resolution', 200);
        catch ME
            warning('plot_starting_position_match:saveFailed', ...
                    'Could not save PNG: %s', ME.message);
        end
    end
end

%% =====================================================================
function opts = local_fill_defaults(opts, target)
    if ~isfield(opts, 'visible'),    opts.visible    = 'on';                  end
    if ~isfield(opts, 'impact_idx'), opts.impact_idx = double(target.impact_idx); end
    if ~isfield(opts, 'save_png'),   opts.save_png   = '';                    end
end

%% =====================================================================
function local_validate(skel, target)
    needed = {'butt','mp','ch','lw','le','ls','rw','re','rs','hub'};
    for k = 1:numel(needed)
        if ~isfield(skel, needed{k})
            error('plot_starting_position_match:badSkel', ...
                  'skel struct missing field "%s" (run load_impact_starting_position).', needed{k});
        end
    end
    % Optional fields the new skeleton extractor surfaces — fall back to
    % NaN if a caller built the struct manually.
    optional = {'hip','torso','spine'};
    for k = 1:numel(optional)
        if ~isfield(skel, optional{k}); skel.(optional{k}) = nan(1, 3); end %#ok<NASGU>
    end
    needed_t = {'time','butt','clubhead','impact_idx'};
    for k = 1:numel(needed_t)
        if ~isfield(target, needed_t{k})
            error('plot_starting_position_match:badTarget', ...
                  'target struct missing field "%s" (load with load_club_target_excel).', needed_t{k});
        end
    end
end

%% =====================================================================
function h = local_init_graphics(ax, skel, target)
    skin   = [1.00 0.80 0.60];
    shirt  = [0.60 0.80 1.00];
    grey   = [0.50 0.50 0.50];
    meas_c = [31 119 180]/255;     % matches default_viz_options measured colour

    % Skeleton segments — drawn as plain lines so we don't need cylinders
    % for the static-pose match.  We add joint-centre markers separately.
    seg = @(c, lw) plot3(ax, nan, nan, nan, '-', 'Color', c, 'LineWidth', lw);
    h.shaft         = seg(grey,  3);
    h.l_forearm     = seg(skin,  4);
    h.l_upperarm    = seg(shirt, 4);
    h.l_shoulder    = seg(shirt, 5);
    h.r_forearm     = seg(skin,  4);
    h.r_upperarm    = seg(shirt, 4);
    h.r_shoulder    = seg(shirt, 5);
    h.shoulder_line = plot3(ax, nan, nan, nan, '-', 'Color', shirt, 'LineWidth', 5);
    h.spine_line    = plot3(ax, nan, nan, nan, '-', 'Color', [0.3 0.3 0.6], 'LineWidth', 3);
    h.l_grip_arm    = plot3(ax, nan, nan, nan, '-', 'Color', skin, 'LineWidth', 3);
    h.r_grip_arm    = plot3(ax, nan, nan, nan, '-', 'Color', skin, 'LineWidth', 3);

    h.joints = scatter3(ax, nan, nan, nan, 60, [0.2 0.2 0.2], 'filled', 'MarkerEdgeColor', 'k');
    h.joint_labels = gobjects(0);

    % Measured club path — full trace + impact-frame markers.
    h.meas_trace = plot3(ax, target.clubhead(:,1), target.clubhead(:,2), target.clubhead(:,3), ...
                         '-', 'Color', meas_c, 'LineWidth', 1.0);
    try, h.meas_trace.Color(4) = 0.4; end %#ok<TRYNC>

    h.meas_butt    = scatter3(ax, nan, nan, nan, 80, meas_c, 'filled', 'MarkerEdgeColor', 'k');
    h.meas_ch      = scatter3(ax, nan, nan, nan, 80, [0.84 0.15 0.16], 'filled', 'MarkerEdgeColor', 'k');
    h.meas_shaft   = plot3(ax, nan, nan, nan, '-', 'Color', meas_c, 'LineWidth', 2.5);
    h.error_butt   = plot3(ax, nan, nan, nan, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);
    h.error_ch     = plot3(ax, nan, nan, nan, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);

    h.title = title(ax, '');
end

%% =====================================================================
function local_lock_camera(ax, skel, target)
%LOCAL_LOCK_CAMERA  Set view limits from union of model + measured data.
    extras = {};
    for f = {'hip','torso','spine'}
        if isfield(skel, f{1}); extras{end+1} = skel.(f{1}); end %#ok<AGROW>
    end
    pts = [skel.butt; skel.mp; skel.ch; skel.lw; skel.le; skel.ls; ...
           skel.rw; skel.re; skel.rs; skel.hub; ...
           target.butt; target.clubhead; vertcat(extras{:})];
    pts = pts(all(~isnan(pts), 2), :);
    margin = 0.25;
    xlim(ax, [min(pts(:,1))-margin, max(pts(:,1))+margin]);
    ylim(ax, [min(pts(:,2))-margin, max(pts(:,2))+margin]);
    zlim(ax, [min(pts(:,3))-margin, max(pts(:,3))+margin]);
end

%% =====================================================================
function ctrl = local_build_controls(panel, fig) %#ok<INUSD>
    L = 0.08; W = 0.84;
    add_label = @(y, s) uicontrol('Parent', panel, 'Style', 'text', ...
        'Units','normalized','Position',[L y W 0.04], 'String', s, ...
        'BackgroundColor',[0.96 0.96 0.96], 'HorizontalAlignment','left', 'FontSize', 9);

    add_label(0.94, 'X offset (m)');
    ctrl.dx  = uicontrol('Parent', panel, 'Style','slider', 'Min',-0.5, 'Max',0.5, 'Value',0, ...
        'Units','normalized','Position',[L 0.91 W 0.03]);
    add_label(0.86, 'Y offset (m)');
    ctrl.dy  = uicontrol('Parent', panel, 'Style','slider', 'Min',-0.5, 'Max',0.5, 'Value',0, ...
        'Units','normalized','Position',[L 0.83 W 0.03]);
    add_label(0.78, 'Z offset (m)');
    ctrl.dz  = uicontrol('Parent', panel, 'Style','slider', 'Min',-0.5, 'Max',0.5, 'Value',0, ...
        'Units','normalized','Position',[L 0.75 W 0.03]);
    add_label(0.70, 'Yaw rotation (deg)');
    ctrl.yaw = uicontrol('Parent', panel, 'Style','slider', 'Min',-45, 'Max',45, 'Value',0, ...
        'Units','normalized','Position',[L 0.67 W 0.03]);

    ctrl.btn_reset = uicontrol('Parent', panel, 'Style','pushbutton', 'String','Reset', ...
        'Units','normalized','Position',[L 0.58 W 0.05]);
    ctrl.btn_save  = uicontrol('Parent', panel, 'Style','pushbutton', 'String','Save offset to .mat', ...
        'Units','normalized','Position',[L 0.51 W 0.05]);

    add_label(0.43, 'Camera presets');
    ctrl.btn_view_face = uicontrol('Parent', panel, 'Style','pushbutton', 'String','Face-on', ...
        'Units','normalized','Position',[L 0.37 W/2-0.01 0.05]);
    ctrl.btn_view_dtl  = uicontrol('Parent', panel, 'Style','pushbutton', 'String','Down-the-line', ...
        'Units','normalized','Position',[L+W/2+0.01 0.37 W/2-0.01 0.05]);
    ctrl.btn_view_top  = uicontrol('Parent', panel, 'Style','pushbutton', 'String','Top-down', ...
        'Units','normalized','Position',[L 0.30 W/2-0.01 0.05]);
    ctrl.btn_view_iso  = uicontrol('Parent', panel, 'Style','pushbutton', 'String','Isometric', ...
        'Units','normalized','Position',[L+W/2+0.01 0.30 W/2-0.01 0.05]);
end

%% =====================================================================
function local_on_slider(fig, name, value)
    s = getappdata(fig, 'state');
    s.(name) = value;
    setappdata(fig, 'state', s);
    local_redraw(fig);
end

%% =====================================================================
function local_reset(fig)
    s = getappdata(fig, 'state');
    s.dx = 0; s.dy = 0; s.dz = 0; s.yaw_deg = 0;
    set(s.controls.dx,  'Value', 0);
    set(s.controls.dy,  'Value', 0);
    set(s.controls.dz,  'Value', 0);
    set(s.controls.yaw, 'Value', 0);
    setappdata(fig, 'state', s);
    local_redraw(fig);
end

%% =====================================================================
function local_save_offset(fig)
    s = getappdata(fig, 'state');
    [file, path] = uiputfile({'*.mat', 'Offset MAT files (*.mat)'}, ...
        'Save measured-to-model offset', 'measured_to_model_offset.mat');
    if isequal(file, 0); return; end
    offset = struct( ...
        'dx', s.dx, 'dy', s.dy, 'dz', s.dz, ...
        'yaw_deg', s.yaw_deg, ...
        'transform_doc', 'p_model = R_z(yaw_deg) * p_meas + [dx;dy;dz]', ...
        'created', datestr(datetime('now'))); %#ok<DATST>
    save(fullfile(path, file), 'offset');
    fprintf('[plot_starting_position_match] saved offset to %s\n', fullfile(path, file));
end

%% =====================================================================
function local_redraw(fig)
    s = getappdata(fig, 'state');

    R = local_rot_z(s.yaw_deg);
    t = [s.dx; s.dy; s.dz];

    % --- Apply transform to measured data (only butt/clubhead at impact_idx
    %     and the clubhead trace for visualisation). ---------------------
    idx = max(1, min(s.impact_idx, size(s.target.butt, 1)));
    butt_meas = (R * s.target.butt(idx, :)')'    + t';
    ch_meas   = (R * s.target.clubhead(idx, :)')'+ t';
    trace     = (R * s.target.clubhead')'        + t';

    % --- Update model skeleton lines.  We tolerate NaN endpoints — any
    %     segment with a NaN simply renders nothing. -------------------
    K = s.skel;  H = s.handles;
    set_seg = @(h, p1, p2) set(h, 'XData', [p1(1) p2(1)], 'YData', [p1(2) p2(2)], 'ZData', [p1(3) p2(3)]);
    set_seg(H.shaft,      K.butt, K.ch);
    set_seg(H.l_forearm,  K.lw,   K.le);
    set_seg(H.l_upperarm, K.le,   K.ls);
    set_seg(H.l_shoulder, K.ls,   K.hub);
    set_seg(H.r_forearm,  K.rw,   K.re);
    set_seg(H.r_upperarm, K.re,   K.rs);
    set_seg(H.r_shoulder, K.rs,   K.hub);
    set_seg(H.shoulder_line, K.ls, K.rs);

    if isfield(K, 'hip') && isfield(K, 'spine')
        set_seg(H.spine_line, K.hip, K.spine);
    end
    % If elbows/shoulders are missing, fall back to a hands→mid-grip stub
    % so the visualisation isn't blank between the spine and the club.
    if any(isnan(K.le)) || any(isnan(K.ls))
        set_seg(H.l_grip_arm, K.lw, K.spine);
    else
        set_seg(H.l_grip_arm, [nan nan nan], [nan nan nan]);
    end
    if any(isnan(K.re)) || any(isnan(K.rs))
        set_seg(H.r_grip_arm, K.rw, K.spine);
    else
        set_seg(H.r_grip_arm, [nan nan nan], [nan nan nan]);
    end

    extras = [];
    for f = {'hip','torso','spine'}
        if isfield(K, f{1}) && all(~isnan(K.(f{1}))); extras = [extras; K.(f{1})]; end %#ok<AGROW>
    end
    pts = [K.butt; K.mp; K.ch; K.lw; K.le; K.ls; K.rw; K.re; K.rs; K.hub; extras];
    keep = all(~isnan(pts), 2);
    set(H.joints, 'XData', pts(keep,1), 'YData', pts(keep,2), 'ZData', pts(keep,3));

    % Joint text labels — recreate cheaply each draw.
    delete(H.joint_labels(isvalid(H.joint_labels)));
    labels = s.skel.joint_order;
    H.joint_labels = gobjects(numel(labels), 1);
    for k = 1:numel(labels)
        v = s.skel.(labels{k});
        if all(~isnan(v))
            H.joint_labels(k) = text(s.ax, v(1)+0.02, v(2)+0.02, v(3)+0.02, ...
                                     upper(labels{k}), 'FontSize', 8, 'Color', [0.2 0.2 0.2]);
        end
    end
    s.handles = H;
    setappdata(fig, 'state', s);

    % --- Update measured markers and shaft. --------------------------
    set(H.meas_butt, 'XData', butt_meas(1), 'YData', butt_meas(2), 'ZData', butt_meas(3));
    set(H.meas_ch,   'XData', ch_meas(1),   'YData', ch_meas(2),   'ZData', ch_meas(3));
    set(H.meas_shaft,'XData', [butt_meas(1) ch_meas(1)], ...
                     'YData', [butt_meas(2) ch_meas(2)], ...
                     'ZData', [butt_meas(3) ch_meas(3)]);
    set(H.meas_trace,'XData', trace(:,1), 'YData', trace(:,2), 'ZData', trace(:,3));

    % --- Error lines (sim → meas). -----------------------------------
    set(H.error_butt, 'XData', [K.butt(1) butt_meas(1)], ...
                      'YData', [K.butt(2) butt_meas(2)], ...
                      'ZData', [K.butt(3) butt_meas(3)]);
    set(H.error_ch,   'XData', [K.ch(1)   ch_meas(1)], ...
                      'YData', [K.ch(2)   ch_meas(2)], ...
                      'ZData', [K.ch(3)   ch_meas(3)]);

    % --- Live RMS errors (mm). ---------------------------------------
    err_butt_mm = 1000 * norm(butt_meas - K.butt);
    err_ch_mm   = 1000 * norm(ch_meas   - K.ch);
    msg = sprintf(['Offset  dx=%+.3f m  dy=%+.3f m  dz=%+.3f m  yaw=%+.1f°\n', ...
                   'Butt error: %6.1f mm     Clubhead error: %6.1f mm     impact_idx=%d  (t_meas=%.3fs)'], ...
                   s.dx, s.dy, s.dz, s.yaw_deg, err_butt_mm, err_ch_mm, ...
                   idx, s.target.time(idx));
    set(s.rmse_text, 'String', msg);
    set(H.title, 'String', sprintf('Model skeleton (Impact pose) vs measured club path — butt %.1f mm | head %.1f mm', ...
                                   err_butt_mm, err_ch_mm));

    drawnow limitrate;
end

%% =====================================================================
function R = local_rot_z(deg)
    c = cosd(deg); sn = sind(deg);
    R = [c -sn 0; sn c 0; 0 0 1];
end
