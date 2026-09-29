function out = demo_starting_position_match(varargin)
%DEMO_STARTING_POSITION_MATCH  End-to-end demo of the starting-position
%   matching pipeline.  Runs:
%     1. Model impact-pose loader (full skeleton from CombinedSignalBus).
%     2. Wiffle ProV1 measured-club-path loader.
%     3. Auto-align solver — finds (dx, dy, dz, yaw) that puts the
%        measured ball location (clubhead at impact frame) onto the
%        model's clubhead, jointly minimising the butt residual.
%     4. Renders a four-panel comparison figure to disk:
%          (a) before alignment (model + raw measured)
%          (b) after alignment  (model + transformed measured)
%          (c) close-up at impact frame
%          (d) full-swing overlay with skeleton at impact pose
%     5. Renders a 60-frame animated MP4 of the full measured swing
%        playing through the aligned coordinate frame, with the model
%        skeleton fixed at impact pose.
%
%   OUT = DEMO_STARTING_POSITION_MATCH() saves artefacts under
%   matlab/output/starting_position_match/<timestamp>/ and returns the
%   alignment + path metadata as a struct.
%
%   Name/value args:
%     'sheet'     measured-data sheet (default 'TW_ProV1')
%     'xlsx'      measured-data file path (auto-resolves to the Wiffle file)
%     'animate'   true | false  (default true)
%     'frames'    animation frame count (default 60)

    p = inputParser();
    addParameter(p, 'sheet', 'TW_ProV1', @(s) ischar(s) || isstring(s));
    addParameter(p, 'xlsx', '', @(s) ischar(s) || isstring(s));
    addParameter(p, 'animate', true, @islogical);
    addParameter(p, 'frames', 60, @(n) isnumeric(n) && n > 0);
    addParameter(p, 'mode', 'grip_pose', @(s) ischar(s) || isstring(s));
    parse(p, varargin{:});
    args = p.Results;

    here     = fileparts(mfilename('fullpath'));
    shared   = fileparts(here);
    mm_root  = fileparts(shared);
    mat_root = fileparts(mm_root);
    addpath(shared);
    addpath(genpath(fullfile(mat_root, 'src')));

    if isempty(char(args.xlsx))
        args.xlsx = fullfile(mat_root, 'src', 'apps', 'golf_gui', ...
            'Motion Capture Plotter', 'Wiffle_ProV1_club_3D_data.xlsx');
    end

    timestamp = datestr(datetime('now'), 'yyyymmdd_HHMMSS'); %#ok<DATST>
    out_dir   = fullfile(mat_root, 'output', 'starting_position_match', timestamp);
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    fprintf('=== demo_starting_position_match ===\n');
    fprintf('  output dir: %s\n', out_dir);

    % --- 1. Load skeleton + measured ----------------------------------
    fprintf('\n[1/5] loading model impact pose...\n');
    skel = load_impact_starting_position(struct('verbose', false));

    fprintf('[2/5] loading measured club path...\n');
    % For the starting-position demo we want the *whole* swing visible,
    % not just the 300-ms motion-matching window.  Override the loader's
    % alignment options to keep ~1 s before and ~1 s after impact.
    align_opts = default_align_options(struct( ...
        "pre_impact_s",      1.5, ...
        "post_impact_s",     1.0, ...
        "expected_impact_s", 1.5, ...
        "verbosity",         "Normal"));
    target = load_club_target_excel(string(args.xlsx), string(args.sheet), align_opts);
    fprintf('       %d measured frames; impact at idx %d (t=%.3fs)\n', ...
            numel(target.time), double(target.impact_idx), ...
            target.time(double(target.impact_idx)));
    if isfield(target, 'events')
        ev = target.events;
        fprintf('       events: A=%g T=%g I=%g F=%g  CHS=%.1f mph\n', ...
                ev.A_sample, ev.T_sample, ev.I_sample, ev.F_sample, ev.CHS_mph);
    end

    % --- 2. Auto-align ------------------------------------------------
    fprintf('[3/5] solving for ball-position alignment (closed-form)...\n');
    align = align_measured_to_model(skel, target, struct('mode', char(args.mode)));
    fprintf('       mode   : %s\n', align.mode);
    fprintf('       initial:  butt %7.1f mm   clubhead %7.1f mm\n', ...
            align.initial_butt_error_mm, align.initial_clubhead_error_mm);
    fprintf('       solved :  butt %7.1f mm   clubhead %7.1f mm\n', ...
            align.butt_error_mm, align.clubhead_error_mm);
    fprintf('       offset :  dx=%+.3f m  dy=%+.3f m  dz=%+.3f m\n', ...
            align.dx, align.dy, align.dz);
    fprintf('       rotate :  roll=%+.2f°  pitch=%+.2f°  yaw=%+.2f°\n', ...
            align.roll_deg, align.pitch_deg, align.yaw_deg);
    if isfield(align, 'measured_shaft_length_m')
        fprintf('       shafts :  model=%.3fm  measured=%.3fm  (Δ=%.3fm)\n', ...
                align.model_shaft_length_m, align.measured_shaft_length_m, ...
                align.measured_shaft_length_m - align.model_shaft_length_m);
        if abs(align.measured_shaft_length_m - align.model_shaft_length_m) > 0.20
            fprintf(['       NOTE   : measured ''butt'' column is a body-mounted tracker,\n', ...
                     '                not the grip butt — explains the %.2fm shaft-length gap.\n', ...
                     '                Alignment matches the BALL (clubhead) and shaft direction\n', ...
                     '                exactly; the butt residual is expected.\n'], ...
                     align.measured_shaft_length_m - align.model_shaft_length_m);
        end
    end

    % --- 3. Build aligned target (apply transform to entire trajectory)
    aligned = local_apply_transform(target, align);

    % --- 4. Composite four-panel comparison figure --------------------
    fprintf('[4/5] rendering before/after comparison figure...\n');
    fig = local_render_quad_panel(skel, target, aligned, align);
    png_path = fullfile(out_dir, 'starting_position_compare.png');
    exportgraphics(fig, png_path, 'Resolution', 200);
    close(fig);
    fprintf('       saved %s\n', png_path);

    % Save offset MAT for downstream use.
    offset = struct('dx', align.dx, 'dy', align.dy, 'dz', align.dz, ...
                    'yaw_deg', align.yaw_deg, ...
                    'sheet', string(args.sheet), ...
                    'xlsx_sha256', target.source.sha256, ...
                    'created', datestr(datetime('now'))); %#ok<DATST>
    save(fullfile(out_dir, 'measured_to_model_offset.mat'), 'offset');

    % --- 5. Animated swing in the aligned frame -----------------------
    mp4_path = '';
    if args.animate
        fprintf('[5/5] rendering animation (%d frames) ...\n', args.frames);
        try
            mp4_path = local_render_animation(skel, aligned, align, ...
                                              args.frames, out_dir);
            fprintf('       saved %s\n', mp4_path);
        catch ME
            fprintf('       animation failed: %s\n', ME.message);
        end
    end

    % --- 6. Text summary report --------------------------------------
    report_path = fullfile(out_dir, 'report.txt');
    local_write_report(report_path, skel, target, align, mp4_path, png_path);
    fprintf('\n=== done ===\n');
    fprintf('see %s\n', out_dir);

    out = struct( ...
        'skel', skel, 'target', target, 'aligned', aligned, ...
        'align', align, 'output_dir', string(out_dir), ...
        'png_path', string(png_path), 'mp4_path', string(mp4_path), ...
        'report_path', string(report_path));
end

%% =====================================================================
function aligned = local_apply_transform(target, align)
    R = align.R; t = align.t;
    if isfield(align, 'scale'); s = align.scale; else; s = 1.0; end
    aligned = target;
    if isfield(target, 'grip')
        aligned.grip = (s * R * target.grip')' + t';
    end
    aligned.butt     = (s * R * target.butt')' + t';
    aligned.clubhead = (s * R * target.clubhead')' + t';
end

%% =====================================================================
function fig = local_render_quad_panel(skel, target, aligned, align)
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1600 900]);

    % Panel A: before
    ax = subplot(2, 2, 1, 'Parent', fig); local_plot_axes(ax);
    title(ax, sprintf('(a) BEFORE  butt %.0fmm  head %.0fmm', ...
        align.initial_butt_error_mm, align.initial_clubhead_error_mm));
    local_draw_skeleton(ax, skel);
    local_draw_measured(ax, target, double(target.impact_idx), [31 119 180]/255);
    local_lock(ax, skel, target);

    % Panel B: after
    ax = subplot(2, 2, 2, 'Parent', fig); local_plot_axes(ax);
    title(ax, sprintf('(b) AFTER  butt %.1fmm  head %.1fmm', ...
        align.butt_error_mm, align.clubhead_error_mm));
    local_draw_skeleton(ax, skel);
    local_draw_measured(ax, aligned, double(aligned.impact_idx), [44 160 44]/255);
    local_lock(ax, skel, aligned);

    % Panel C: close-up at impact (zoom on grip — the matched anchor)
    ax = subplot(2, 2, 3, 'Parent', fig); local_plot_axes(ax);
    if strcmpi(string(align.mode), "grip_pose")
        title(ax, '(c) close-up at impact: GRIP is the anchor (rigid contact)');
    else
        title(ax, '(c) close-up at impact (ball location)');
    end
    local_draw_skeleton(ax, skel);
    local_draw_measured(ax, aligned, double(aligned.impact_idx), [44 160 44]/255);
    grip_pt = aligned.grip(double(aligned.impact_idx), :);
    if isfield(aligned, 'grip')
        plot3(ax, grip_pt(1), grip_pt(2), grip_pt(3), 'pk', ...
              'MarkerFaceColor', [0.85 0.65 0.13], 'MarkerSize', 18);
        text(ax, grip_pt(1)+0.05, grip_pt(2), grip_pt(3)+0.05, 'GRIP', ...
             'FontSize', 9, 'FontWeight', 'bold');
    end
    impact_pt = aligned.clubhead(double(aligned.impact_idx), :);
    plot3(ax, impact_pt(1), impact_pt(2), impact_pt(3), 'pk', ...
          'MarkerFaceColor', [1 0.7 0], 'MarkerSize', 14);
    text(ax, impact_pt(1)+0.05, impact_pt(2), impact_pt(3)+0.05, 'BALL', ...
         'FontSize', 9, 'FontWeight', 'bold');
    R = 0.5;
    centre = (grip_pt + impact_pt) / 2;
    xlim(ax, centre(1) + [-R R]);
    ylim(ax, centre(2) + [-R R]);
    zlim(ax, centre(3) + [-R R]);

    % Panel D: full overlay with measured swing trace + event markers
    ax = subplot(2, 2, 4, 'Parent', fig); local_plot_axes(ax);
    title(ax, '(d) full swing trace + A/T/I/F event markers');
    local_draw_skeleton(ax, skel);
    plot3(ax, aligned.clubhead(:, 1), aligned.clubhead(:, 2), aligned.clubhead(:, 3), ...
          '-', 'Color', [44 160 44]/255, 'LineWidth', 1.2);
    plot3(ax, aligned.butt(:, 1), aligned.butt(:, 2), aligned.butt(:, 3), ...
          '-', 'Color', [44 160 44]/255 .* 0.7, 'LineWidth', 1.0);
    local_overlay_event_markers(ax, aligned);
    local_lock(ax, skel, aligned);

    sgtitle(fig, ...
        sprintf('Starting-position match  —  offset dx=%+.3fm  dy=%+.3fm  dz=%+.3fm  yaw=%+.1f°', ...
                align.dx, align.dy, align.dz, align.yaw_deg), ...
        'FontSize', 12, 'FontWeight', 'bold');
end

%% =====================================================================
function local_plot_axes(ax)
    set(ax, 'Color', 'w', 'GridColor', [0.7 0.7 0.7], 'Box', 'on');
    hold(ax, 'on'); grid(ax, 'on'); axis(ax, 'equal');
    xlabel(ax, 'X (m)'); ylabel(ax, 'Y (m)'); zlabel(ax, 'Z (m)');
    view(ax, [-45 22]);
end

%% =====================================================================
function local_draw_skeleton(ax, K)
    skin   = [1 0.8 0.6];
    shirt  = [0.6 0.8 1];
    grey   = [0.5 0.5 0.5];
    seg = @(p1, p2, c, lw) plot3(ax, [p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], ...
                                  '-', 'Color', c, 'LineWidth', lw);
    seg(K.butt, K.ch,  grey,  3);
    seg(K.lw, K.le,    skin,  4);
    seg(K.le, K.ls,    shirt, 4);
    seg(K.ls, K.hub,   shirt, 4);
    seg(K.rw, K.re,    skin,  4);
    seg(K.re, K.rs,    shirt, 4);
    seg(K.rs, K.hub,   shirt, 4);
    seg(K.ls, K.rs,    shirt, 5);
    if isfield(K, 'spine'); seg(K.hip, K.spine, [0.3 0.3 0.6], 3); end
    if isfield(K, 'hub') && isfield(K, 'spine'); seg(K.spine, K.hub, [0.3 0.3 0.6], 3); end

    pts = [K.butt; K.mp; K.ch; K.lw; K.le; K.ls; K.rw; K.re; K.rs; K.hub];
    if isfield(K, 'hip');   pts = [pts; K.hip];   end
    if isfield(K, 'spine'); pts = [pts; K.spine]; end
    pts = pts(all(~isnan(pts), 2), :);
    scatter3(ax, pts(:,1), pts(:,2), pts(:,3), 50, 'k', 'filled');
end

%% =====================================================================
function local_draw_measured(ax, target, impact_idx, color)
    plot3(ax, target.clubhead(:, 1), target.clubhead(:, 2), target.clubhead(:, 3), ...
          '-', 'Color', [color 0.4], 'LineWidth', 1.0);
    bm = target.butt(impact_idx, :);
    hm = target.clubhead(impact_idx, :);
    scatter3(ax, bm(1), bm(2), bm(3), 80, color, 'filled', 'MarkerEdgeColor', 'k');
    scatter3(ax, hm(1), hm(2), hm(3), 80, [0.84 0.15 0.16], 'filled', 'MarkerEdgeColor', 'k');
    plot3(ax, [bm(1) hm(1)], [bm(2) hm(2)], [bm(3) hm(3)], '-', 'Color', color, 'LineWidth', 2.5);
end

%% =====================================================================
function local_overlay_event_markers(ax, aligned)
%LOCAL_OVERLAY_EVENT_MARKERS  Drop A/T/I/F dots on the clubhead trace
%   at the documented event samples.  Sample numbers come from the row-1
%   header of the Wiffle xlsx; we map them to the current aligned grid
%   by the time-from-impact relationship.
    if ~isfield(aligned, 'events'); return; end
    ev = aligned.events;
    if all(structfun(@isnan, ev)); return; end
    impact_idx = double(aligned.impact_idx);
    t_impact = aligned.time(impact_idx);
    fps = 240;   % Wiffle sampling rate
    labels = {'A','T','I','F'};
    fields = {'A_sample','T_sample','I_sample','F_sample'};
    colors = [0.2 0.6 0.2; 0.85 0.65 0.13; 0.85 0.10 0.10; 0.20 0.20 0.65];
    for k = 1:numel(labels)
        s = ev.(fields{k});
        if isnan(s); continue; end
        % seconds from impact in the original 240Hz timeline
        dt = (s - ev.I_sample) / fps;
        t_target = t_impact + dt;
        if t_target < aligned.time(1) || t_target > aligned.time(end); continue; end
        [~, ix] = min(abs(aligned.time - t_target));
        p = aligned.clubhead(ix, :);
        scatter3(ax, p(1), p(2), p(3), 90, colors(k, :), 'filled', ...
                 'MarkerEdgeColor', 'k');
        text(ax, p(1) + 0.04, p(2), p(3) + 0.04, labels{k}, ...
             'FontSize', 10, 'FontWeight', 'bold', 'Color', colors(k, :));
    end
end

%% =====================================================================
function local_lock(ax, skel, target)
    pts = [skel.butt; skel.mp; skel.ch; skel.lw; skel.le; skel.ls; ...
           skel.rw; skel.re; skel.rs; skel.hub; ...
           target.butt; target.clubhead];
    if isfield(skel, 'hip');   pts = [pts; skel.hip];   end
    if isfield(skel, 'spine'); pts = [pts; skel.spine]; end
    pts = pts(all(~isnan(pts), 2), :);
    margin = 0.25;
    xlim(ax, [min(pts(:,1))-margin, max(pts(:,1))+margin]);
    ylim(ax, [min(pts(:,2))-margin, max(pts(:,2))+margin]);
    zlim(ax, [min(pts(:,3))-margin, max(pts(:,3))+margin]);
end

%% =====================================================================
function mp4 = local_render_animation(skel, aligned, align, n_frames, out_dir)
    mp4 = fullfile(out_dir, 'starting_position_animation.mp4');
    vw = VideoWriter(mp4, 'MPEG-4');
    vw.FrameRate = 24;
    vw.Quality   = 90;
    open(vw);

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1100 800]);
    ax = axes('Parent', fig); local_plot_axes(ax);
    title(ax, 'Measured swing in model-aligned frame  (model skeleton at IMPACT POSE)');
    local_draw_skeleton(ax, skel);
    plot3(ax, aligned.clubhead(:, 1), aligned.clubhead(:, 2), aligned.clubhead(:, 3), ...
          '-', 'Color', [44 160 44 0.3]/255, 'LineWidth', 0.8); %#ok<NBRAK>
    local_lock(ax, skel, aligned);

    moving_butt = scatter3(ax, nan, nan, nan, 90, [44 160 44]/255, 'filled', 'MarkerEdgeColor', 'k');
    moving_head = scatter3(ax, nan, nan, nan, 90, [0.84 0.15 0.16], 'filled', 'MarkerEdgeColor', 'k');
    moving_shaft = plot3(ax, nan, nan, nan, '-', 'Color', [44 160 44]/255, 'LineWidth', 2.5);
    msg = annotation(fig, 'textbox', [0.02 0.94 0.96 0.05], 'String', '', ...
                     'EdgeColor', 'none', 'FontSize', 10, 'FontWeight', 'bold');

    N = size(aligned.clubhead, 1);
    idxs = round(linspace(1, N, n_frames));
    for k = 1:numel(idxs)
        i = idxs(k);
        bm = aligned.butt(i, :); hm = aligned.clubhead(i, :);
        set(moving_butt, 'XData', bm(1), 'YData', bm(2), 'ZData', bm(3));
        set(moving_head, 'XData', hm(1), 'YData', hm(2), 'ZData', hm(3));
        set(moving_shaft, 'XData', [bm(1) hm(1)], 'YData', [bm(2) hm(2)], 'ZData', [bm(3) hm(3)]);
        set(msg, 'String', sprintf( ...
            'frame %d/%d   t=%.3fs   alignment offset:  dx=%+.3fm  dy=%+.3fm  dz=%+.3fm  yaw=%+.1f°', ...
            i, N, aligned.time(i), align.dx, align.dy, align.dz, align.yaw_deg));
        drawnow limitrate;
        writeVideo(vw, getframe(fig));
    end
    close(vw);
    close(fig);
end

%% =====================================================================
function local_write_report(path, skel, target, align, mp4_path, png_path)
    fid = fopen(path, 'w');
    if fid < 0; return; end
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, 'Starting-position match report\n');
    fprintf(fid, 'Generated: %s\n\n', datestr(datetime('now'))); %#ok<DATST>

    fprintf(fid, '== Model impact pose (%s) ==\n', skel.input_file);
    fns = skel.joint_order;
    for k = 1:numel(fns)
        n = fns{k}; v = skel.(n);
        if any(isnan(v))
            fprintf(fid, '  %-6s  <missing>\n', n);
        else
            fprintf(fid, '  %-6s  [% 8.4f % 8.4f % 8.4f]\n', n, v(1), v(2), v(3));
        end
    end

    fprintf(fid, '\n== Measured ==\n');
    fprintf(fid, '  source:      %s\n',  target.source.filename);
    fprintf(fid, '  sheet:       %s\n',  target.source.trial_id);
    fprintf(fid, '  frames:      %d\n',  numel(target.time));
    fprintf(fid, '  impact idx:  %d  (t=%.4fs)\n',  double(target.impact_idx), ...
                                                      target.time(double(target.impact_idx)));
    if isfield(target, 'events')
        ev = target.events;
        fprintf(fid, '  events    :  A=%g  T=%g  I=%g  F=%g  CHS=%.1f mph\n', ...
                ev.A_sample, ev.T_sample, ev.I_sample, ev.F_sample, ev.CHS_mph);
    end
    fprintf(fid, '  ball pos@imp [%+8.4f %+8.4f %+8.4f]  (raw mocap frame)\n', ...
                 target.clubhead(double(target.impact_idx), :));
    if isfield(align, 'measured_shaft_length_m')
        fprintf(fid, '  shaft length: model=%.4fm  measured=%.4fm  (Δ=%+.4fm)\n', ...
                align.model_shaft_length_m, align.measured_shaft_length_m, ...
                align.measured_shaft_length_m - align.model_shaft_length_m);
    end
    if isfield(align, 'scale')
        fprintf(fid, '  uniform scale applied to absorb shaft-length gap: %.4f\n', align.scale);
    end

    fprintf(fid, '\n== Alignment ==\n');
    fprintf(fid, '  initial:    butt %7.1f mm   clubhead %7.1f mm\n', ...
            align.initial_butt_error_mm, align.initial_clubhead_error_mm);
    fprintf(fid, '  solved:     butt %7.1f mm   clubhead %7.1f mm   (%d iters)\n', ...
            align.butt_error_mm, align.clubhead_error_mm, align.iters);
    fprintf(fid, '  transform:  p_model = R_z(%+.3f°) * p_meas + [%+.4f; %+.4f; %+.4f] m\n', ...
                 align.yaw_deg, align.dx, align.dy, align.dz);
    fprintf(fid, '  ball aligned: [%+8.4f %+8.4f %+8.4f]   (model frame)\n', ...
                 align.clubhead_aligned);

    fprintf(fid, '\n== Artefacts ==\n');
    fprintf(fid, '  comparison png: %s\n', png_path);
    if ~isempty(mp4_path)
        fprintf(fid, '  animation:      %s\n', mp4_path);
    end
end
