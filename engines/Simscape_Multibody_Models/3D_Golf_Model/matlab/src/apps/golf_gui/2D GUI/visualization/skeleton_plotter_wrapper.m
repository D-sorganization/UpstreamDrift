function skeleton_plotter_wrapper(BASEQ, ZTCFQ, DELTAQ)
% SKELETON_PLOTTER_WRAPPER - Wrapper function to launch skeleton plotter from GUI
%
% Inputs:
%   BASEQ - Base data table (Q-spaced)
%   ZTCFQ - ZTCF data table (Q-spaced)
%   DELTAQ - DELTA data table (Q-spaced)
%
% This function launches the skeleton plotter with the provided data
% and provides integration with the main GUI

    % Check if data is provided
    if nargin < 3
        error('Skeleton plotter requires BASEQ, ZTCFQ, and DELTAQ data tables');
    end

    % Validate data structure
    if ~istable(BASEQ) || ~istable(ZTCFQ) || ~istable(DELTAQ)
        error('All inputs must be tables');
    end

    % Check for required columns in BASEQ
    required_columns = {'Buttx', 'Butty', 'Buttz', 'CHx', 'CHy', 'CHz', ...
                       'MPx', 'MPy', 'MPz', 'LWx', 'LWy', 'LWz', ...
                       'LEx', 'LEy', 'LEz', 'LSx', 'LSy', 'LSz', ...
                       'RWx', 'RWy', 'RWz', 'REx', 'REy', 'REz', ...
                       'RSx', 'RSy', 'RSz', 'HUBx', 'HUBy', 'HUBz'};

    missing_columns = setdiff(required_columns, BASEQ.Properties.VariableNames);
    if ~isempty(missing_columns)
        warning('Missing columns in BASEQ: %s', strjoin(missing_columns, ', '));
    end

    % Check for force and torque data
    force_columns = {'TotalHandForceGlobal', 'EquivalentMidpointCoupleGlobal'};
    missing_forces = setdiff(force_columns, BASEQ.Properties.VariableNames);
    if ~isempty(missing_forces)
        warning('Missing force/torque columns: %s', strjoin(missing_forces, ', '));
    end

    fprintf('🦴 Launching Skeleton Plotter...\n');
    fprintf('   BASEQ data points: %d\n', height(BASEQ));
    fprintf('   ZTCFQ data points: %d\n', height(ZTCFQ));
    fprintf('   DELTAQ data points: %d\n', height(DELTAQ));

    try
        % Launch the skeleton plotter
        SkeletonPlotter(BASEQ, ZTCFQ, DELTAQ);

        fprintf('✅ Skeleton Plotter launched successfully\n');

    catch ME
        fprintf('❌ Error launching Skeleton Plotter: %s\n', ME.message);
        fprintf('📍 Error location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        rethrow(ME);
    end

end

function SkeletonPlotter(BASEQ, ZTCFQ, DELTAQ)
% SKELETONPLOTTER - Advanced 3D Golf Swing Visualizer
%
% This is the main skeleton plotter function that provides:
% - 3D visualization of golf swing
% - Interactive playback controls
% - Force and torque vector display
% - Multiple dataset comparison
% - Zoom and camera controls
% - Recording capabilities

    % Save initial workspace variables
    vars_before = who;

    %% === 1. Settings and Constants ===
    shirt_color = [0.6, 0.8, 1];
    skin_color = [1, 0.8, 0.6];
    figure_background_color = [0.9, 1, 0.9];
    axes_background_color = [1, 1, 0.8];
    panel_background = [0.8, 1, 0.8];
    text_background = [1, 1, 1];
    record_idle_color = [1.0 0.6 0.0];  % Orange (idle)
    record_active_color = [1.0 0.4 0.4]; % Red (recording)
    font_size = 10;

    % Sizes
    inches_to_meters = 0.0254;
    clubhead_diameter = 4 * inches_to_meters;
    shaft_diameter = 0.5 * inches_to_meters;
    forearm_diameter = 3 * inches_to_meters;
    upperarm_diameter = 4 * inches_to_meters;
    shoulderneck_diameter = 5 * inches_to_meters;

    % Datasets
    datasets = {'BASE', BASEQ; 'ZTCF', ZTCFQ; 'DELTA', DELTAQ};
    colors_force = {[1 0 0], [0 0 1], [1 0.5 0]};
    colors_torque = {[0.5 0 0.5], [0 1 1], [1 0 1]};

    % Frames
    num_frames = length(BASEQ.Buttx);

    % Scaling
    Max_force_mag = max(vecnorm(BASEQ.TotalHandForceGlobal,2,2));
    Max_torque_mag = max(vecnorm(BASEQ.EquivalentMidpointCoupleGlobal,2,2));

    %% === 2. Create Main Figure ===
    fig = figure('Name', 'Golf Swing Skeleton Plotter', ...
                 'NumberTitle', 'off', ...
                 'Color', figure_background_color, ...
                 'Position', [100, 100, 1400, 800]);

    %% === 3. Create 3D Axes ===
    handles.ax = axes('Parent', fig, ...
                      'Color', axes_background_color, ...
                      'Position', [0.25 0.05 0.6 0.9]);
    axis(handles.ax, 'equal');
    grid(handles.ax, 'on');
    xlabel(handles.ax, 'X (m)');
    ylabel(handles.ax, 'Y (m)');
    zlabel(handles.ax, 'Z (m)');
    hold(handles.ax, 'on');
    view(handles.ax, 3);

    % Set Plot Limits based on BASEQ
    all_x = [BASEQ.Buttx; BASEQ.CHx; BASEQ.MPx; BASEQ.LWx; BASEQ.LEx; BASEQ.LSx; BASEQ.RWx; BASEQ.REx; BASEQ.RSx; BASEQ.HUBx];
    all_y = [BASEQ.Butty; BASEQ.CHy; BASEQ.MPy; BASEQ.LWy; BASEQ.LEy; BASEQ.LSy; BASEQ.RWy; BASEQ.REy; BASEQ.RSy; BASEQ.HUBy];
    all_z = [BASEQ.Buttz; BASEQ.CHz; BASEQ.MPz; BASEQ.LWz; BASEQ.LEz; BASEQ.LSz; BASEQ.RWz; BASEQ.REz; BASEQ.RSz; BASEQ.HUBz];
    margin = 0.3;
    xlim(handles.ax, [min(all_x)-margin, max(all_x)+margin]);
    ylim(handles.ax, [min(all_y)-margin, max(all_y)+margin]);
    zlim(handles.ax, [min(all_z)-margin, max(all_z)+margin]);
    axis manual

    camlight(handles.ax, 'headlight');
    lighting(handles.ax, 'gouraud');
    material(handles.ax, 'dull');
    shading(handles.ax, 'interp');

    %% === 4. Create GUI Panels and Controls ===

    % --- 4.1 Checkbox Panel (Left Center) ---
    handles.panel_checkboxes = uipanel('Parent', fig, ...
        'Title', 'Segments and Vectors', ...
        'FontSize', font_size, ...
        'BackgroundColor', panel_background, ...
        'Units', 'normalized', 'Position', [0.01 0.4 0.22 0.55]);

    checkbox_names = { ...
        'Force BASE', 'Force ZTCF', 'Force DELTA', ...
        'Torque BASE', 'Torque ZTCF', 'Torque DELTA', ...
        'Shaft', 'Face Normal', ...
        'Left Forearm', 'Left Upper Arm', 'Left Shoulder-Neck', ...
        'Right Forearm', 'Right Upper Arm', 'Right Shoulder-Neck'};

    handles.checkbox_list = gobjects(length(checkbox_names),1);

    for k = 1:length(checkbox_names)
        handles.checkbox_list(k) = uicontrol('Parent', handles.panel_checkboxes, ...
            'Style', 'checkbox', ...
            'String', checkbox_names{k}, ...
            'Units', 'normalized', ...
            'Position', [0.05, 1-k*0.065, 0.9, 0.05], ...
            'BackgroundColor', panel_background, ...
            'FontSize', 9, ...
            'Value', 1);
    end

    % --- 4.2 Sliders + Play/Pause Button Panel (Bottom Left) ---
    handles.panel_sliders = uipanel('Parent', fig, ...
        'Title', 'Playback and Scaling', ...
        'FontSize', font_size, ...
        'BackgroundColor', panel_background, ...
        'Units', 'normalized', 'Position', [0.01 0.05 0.22 0.35]);

    % Playback Speed Slider
    handles.speed_slider = uicontrol('Parent', handles.panel_sliders, 'Style', 'slider', ...
        'Min', 0.1, 'Max', 3, 'Value', 1, ...
        'Units', 'normalized', 'Position', [0.05 0.8 0.9 0.08], ...
        'BackgroundColor', [1 1 1], 'FontSize', 9);
    uicontrol('Parent', handles.panel_sliders, 'Style', 'text', 'String', 'Playback Speed', ...
              'Units', 'normalized', 'Position', [0.05 0.88 0.9 0.08], ...
              'BackgroundColor', panel_background, 'FontSize', 9, 'HorizontalAlignment', 'center');

    % Vector Scale Slider
    handles.scale_slider = uicontrol('Parent', handles.panel_sliders, 'Style', 'slider', ...
        'Min', 0.1, 'Max', 9, 'Value', 1, ...
        'Units', 'normalized', 'Position', [0.05 0.6 0.9 0.08], ...
        'BackgroundColor', [1 1 1], 'FontSize', 9);
    uicontrol('Parent', handles.panel_sliders, 'Style', 'text', 'String', 'Vector Scale', ...
              'Units', 'normalized', 'Position', [0.05 0.68 0.9 0.08], ...
              'BackgroundColor', panel_background, 'FontSize', 9, 'HorizontalAlignment', 'center');

    % Frame Slider
    handles.slider = uicontrol('Parent', handles.panel_sliders, 'Style', 'slider', ...
        'Min', 1, 'Max', num_frames, 'Value', 1, ...
        'Units', 'normalized', 'Position', [0.05 0.4 0.9 0.08], ...
        'BackgroundColor', [1 1 1], 'FontSize', 9, ...
        'SliderStep', [1/(num_frames-1) 10/(num_frames-1)], ...
        'Callback', @updatePlot);
    uicontrol('Parent', handles.panel_sliders, 'Style', 'text', 'String', 'Frame', ...
              'Units', 'normalized', 'Position', [0.05 0.48 0.9 0.08], ...
              'BackgroundColor', panel_background, 'FontSize', 9, 'HorizontalAlignment', 'center');

    % Play/Pause Button
    handles.play_pause_button = uicontrol('Parent', handles.panel_sliders, 'Style', 'togglebutton', ...
        'String', 'Play', ...
        'Units', 'normalized', 'Position', [0.25 0.1 0.5 0.2], ...
        'FontSize', 10, ...
        'BackgroundColor', [0.4 0.8 0.4], ...
        'Callback', @togglePlayPause);

    % --- 4.3 Zoom Panel (Top Left) ---
    handles.panel_zoom = uipanel('Parent', fig, ...
        'Title', 'Zoom', ...
        'FontSize', font_size, ...
        'BackgroundColor', panel_background, ...
        'Units', 'normalized', 'Position', [0.01 0.93 0.22 0.05]);

    % Zoom Slider
    handles.zoom_slider = uicontrol('Parent', handles.panel_zoom, 'Style', 'slider', ...
        'Min', 0.5, 'Max', 2.0, 'Value', 1, ...
        'Units', 'normalized', 'Position', [0.05 0.2 0.9 0.6], ...
        'BackgroundColor', [1 1 1], 'FontSize', 9, ...
        'Callback', @(src,~) updateZoom(src.Value));

    % --- 4.4 Record Button (Bottom Right) ---
    handles.record_button = uicontrol('Parent', fig, 'Style', 'togglebutton', ...
        'String', 'Record', ...
        'Units', 'normalized', 'Position', [0.86 0.01 0.12 0.04], ...
        'BackgroundColor', record_idle_color, ...
        'FontSize', 10, ...
        'Callback', @toggleRecord);

    % --- 4.5 Legend Panel (Right Center) ---
    handles.panel_legend = uipanel('Parent', fig, ...
        'Title', 'Legend', ...
        'FontSize', font_size, ...
        'BackgroundColor', [1 1 1], ...
        'Units', 'normalized', 'Position', [0.86 0.35 0.12 0.3]);

    legend_entries = {
        'BASE (Force)', [1 0 0];
        'ZTCF (Force)', [0 0 1];
        'DELTA (Force)', [1 0.5 0];
        'BASE (Torque)', [0.5 0 0.5];
        'ZTCF (Torque)', [0 1 1];
        'DELTA (Torque)', [1 0 1];
    };

    for k = 1:size(legend_entries,1)
        uicontrol('Parent', handles.panel_legend, 'Style', 'text', ...
            'String', legend_entries{k,1}, ...
            'Units', 'normalized', ...
            'Position', [0.05, 1-k*0.15, 0.9, 0.1], ...
            'BackgroundColor', legend_entries{k,2}, ...
            'FontSize', 8, 'HorizontalAlignment', 'center');
    end

    %% === 5. Initialize Plot Handles ===
    handles.plot_handles = struct();
    handles.plot_handles.shaft = [];
    handles.plot_handles.face_normal = [];
    handles.plot_handles.left_forearm = [];
    handles.plot_handles.left_upperarm = [];
    handles.plot_handles.left_shoulderneck = [];
    handles.plot_handles.right_forearm = [];
    handles.plot_handles.right_upperarm = [];
    handles.plot_handles.right_shoulderneck = [];
    handles.plot_handles.forces = [];
    handles.plot_handles.torques = [];

    %% === 6. Store Data and Handles ===
    handles.BASEQ = BASEQ;
    handles.ZTCFQ = ZTCFQ;
    handles.DELTAQ = DELTAQ;
    handles.datasets = datasets;
    handles.colors_force = colors_force;
    handles.colors_torque = colors_torque;
    handles.Max_force_mag = Max_force_mag;
    handles.Max_torque_mag = Max_torque_mag;
    handles.num_frames = num_frames;
    handles.current_frame = 1;
    handles.is_playing = false;
    handles.is_recording = false;
    handles.timer = [];

    % Store handles in figure
    setappdata(fig, 'handles', handles);

    %% === 7. Initial Plot ===
    updatePlot();

    %% === 8. Callback Functions ===
    function updatePlot()
        % Get current frame
        current_frame = round(get(handles.slider, 'Value'));
        handles.current_frame = current_frame;

        % Clear previous plots
        delete(handles.plot_handles.shaft);
        delete(handles.plot_handles.face_normal);
        delete(handles.plot_handles.left_forearm);
        delete(handles.plot_handles.left_upperarm);
        delete(handles.plot_handles.left_shoulderneck);
        delete(handles.plot_handles.right_forearm);
        delete(handles.plot_handles.right_upperarm);
        delete(handles.plot_handles.right_shoulderneck);
        delete(handles.plot_handles.forces);
        delete(handles.plot_handles.torques);

        % Reset arrays
        handles.plot_handles.shaft = [];
        handles.plot_handles.face_normal = [];
        handles.plot_handles.left_forearm = [];
        handles.plot_handles.left_upperarm = [];
        handles.plot_handles.left_shoulderneck = [];
        handles.plot_handles.right_forearm = [];
        handles.plot_handles.right_upperarm = [];
        handles.plot_handles.right_shoulderneck = [];
        handles.plot_handles.forces = [];
        handles.plot_handles.torques = [];

        % Plot segments based on checkboxes
        if get(handles.checkbox_list(8), 'Value') % Shaft
            plotShaft(current_frame);
        end

        if get(handles.checkbox_list(9), 'Value') % Face Normal
            plotFaceNormal(current_frame);
        end

        % Plot body segments
        if get(handles.checkbox_list(10), 'Value') % Left Forearm
            plotLeftForearm(current_frame);
        end

        if get(handles.checkbox_list(11), 'Value') % Left Upper Arm
            plotLeftUpperArm(current_frame);
        end

        if get(handles.checkbox_list(12), 'Value') % Left Shoulder-Neck
            plotLeftShoulderNeck(current_frame);
        end

        if get(handles.checkbox_list(13), 'Value') % Right Forearm
            plotRightForearm(current_frame);
        end

        if get(handles.checkbox_list(14), 'Value') % Right Upper Arm
            plotRightUpperArm(current_frame);
        end

        if get(handles.checkbox_list(15), 'Value') % Right Shoulder-Neck
            plotRightShoulderNeck(current_frame);
        end

        % Plot forces and torques
        plotForcesAndTorques(current_frame);

        % Update title with frame info
        title(handles.ax, sprintf('Golf Swing - Frame %d/%d (%.3f s)', ...
            current_frame, num_frames, BASEQ.Time(current_frame)));

        drawnow;
    end

    function plotShaft(frame)
        % Plot club shaft
        x = [BASEQ.MPx(frame), BASEQ.CHx(frame)];
        y = [BASEQ.MPy(frame), BASEQ.CHy(frame)];
        z = [BASEQ.MPz(frame), BASEQ.CHz(frame)];

        handles.plot_handles.shaft = plot3(handles.ax, x, y, z, 'k-', 'LineWidth', 3);
    end

    function plotFaceNormal(frame)
        % Plot club face normal (simplified)
        % This would need to be implemented based on actual club face data
    end

    function plotLeftForearm(frame)
        % Plot left forearm
        x = [BASEQ.LWx(frame), BASEQ.LEx(frame)];
        y = [BASEQ.LWy(frame), BASEQ.LEy(frame)];
        z = [BASEQ.LWz(frame), BASEQ.LEz(frame)];

        handles.plot_handles.left_forearm = plot3(handles.ax, x, y, z, 'Color', skin_color, 'LineWidth', forearm_diameter*1000);
    end

    function plotLeftUpperArm(frame)
        % Plot left upper arm
        x = [BASEQ.LEx(frame), BASEQ.LSx(frame)];
        y = [BASEQ.LEy(frame), BASEQ.LSy(frame)];
        z = [BASEQ.LEz(frame), BASEQ.LSz(frame)];

        handles.plot_handles.left_upperarm = plot3(handles.ax, x, y, z, 'Color', skin_color, 'LineWidth', upperarm_diameter*1000);
    end

    function plotLeftShoulderNeck(frame)
        % Plot left shoulder to neck
        x = [BASEQ.LSx(frame), BASEQ.HUBx(frame)];
        y = [BASEQ.LSy(frame), BASEQ.HUBy(frame)];
        z = [BASEQ.LSz(frame), BASEQ.HUBz(frame)];

        handles.plot_handles.left_shoulderneck = plot3(handles.ax, x, y, z, 'Color', shirt_color, 'LineWidth', shoulderneck_diameter*1000);
    end

    function plotRightForearm(frame)
        % Plot right forearm
        x = [BASEQ.RWx(frame), BASEQ.REx(frame)];
        y = [BASEQ.RWy(frame), BASEQ.REy(frame)];
        z = [BASEQ.RWz(frame), BASEQ.REz(frame)];

        handles.plot_handles.right_forearm = plot3(handles.ax, x, y, z, 'Color', skin_color, 'LineWidth', forearm_diameter*1000);
    end

    function plotRightUpperArm(frame)
        % Plot right upper arm
        x = [BASEQ.REx(frame), BASEQ.RSx(frame)];
        y = [BASEQ.REy(frame), BASEQ.RSy(frame)];
        z = [BASEQ.REz(frame), BASEQ.RSz(frame)];

        handles.plot_handles.right_upperarm = plot3(handles.ax, x, y, z, 'Color', skin_color, 'LineWidth', upperarm_diameter*1000);
    end

    function plotRightShoulderNeck(frame)
        % Plot right shoulder to neck
        x = [BASEQ.RSx(frame), BASEQ.HUBx(frame)];
        y = [BASEQ.RSy(frame), BASEQ.HUBy(frame)];
        z = [BASEQ.RSz(frame), BASEQ.HUBz(frame)];

        handles.plot_handles.right_shoulderneck = plot3(handles.ax, x, y, z, 'Color', shirt_color, 'LineWidth', shoulderneck_diameter*1000);
    end

    function plotForcesAndTorques(frame)
        % Plot forces and torques for all datasets
        scale_factor = get(handles.scale_slider, 'Value');

        for i = 1:size(datasets, 1)
            dataset_name = datasets{i, 1};
            dataset = datasets{i, 2};

            % Plot forces
            if get(handles.checkbox_list(i), 'Value') && isfield(dataset, 'TotalHandForceGlobal')
                force = dataset.TotalHandForceGlobal(frame, :);
                force_mag = norm(force);
                if force_mag > 0
                    force_scaled = force * scale_factor / Max_force_mag;
                    quiver3(handles.ax, BASEQ.MPx(frame), BASEQ.MPy(frame), BASEQ.MPz(frame), ...
                           force_scaled(1), force_scaled(2), force_scaled(3), ...
                           'Color', colors_force{i}, 'LineWidth', 2, 'MaxHeadSize', 0.5);
                end
            end

            % Plot torques
            if get(handles.checkbox_list(i+3), 'Value') && isfield(dataset, 'EquivalentMidpointCoupleGlobal')
                torque = dataset.EquivalentMidpointCoupleGlobal(frame, :);
                torque_mag = norm(torque);
                if torque_mag > 0
                    torque_scaled = torque * scale_factor / Max_torque_mag;
                    quiver3(handles.ax, BASEQ.MPx(frame), BASEQ.MPy(frame), BASEQ.MPz(frame), ...
                           torque_scaled(1), torque_scaled(2), torque_scaled(3), ...
                           'Color', colors_torque{i}, 'LineWidth', 2, 'MaxHeadSize', 0.5, 'LineStyle', '--');
                end
            end
        end
    end

    function togglePlayPause(src, ~)
        if get(src, 'Value')
            % Start playing
            set(src, 'String', 'Pause', 'BackgroundColor', [0.8 0.4 0.4]);
            handles.is_playing = true;
            startPlayback();
        else
            % Stop playing
            set(src, 'String', 'Play', 'BackgroundColor', [0.4 0.8 0.4]);
            handles.is_playing = false;
            stopPlayback();
        end
    end

    function startPlayback()
        if ~isempty(handles.timer)
            stop(handles.timer);
            delete(handles.timer);
        end

        speed = get(handles.speed_slider, 'Value');
        interval = 0.1 / speed; % 10 FPS base

        handles.timer = timer('ExecutionMode', 'fixedRate', ...
                            'Period', interval, ...
                            'TimerFcn', @playbackTimer);
        start(handles.timer);
    end

    function stopPlayback()
        if ~isempty(handles.timer)
            stop(handles.timer);
            delete(handles.timer);
            handles.timer = [];
        end
    end

    function playbackTimer(~, ~)
        if handles.is_playing
            current_frame = handles.current_frame + 1;
            if current_frame > num_frames
                current_frame = 1; % Loop back to start
            end

            set(handles.slider, 'Value', current_frame);
            updatePlot();
        end
    end

    function updateZoom(zoom_factor)
        % Update zoom level
        zoom_factor = 3 - zoom_factor; % Invert the factor

        margin = 0.3;
        xlim(handles.ax, [min(all_x) - margin, max(all_x) + margin] * zoom_factor);
        ylim(handles.ax, [min(all_y) - margin, max(all_y) + margin] * zoom_factor);
        zlim(handles.ax, [min(all_z) - margin, max(all_z) + margin] * zoom_factor);
    end

    function toggleRecord(src, ~)
        if get(src, 'Value')
            % Start recording
            set(src, 'String', 'Recording...', 'BackgroundColor', record_active_color);
            handles.is_recording = true;
            fprintf('🎬 Recording started...\n');
        else
            % Stop recording
            set(src, 'String', 'Record', 'BackgroundColor', record_idle_color);
            handles.is_recording = false;
            fprintf('⏹️ Recording stopped\n');
        end
    end

    % Clean up function
    set(fig, 'CloseRequestFcn', @closeFigure);

    function closeFigure(src, ~)
        stopPlayback();
        delete(src);
    end

end
