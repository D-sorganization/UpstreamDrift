function handles = EmbeddedSkeletonPlotter(parent_container, BASEQ, ZTCFQ, DELTAQ)
% EMBEDDEDSKELETONPLOTTER - Skeleton plotter embedded in a parent container
%
% This is a modified version of SkeletonPlotter that renders inside a
% parent container (like a tab) instead of creating its own figure.
%
% Usage:
%   handles = EmbeddedSkeletonPlotter(parent_panel, BASEQ, ZTCFQ, DELTAQ)
%
% Inputs:
%   parent_container - Handle to parent uipanel or figure
%   BASEQ, ZTCFQ, DELTAQ - Data tables
%
% Returns:
%   handles - Structure with all UI and plot handles

%% Settings and Constants
shirt_color = [0.6, 0.8, 1];
skin_color = [1, 0.8, 0.6];
axes_background_color = [1, 1, 0.8];
panel_background = [0.8, 1, 0.8];
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

% Store datasets
datasets_struct = struct('BASEQ', BASEQ, 'ZTCFQ', ZTCFQ, 'DELTAQ', DELTAQ);

% Frames
num_frames = length(BASEQ.Buttx);

% Scaling
Max_force_mag = max(vecnorm(BASEQ.TotalHandForceGlobal,2,2));
Max_torque_mag = max(vecnorm(BASEQ.EquivalentMidpointCoupleGlobal,2,2));

%% Create Layout within Parent Container

% Clear parent
delete(allchild(parent_container));

% Main axes (right side)
handles.ax = axes('Parent', parent_container, ...
    'Color', axes_background_color, ...
    'Units', 'normalized', ...
    'Position', [0.25 0.05 0.7 0.9]);
axis(handles.ax, 'equal');
grid(handles.ax, 'on');
xlabel(handles.ax, 'X (m)');
ylabel(handles.ax, 'Y (m)');
zlabel(handles.ax, 'Z (m)');
hold(handles.ax, 'on');
view(handles.ax, 3);

% Set plot limits
all_x = [BASEQ.Buttx; BASEQ.CHx; BASEQ.MPx; BASEQ.LWx; BASEQ.LEx; BASEQ.LSx; BASEQ.RWx; BASEQ.REx; BASEQ.RSx; BASEQ.HUBx];
all_y = [BASEQ.Butty; BASEQ.CHy; BASEQ.MPy; BASEQ.LWy; BASEQ.LEy; BASEQ.LSy; BASEQ.RWy; BASEQ.REy; BASEQ.RSy; BASEQ.HUBy];
all_z = [BASEQ.Buttz; BASEQ.CHz; BASEQ.MPz; BASEQ.LWz; BASEQ.LEz; BASEQ.LSz; BASEQ.RWz; BASEQ.REz; BASEQ.RSz; BASEQ.HUBz];
margin = 0.3;
xlim(handles.ax, [min(all_x)-margin, max(all_x)+margin]);
ylim(handles.ax, [min(all_y)-margin, max(all_y)+margin]);
zlim(handles.ax, [min(all_z)-margin, max(all_z)+margin]);
axis manual;

camlight(handles.ax, 'headlight');
lighting(handles.ax, 'gouraud');
material(handles.ax, 'dull');
shading(handles.ax, 'interp');

%% Create Control Panels (Left Side)

% Dataset selection panel
handles.panel_dataset = uipanel('Parent', parent_container, ...
    'Title', 'Dataset', ...
    'FontSize', font_size, ...
    'BackgroundColor', panel_background, ...
    'Units', 'normalized', ...
    'Position', [0.01 0.88 0.22 0.10]);

handles.dataset_dropdown = uicontrol('Parent', handles.panel_dataset, ...
    'Style', 'popupmenu', ...
    'String', {'BASEQ', 'ZTCFQ', 'DELTAQ'}, ...
    'Value', 1, ...
    'Units', 'normalized', ...
    'Position', [0.05 0.2 0.9 0.6], ...
    'BackgroundColor', [1 1 1], ...
    'FontSize', 9, ...
    'Callback', @onDatasetChanged);

% Playback controls panel
handles.panel_playback = uipanel('Parent', parent_container, ...
    'Title', 'Playback', ...
    'FontSize', font_size, ...
    'BackgroundColor', panel_background, ...
    'Units', 'normalized', ...
    'Position', [0.01 0.70 0.22 0.16]);

% Play button
handles.play_button = uicontrol('Parent', handles.panel_playback, ...
    'Style', 'pushbutton', ...
    'String', 'Play', ...
    'FontSize', font_size, ...
    'Units', 'normalized', ...
    'Position', [0.05 0.60 0.42 0.30], ...
    'Callback', @onPlayPause);

% Stop button
handles.stop_button = uicontrol('Parent', handles.panel_playback, ...
    'Style', 'pushbutton', ...
    'String', 'Stop', ...
    'FontSize', font_size, ...
    'Units', 'normalized', ...
    'Position', [0.53 0.60 0.42 0.30], ...
    'Callback', @onStop);

% Speed label and slider
uicontrol('Parent', handles.panel_playback, ...
    'Style', 'text', ...
    'String', 'Speed:', ...
    'FontSize', font_size-1, ...
    'Units', 'normalized', ...
    'Position', [0.05 0.25 0.35 0.25], ...
    'BackgroundColor', panel_background, ...
    'HorizontalAlignment', 'left');

handles.speed_slider = uicontrol('Parent', handles.panel_playback, ...
    'Style', 'slider', ...
    'Min', 0.1, 'Max', 2, 'Value', 1, ...
    'Units', 'normalized', ...
    'Position', [0.05 0.05 0.9 0.20]);

% Frame slider
handles.slider = uicontrol('Parent', parent_container, ...
    'Style', 'slider', ...
    'Min', 1, 'Max', num_frames, ...
    'Value', 1, ...
    'SliderStep', [1/(num_frames-1), 10/(num_frames-1)], ...
    'Units', 'normalized', ...
    'Position', [0.25 0.01 0.7 0.03], ...
    'Callback', @onSliderChange);

% Frame label
handles.frame_label = uicontrol('Parent', parent_container, ...
    'Style', 'text', ...
    'String', sprintf('Frame: 1 / %d', num_frames), ...
    'FontSize', font_size, ...
    'Units', 'normalized', ...
    'Position', [0.01 0.01 0.22 0.03], ...
    'BackgroundColor', panel_background, ...
    'HorizontalAlignment', 'center');

% Display options panel
handles.panel_display = uipanel('Parent', parent_container, ...
    'Title', 'Display Options', ...
    'FontSize', font_size, ...
    'BackgroundColor', panel_background, ...
    'Units', 'normalized', ...
    'Position', [0.01 0.40 0.22 0.28]);

% Checkboxes for display options
y_pos = 0.75;
dy = 0.15;

handles.check_forces = uicontrol('Parent', handles.panel_display, ...
    'Style', 'checkbox', ...
    'String', 'Show Forces', ...
    'Value', 1, ...
    'Units', 'normalized', ...
    'Position', [0.05 y_pos 0.9 dy], ...
    'BackgroundColor', panel_background, ...
    'Callback', @updatePlot);
y_pos = y_pos - dy;

handles.check_torques = uicontrol('Parent', handles.panel_display, ...
    'Style', 'checkbox', ...
    'String', 'Show Torques', ...
    'Value', 1, ...
    'Units', 'normalized', ...
    'Position', [0.05 y_pos 0.9 dy], ...
    'BackgroundColor', panel_background, ...
    'Callback', @updatePlot);
y_pos = y_pos - dy;

handles.check_trail = uicontrol('Parent', handles.panel_display, ...
    'Style', 'checkbox', ...
    'String', 'Show Trail', ...
    'Value', 0, ...
    'Units', 'normalized', ...
    'Position', [0.05 y_pos 0.9 dy], ...
    'BackgroundColor', panel_background, ...
    'Callback', @updatePlot);
y_pos = y_pos - dy;

handles.check_club = uicontrol('Parent', handles.panel_display, ...
    'Style', 'checkbox', ...
    'String', 'Show Club', ...
    'Value', 1, ...
    'Units', 'normalized', ...
    'Position', [0.05 y_pos 0.9 dy], ...
    'BackgroundColor', panel_background, ...
    'Callback', @updatePlot);

%% Initialize State
handles.current_frame = 1;
handles.current_dataset = 1;
handles.current_data = BASEQ;
handles.playing = false;
handles.play_speed = 1;
handles.datasets = datasets;
handles.datasets_struct = datasets_struct;
handles.colors_force = colors_force;
handles.colors_torque = colors_torque;
handles.Max_force_mag = Max_force_mag;
handles.Max_torque_mag = Max_torque_mag;
handles.clubhead_diameter = clubhead_diameter;
handles.shaft_diameter = shaft_diameter;
handles.forearm_diameter = forearm_diameter;
handles.upperarm_diameter = upperarm_diameter;
handles.shoulderneck_diameter = shoulderneck_diameter;
handles.shirt_color = shirt_color;
handles.skin_color = skin_color;
handles.num_frames = num_frames;
handles.parent_container = parent_container;

% Initialize graphics objects
handles.skeleton_lines = [];
handles.force_arrows = [];
handles.torque_arrows = [];
handles.trail_line = [];
handles.club_graphics = [];

% Initial plot
updatePlot();

%% Callback Functions

    function onDatasetChanged(~, ~)
        dataset_idx = get(handles.dataset_dropdown, 'Value');
        handles.current_dataset = dataset_idx;
        handles.current_data = datasets{dataset_idx, 2};
        updatePlot();
    end

    function onPlayPause(~, ~)
        handles.playing = ~handles.playing;
        if handles.playing
            set(handles.play_button, 'String', 'Pause');
            playbackLoop();
        else
            set(handles.play_button, 'String', 'Play');
        end
    end

    function onStop(~, ~)
        handles.playing = false;
        set(handles.play_button, 'String', 'Play');
        handles.current_frame = 1;
        set(handles.slider, 'Value', 1);
        updatePlot();
    end

    function onSliderChange(~, ~)
        handles.current_frame = round(get(handles.slider, 'Value'));
        updatePlot();
    end

    function playbackLoop()
        while handles.playing && ishandle(parent_container)
            % Update frame
            handles.current_frame = handles.current_frame + 1;
            if handles.current_frame > handles.num_frames
                handles.current_frame = 1;
            end

            % Update slider and plot
            set(handles.slider, 'Value', handles.current_frame);
            updatePlot();

            % Control speed
            handles.play_speed = get(handles.speed_slider, 'Value');
            pause(0.03 / handles.play_speed);
        end
        set(handles.play_button, 'String', 'Play');
        handles.playing = false;
    end

    function updatePlot()
        % Clear previous graphics
        delete(handles.skeleton_lines);
        delete(handles.force_arrows);
        delete(handles.torque_arrows);
        delete(handles.trail_line);
        delete(handles.club_graphics);

        % Get current frame data
        frame = handles.current_frame;
        data = handles.current_data;

        % Extract positions
        positions = {
            [data.Buttx(frame), data.Butty(frame), data.Buttz(frame)];
            [data.LSx(frame), data.LSy(frame), data.LSz(frame)];
            [data.RSx(frame), data.RSy(frame), data.RSz(frame)];
            [data.LEx(frame), data.LEy(frame), data.LEz(frame)];
            [data.REx(frame), data.REy(frame), data.REz(frame)];
            [data.LWx(frame), data.LWy(frame), data.LWz(frame)];
            [data.RWx(frame), data.RWy(frame), data.RWz(frame)];
            [data.MPx(frame), data.MPy(frame), data.MPz(frame)];
            [data.HUBx(frame), data.HUBy(frame), data.HUBz(frame)];
            [data.CHx(frame), data.CHy(frame), data.CHz(frame)]
            };

        % Draw skeleton
        connections = [1 2; 1 3; 2 4; 3 5; 4 6; 5 7; 6 8; 7 8];
        handles.skeleton_lines = [];

        for i = 1:size(connections, 1)
            p1 = positions{connections(i,1)};
            p2 = positions{connections(i,2)};
            h = plot3(handles.ax, [p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], ...
                'b-', 'LineWidth', 3);
            handles.skeleton_lines = [handles.skeleton_lines; h];
        end

        % Draw club if enabled
        if get(handles.check_club, 'Value')
            mp = positions{8};
            hub = positions{9};
            ch = positions{10};

            h1 = plot3(handles.ax, [mp(1) hub(1)], [mp(2) hub(2)], [mp(3) hub(3)], ...
                'k-', 'LineWidth', 4);
            h2 = plot3(handles.ax, [hub(1) ch(1)], [hub(2) ch(2)], [hub(3) ch(3)], ...
                'Color', [0.5 0.3 0.1], 'LineWidth', 2);
            handles.club_graphics = [h1; h2];
        end

        % Update frame label
        set(handles.frame_label, 'String', sprintf('Frame: %d / %d', frame, handles.num_frames));
    end

end
