function SkeletonPlotter(BASEQ, ZTCFQ, DELTAQ)
% === SkeletonPlotter - FINAL BUNDLED VERSION ===
% Golf Swing Visualizer with Playback, Zoom, Recording, Multi-Dataset Forces

% --- Save initial workspace variables ---
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
fig = figure('Name', 'Golf Swing Plotter', ...
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

% --- Single Play/Pause Button inside panel (Dark Green) ---
handles.play_pause_button = uicontrol('Parent', handles.panel_sliders, 'Style', 'togglebutton', ...
    'String', 'Play', ...
    'Units', 'normalized', 'Position', [0.25 0.1 0.5 0.2], ...
    'FontSize', 10, ...
    'BackgroundColor', [0.4 0.8 0.4], ... % <-- Darker green
    'Callback', @togglePlayPause);

% --- 4.3 Zoom Panel (Top Left) ---
handles.panel_zoom = uipanel('Parent', fig, ...
    'Title', 'Zoom', ...
    'FontSize', font_size, ...
    'BackgroundColor', panel_background, ...
    'Units', 'normalized', 'Position', [0.01 0.93 0.22 0.05]);

% Zoom Slider (Controls camera zoom for the entire view)
handles.zoom_slider = uicontrol('Parent', handles.panel_zoom, 'Style', 'slider', ...
    'Min', 0.5, 'Max', 2.0, 'Value', 1, ...
    'Units', 'normalized', 'Position', [0.05 0.2 0.9 0.6], ...
    'BackgroundColor', [1 1 1], 'FontSize', 9, ...
    'Callback', @(src,~) updateZoom(src.Value));

% Zoom Update Function (inverted zoom direction)
function updateZoom(zoom_factor)
    % Invert the zoom direction: right should zoom in, left should zoom out
    zoom_factor = 3 - zoom_factor; % Invert the factor (to make right zoom in)

    % Calculate new axis limits based on zoom factor
    margin = 0.3;  % Keep the margin the same
    all_x = [BASEQ.Buttx; BASEQ.CHx; BASEQ.MPx; BASEQ.LWx; BASEQ.LEx; BASEQ.LSx; BASEQ.RWx; BASEQ.REx; BASEQ.RSx; BASEQ.HUBx];
    all_y = [BASEQ.Butty; BASEQ.CHy; BASEQ.MPy; BASEQ.LWy; BASEQ.LEy; BASEQ.LSy; BASEQ.RWy; BASEQ.REy; BASEQ.RSy; BASEQ.HUBy];
    all_z = [BASEQ.Buttz; BASEQ.CHz; BASEQ.MPz; BASEQ.LWz; BASEQ.LEz; BASEQ.LSz; BASEQ.RWz; BASEQ.REz; BASEQ.RSz; BASEQ.HUBz];

    % Apply zoom scaling to axis limits
    xlim(handles.ax, [min(all_x) - margin, max(all_x) + margin] * zoom_factor);
    ylim(handles.ax, [min(all_y) - margin, max(all_y) + margin] * zoom_factor);
    zlim(handles.ax, [min(all_z) - margin, max(all_z) + margin] * zoom_factor);
end

% --- 4.4 Record Button (Bottom Right, Orange) ---
handles.record_button = uicontrol('Parent', fig, 'Style', 'togglebutton', ...
    'String', 'Record', ...
    'Units', 'normalized', 'Position', [0.86 0.01 0.12 0.04], ...
    'BackgroundColor', [1.0 0.6 0.0], ... % <-- New orange color!
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
        'ForegroundColor', legend_entries{k,2}, ...
        'BackgroundColor', [1 1 1], ...
        'Units', 'normalized', ...
        'Position', [0.05 1-k*0.15 0.9 0.1], ...
        'FontSize', 9, ...
        'HorizontalAlignment', 'left');
end

% Renamed function to avoid conflicts
function toggleLegendVisibility(~, ~)
    if strcmp(handles.panel_legend.Visible, 'on')
        handles.panel_legend.Visible = 'off';
    else
        handles.panel_legend.Visible = 'on';
    end
end

% --- 4.6 Camera View Buttons (Shifted Up and Closer Together) ---
button_width = 0.12;
button_height = 0.05;
button_x_position = 0.86; % Near the right edge
button_spacing = 0.06; % Spacing between buttons

% Starting position for the camera buttons and the Show/Hide Legend button
initial_y_position = 0.92; % Shift the buttons up slightly

% Face-On View Button
uicontrol('Style', 'pushbutton', 'String', 'Face-On', ...
          'Units', 'normalized', 'Position', [button_x_position, initial_y_position, button_width, button_height], ...
          'FontSize', 10, 'Callback', @(~,~) setView('faceon'), ...
          'BackgroundColor', [0.8 0.8 0.8]);

% Down-the-Line View Button
uicontrol('Style', 'pushbutton', 'String', 'Down-the-Line', ...
          'Units', 'normalized', 'Position', [button_x_position, initial_y_position - button_spacing, button_width, button_height], ...
          'FontSize', 10, 'Callback', @(~,~) setView('downline'), ...
          'BackgroundColor', [0.8 0.8 0.8]);

% Top-Down View Button
uicontrol('Style', 'pushbutton', 'String', 'Top-Down', ...
          'Units', 'normalized', 'Position', [button_x_position, initial_y_position - 2 * button_spacing, button_width, button_height], ...
          'FontSize', 10, 'Callback', @(~,~) setView('topdown'), ...
          'BackgroundColor', [0.8 0.8 0.8]);

% Isometric View Button
uicontrol('Style', 'pushbutton', 'String', 'Isometric', ...
          'Units', 'normalized', 'Position', [button_x_position, initial_y_position - 3 * button_spacing, button_width, button_height], ...
          'FontSize', 10, 'Callback', @(~,~) setView('iso'), ...
          'BackgroundColor', [0.8 0.8 0.8]);

% Show/Hide Legend Button (new size and position)
uicontrol('Style', 'togglebutton', ...
    'String', 'Show/Hide Legend', ...
    'Units', 'normalized', ...
    'Position', [button_x_position, initial_y_position - 4 * button_spacing, button_width, button_height], ...  % Positioned below other buttons
    'BackgroundColor', [0.8 0.8 0.8], ...
    'FontSize', 10, ...
    'Callback', @toggleLegendVisibility);

%% === 5. Create Plot Handles (Initialize Empty Graphics) ===

% --- Shaft and Clubhead ---
handles.shaft_cylinder = surf(handles.ax, nan(2), nan(2), nan(2), ...
    'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'none');

handles.clubhead_half = surf(handles.ax, nan(2), nan(2), nan(2), ...
    'FaceColor', 'b', 'EdgeColor', 'none');

% --- Force and Torque Vectors ---
for k = 1:3
    handles.force_quivers{k} = quiver3(handles.ax, 0, 0, 0, 0, 0, 0, ...
        'Color', colors_force{k}, 'LineWidth', 2, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
    handles.torque_quivers{k} = quiver3(handles.ax, 0, 0, 0, 0, 0, 0, ...
        'Color', colors_torque{k}, 'LineWidth', 2, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
end

% --- Face Normal Vector ---
handles.face_normal_quiver = quiver3(handles.ax, 0, 0, 0, 0, 0, 0, ...
    'Color', [0 1 0], 'LineWidth', 2, 'MaxHeadSize', 0.5, 'AutoScale', 'off');

% --- Body Segments (Cylinders) ---
handles.left_forearm = surf(handles.ax, nan(2), nan(2), nan(2), ...
    'FaceColor', skin_color, 'EdgeColor', 'none');
handles.left_upperarm = surf(handles.ax, nan(2), nan(2), nan(2), ...
    'FaceColor', shirt_color, 'EdgeColor', 'none');
handles.left_shoulder_neck = surf(handles.ax, nan(2), nan(2), nan(2), ...
    'FaceColor', shirt_color, 'EdgeColor', 'none');

handles.right_forearm = surf(handles.ax, nan(2), nan(2), nan(2), ...
    'FaceColor', skin_color, 'EdgeColor', 'none');
handles.right_upperarm = surf(handles.ax, nan(2), nan(2), nan(2), ...
    'FaceColor', shirt_color, 'EdgeColor', 'none');
handles.right_shoulder_neck = surf(handles.ax, nan(2), nan(2), nan(2), ...
    'FaceColor', shirt_color, 'EdgeColor', 'none');

% --- Body Joints (Spheres) ---
handles.left_forearm_sphere = surf(handles.ax, nan(2), nan(2), nan(2), ...
    'FaceColor', skin_color, 'EdgeColor', 'none');
handles.left_upperarm_sphere = surf(handles.ax, nan(2), nan(2), nan(2), ...
    'FaceColor', shirt_color, 'EdgeColor', 'none');
handles.left_shoulder_neck_sphere = surf(handles.ax, nan(2), nan(2), nan(2), ...
    'FaceColor', shirt_color, 'EdgeColor', 'none');

handles.right_forearm_sphere = surf(handles.ax, nan(2), nan(2), nan(2), ...
    'FaceColor', skin_color, 'EdgeColor', 'none');
handles.right_upperarm_sphere = surf(handles.ax, nan(2), nan(2), nan(2), ...
    'FaceColor', shirt_color, 'EdgeColor', 'none');
handles.right_shoulder_neck_sphere = surf(handles.ax, nan(2), nan(2), nan(2), ...
    'FaceColor', shirt_color, 'EdgeColor', 'none');

% --- Recording and Playback Management ---
handles.recording = false;
handles.videoObj = [];
handles.playing = false;

% --- Initial Frame Update ---
updatePlot();
%% === 6. Nested Functions ===

function updatePlot(~, ~)
    i = round(get(handles.slider, 'Value'));

    butt = [BASEQ.Buttx(i), BASEQ.Butty(i), BASEQ.Buttz(i)];
    clubhead = [BASEQ.CHx(i), BASEQ.CHy(i), BASEQ.CHz(i)];
    shaft_vec = clubhead - butt;
    shaft_length = norm(shaft_vec);

    scale = get(handles.scale_slider, 'Value');

    % Shaft
    update_cylinder(handles.shaft_cylinder, butt, clubhead, shaft_diameter, handles.checkbox_list(7));

    % Clubhead Hemisphere
    [theta, phi] = meshgrid(linspace(0,2*pi,30), linspace(0,pi/2,15));
    sphere_x = cos(theta).*sin(phi);
    sphere_y = sin(theta).*sin(phi);
    sphere_z = cos(phi);
    shaft_dir = shaft_vec / norm(shaft_vec);
    y_axis = cross([0 0 1], shaft_dir);
    if norm(y_axis) < 1e-6, y_axis = [1 0 0]; else, y_axis = y_axis / norm(y_axis); end
    x_axis = shaft_dir;
    z_axis = cross(x_axis, y_axis);
    R = [x_axis(:), y_axis(:), z_axis(:)];
    pts = [sphere_x(:)'; sphere_y(:)'; sphere_z(:)'];
    rotated = R * pts;
    nx = reshape(rotated(1,:), size(sphere_x));
    ny = reshape(rotated(2,:), size(sphere_y));
    nz = reshape(rotated(3,:), size(sphere_z));
    set(handles.clubhead_half, 'XData', clubhead(1)+(clubhead_diameter/2)*nx, ...
                                'YData', clubhead(2)+(clubhead_diameter/2)*ny, ...
                                'ZData', clubhead(3)+(clubhead_diameter/2)*nz);

    % Face Normal
    face_normal = cross(shaft_vec, [0 0 1]);
    if norm(face_normal) < 1e-6, face_normal = [1 0 0]; else, face_normal = face_normal / norm(face_normal); end
    if get(handles.checkbox_list(8), 'Value')
        set(handles.face_normal_quiver, 'Visible', 'on', ...
            'XData', clubhead(1), 'YData', clubhead(2), 'ZData', clubhead(3), ...
            'UData', face_normal(1)*shaft_length, ...
            'VData', face_normal(2)*shaft_length, ...
            'WData', face_normal(3)*shaft_length);
    else
        set(handles.face_normal_quiver, 'Visible', 'off');
    end

    % Forces and Torques
    for k = 1:3
        data = datasets{k,2};
        force_vec = data.TotalHandForceGlobal(i,:);
        torque_vec = data.EquivalentMidpointCoupleGlobal(i,:);
        mp = [BASEQ.MPx(i), BASEQ.MPy(i), BASEQ.MPz(i)];

        if get(handles.checkbox_list(k), 'Value')
            set(handles.force_quivers{k}, 'Visible', 'on', ...
                'XData', mp(1), 'YData', mp(2), 'ZData', mp(3), ...
                'UData', force_vec(1)*shaft_length/Max_force_mag*scale, ...
                'VData', force_vec(2)*shaft_length/Max_force_mag*scale, ...
                'WData', force_vec(3)*shaft_length/Max_force_mag*scale);
        else
            set(handles.force_quivers{k}, 'Visible', 'off');
        end

        if get(handles.checkbox_list(k+3), 'Value')
            set(handles.torque_quivers{k}, 'Visible', 'on', ...
                'XData', mp(1), 'YData', mp(2), 'ZData', mp(3), ...
                'UData', torque_vec(1)*shaft_length/Max_torque_mag*scale, ...
                'VData', torque_vec(2)*shaft_length/Max_torque_mag*scale, ...
                'WData', torque_vec(3)*shaft_length/Max_torque_mag*scale);
        else
            set(handles.torque_quivers{k}, 'Visible', 'off');
        end
    end

    % Body Segments
    lwrist = [BASEQ.LWx(i), BASEQ.LWy(i), BASEQ.LWz(i)];
    lelbow = [BASEQ.LEx(i), BASEQ.LEy(i), BASEQ.LEz(i)];
    lshoulder = [BASEQ.LSx(i), BASEQ.LSy(i), BASEQ.LSz(i)];
    hub = [BASEQ.HUBx(i), BASEQ.HUBy(i), BASEQ.HUBz(i)];
    rwrist = [BASEQ.RWx(i), BASEQ.RWy(i), BASEQ.RWz(i)];
    relbow = [BASEQ.REx(i), BASEQ.REy(i), BASEQ.REz(i)];
    rshoulder = [BASEQ.RSx(i), BASEQ.RSy(i), BASEQ.RSz(i)];

    update_cylinder(handles.left_forearm, lwrist, lelbow, forearm_diameter, handles.checkbox_list(9));
    update_cylinder(handles.left_upperarm, lelbow, lshoulder, upperarm_diameter, handles.checkbox_list(10));
    update_cylinder(handles.left_shoulder_neck, lshoulder, hub, shoulderneck_diameter, handles.checkbox_list(11));
    update_cylinder(handles.right_forearm, rwrist, relbow, forearm_diameter, handles.checkbox_list(12));
    update_cylinder(handles.right_upperarm, relbow, rshoulder, upperarm_diameter, handles.checkbox_list(13));
    update_cylinder(handles.right_shoulder_neck, rshoulder, hub, shoulderneck_diameter, handles.checkbox_list(14));

    update_sphere(handles.left_forearm_sphere, lwrist, forearm_diameter, handles.checkbox_list(9));
    update_sphere(handles.left_upperarm_sphere, lelbow, upperarm_diameter, handles.checkbox_list(10));
    update_sphere(handles.left_shoulder_neck_sphere, lshoulder, shoulderneck_diameter, handles.checkbox_list(11));
    update_sphere(handles.right_forearm_sphere, rwrist, forearm_diameter, handles.checkbox_list(12));
    update_sphere(handles.right_upperarm_sphere, relbow, upperarm_diameter, handles.checkbox_list(13));
    update_sphere(handles.right_shoulder_neck_sphere, rshoulder, shoulderneck_diameter, handles.checkbox_list(14));

    % If Recording
    if handles.recording
        frame = getframe(fig);
        writeVideo(handles.videoObj, frame);
    end
end

function togglePlayPause(~, ~)
    if get(handles.play_pause_button, 'Value') == 1
        set(handles.play_pause_button, 'String', 'Pause');
        handles.playing = true;
        while handles.playing && ishandle(handles.slider)
            i = round(get(handles.slider, 'Value'));
            speed = get(handles.speed_slider, 'Value');
            if i < num_frames
                set(handles.slider, 'Value', i+1);
            else
                set(handles.slider, 'Value', 1);
            end
            updatePlot();
            pause(0.03 / speed);
            if get(handles.play_pause_button, 'Value') == 0
                break;
            end
        end
        set(handles.play_pause_button, 'String', 'Play');
    else
        handles.playing = false;
        set(handles.play_pause_button, 'String', 'Play');
    end
end

function toggleRecord(~, ~)
    if get(handles.record_button, 'Value')
        [file, path] = uiputfile('*.mp4', 'Save Swing Recording As...');
        if isequal(file,0)
            set(handles.record_button, 'Value', 0);
            return;
        end
        filename = fullfile(path, file);
        handles.videoObj = VideoWriter(filename, 'MPEG-4');
        handles.videoObj.FrameRate = 30;
        open(handles.videoObj);
        handles.recording = true;
        set(handles.record_button, 'String', 'Stop Recording', 'BackgroundColor', record_active_color);
    else
        handles.recording = false;
        if ~isempty(handles.videoObj)
            close(handles.videoObj);
            handles.videoObj = [];
        end
        set(handles.record_button, 'String', 'Record', 'BackgroundColor', record_idle_color);
    end
end

function toggleLegend(~, ~)
    if strcmp(handles.panel_legend.Visible, 'on')
        handles.panel_legend.Visible = 'off';
    else
        handles.panel_legend.Visible = 'on';
    end
end

function update_cylinder(hsurf, pt1, pt2, diameter, checkbox)
    [cyl_x, cyl_y, cyl_z] = cylinder(diameter/2, 20);
    cyl_pts = [cyl_x(:)'; cyl_y(:)'; cyl_z(:)'];
    vec = pt2 - pt1;
    height = norm(vec);
    if height < 1e-6
        R = eye(3);
    else
        z_axis = [0 0 1];
        dir = vec / height;
        v_cross = cross(z_axis, dir);
        c = dot(z_axis, dir);
        v_skew = [0 -v_cross(3) v_cross(2); v_cross(3) 0 -v_cross(1); -v_cross(2) v_cross(1) 0];
        R = eye(3) + v_skew + v_skew^2 * (1/(1+c));
    end
    cyl_pts(3,:) = cyl_pts(3,:) * height;
    cyl_pts_rot = R * cyl_pts;
    Xc = reshape(cyl_pts_rot(1,:) + pt1(1), size(cyl_x));
    Yc = reshape(cyl_pts_rot(2,:) + pt1(2), size(cyl_y));
    Zc = reshape(cyl_pts_rot(3,:) + pt1(3), size(cyl_z));
    if get(checkbox, 'Value')
        set(hsurf, 'Visible', 'on', 'XData', Xc, 'YData', Yc, 'ZData', Zc);
    else
        set(hsurf, 'Visible', 'off');
    end
end

function update_sphere(hsurf, center, diameter, checkbox)
    [sx, sy, sz] = sphere(20);
    sx = sx * (diameter/2);
    sy = sy * (diameter/2);
    sz = sz * (diameter/2);
    if get(checkbox, 'Value')
        set(hsurf, 'Visible', 'on', ...
            'XData', center(1) + sx, ...
            'YData', center(2) + sy, ...
            'ZData', center(3) + sz);
    else
        set(hsurf, 'Visible', 'off');
    end
end

function setView(viewtype)
    switch viewtype
        case 'faceon'
            view(handles.ax, [0 0]); % Side view along +X
            camup(handles.ax, [0 0 1]);
        case 'downline'
            view(handles.ax, [270 0]); % Behind golfer toward target
            camup(handles.ax, [0 0 1]);
        case 'topdown'
            view(handles.ax, [0 90]); % Top view
            camup(handles.ax, [1 0 0]);
        case 'iso'
            view(handles.ax, [-45 30]); % 3D angled view from left rear
            camup(handles.ax, [0 0 1]);
    end
end

end
