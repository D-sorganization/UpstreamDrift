% I want to start with this script as a starting point. (GolfClubInverseDynamicsGUI). Then I want to make it complete the tasks below:
%
% Analyze the excel file attached and use it as a standard input format for data analysis. There are 4 tabs of interest in the spreadsheet and we are looking to read the data from each of the tabs as it pertains to 4 swings. Two swings with golfer TW and two swings with golfer GW. One club is made with a wiffle ball and one swing is made with a real ball.
%
% Multiple filtering options will be evaluated (as shown in the starting file). The inverse dynamics should be computed using each filtering method and the force vector on the club should be computed with each. Additionally, the equivalent mid hands torque should be computed. This corresponds to the couple applied to the club calculated as if the total force was applied at the mid hands point on the club.
%
% The GUI calculation should also be able to have a variation in the midhands point to evaluate what would happen to the torque calculated when the evaluation point is moved one inch up or down the shaft. This can be done with a drop down menu and the default value for the dropdown menu should be zero adjustment (at the midhands point).
%
% The XYZ data should be used for both the center of the clubface and the mid hands for all four golfers. A preliminary script should put this information into a .csv file with a common format that will be useful for future use. Then the program will read this .csv with a loading function and compute the inverse dynamics on the data with each of the filtering methods.
%
% The output from the inverse dynamics should be similarly stored in csv files for each filtering method every time a golfer is analyzed. The data should be stored to the same folder that the script is in, in a folder labelled results that will be generated if it doesn't exist.
%
% The graphics should display a golf club moving through each frame of the motion. This can be represented by a cylinder running between the mid hands point and the center of the club face. Future graphical improvements would be beneficial.
%
%



function InvDynamicsMATLABGPT()
% Golf Swing Analysis GUI - Full Version
close all; clc;

% === FIGURE SETUP ===
fig = figure('Name', 'Golf Club Inverse Dynamics GUI', ...
             'NumberTitle', 'off', ...
             'Color', [0.9 1 0.9], ...
             'Position', [100, 100, 1400, 800]);

handles = struct();
handles.recording = false;

% === LOAD EXCEL FILE FROM SCRIPT DIRECTORY ===
scriptDir = fileparts(mfilename('fullpath'));
xlsFile = fullfile(scriptDir, 'Wiffle_ProV1_club_3D_data.xlsx');
if ~isfile(xlsFile)
    errordlg('Excel file not found in the script folder.');
    return;
end

[~, sheetNames] = xlsfinfo(xlsFile);
swingSheets = sheetNames(2:5); % Swings are on tabs 2-5
nSwings = numel(swingSheets);

% === PROMPT USER FOR SWING SELECTION ===
[selectedSwingIdx, ok] = listdlg('PromptString','Select a swing:', ...
                                 'SelectionMode','single', ...
                                 'ListString', swingSheets, ...
                                 'ListSize', [300 150]);
if ~ok
    disp('User cancelled swing selection.');
    return;
end

selectedSheet = swingSheets{selectedSwingIdx};

% === LOAD DATA FROM SELECTED SWING ===
raw = readmatrix(xlsFile, 'Sheet', selectedSheet);
keyFrames = raw(1:3, 2:end); % Assumes first 3 rows are event frames
data = raw(4:end, 2:end);    % Assumes data starts on row 4

% === INITIALIZE STATE ===
handles.swingData = data;
handles.frame = 1;
handles.mass = 0.2; % Approx. driver mass (kg)
handles.I = diag([0.0018, 0.0022, 0.0015]); % Driver inertia (kg*m^2)
handles.evalOffset = 0; % Midhands default
handles.time = linspace(0, 1, size(data,1));
handles.swingName = selectedSheet;
handles.keyFrames = keyFrames;
%% === 3D AXES SETUP ===
handles.ax = axes('Parent', fig, ...
    'Position', [0.3 0.05 0.65 0.9], ...
    'Color', [1 1 0.9]);
axis(handles.ax, 'equal'); grid(handles.ax, 'on'); view(handles.ax, 3);
xlabel(handles.ax, 'X (m)'); ylabel(handles.ax, 'Y (m)'); zlabel(handles.ax, 'Z (m)');
hold(handles.ax, 'on');

% Initial 3D axis limits (placeholder until data processed)
xlim(handles.ax, [-2 2]);
ylim(handles.ax, [-2 2]);
zlim(handles.ax, [-1 2]);

% Default lighting and material
camlight(handles.ax, 'headlight');
lighting(handles.ax, 'gouraud');
material(handles.ax, 'dull');

%% === CONTROL PANELS ===

% Playback Panel
handles.panel_playback = uipanel('Parent', fig, 'Title', 'Playback & View', ...
    'FontSize', 10, 'BackgroundColor', [0.8 1 0.8], ...
    'Units', 'normalized', 'Position', [0.01 0.05 0.25 0.35]);

% View Buttons
uicontrol('Parent', handles.panel_playback, 'Style', 'pushbutton', ...
    'String', 'Face-On', 'Units', 'normalized', 'Position', [0.05 0.85 0.4 0.08], ...
    'Callback', @(~,~) view(handles.ax, [-1 0 0]));
uicontrol('Parent', handles.panel_playback, 'Style', 'pushbutton', ...
    'String', 'Down-the-Line', 'Units', 'normalized', 'Position', [0.55 0.85 0.4 0.08], ...
    'Callback', @(~,~) view(handles.ax, [0 -1 0]));
uicontrol('Parent', handles.panel_playback, 'Style', 'pushbutton', ...
    'String', 'Top-Down', 'Units', 'normalized', 'Position', [0.05 0.75 0.4 0.08], ...
    'Callback', @(~,~) view(handles.ax, [0 0 1]));
uicontrol('Parent', handles.panel_playback, 'Style', 'pushbutton', ...
    'String', 'Isometric', 'Units', 'normalized', 'Position', [0.55 0.75 0.4 0.08], ...
    'Callback', @(~,~) view(handles.ax, [-45 30]));

% Playback Controls
handles.slider = uicontrol('Parent', handles.panel_playback, 'Style', 'slider', ...
    'Min', 1, 'Max', size(handles.swingData,1), ...
    'Value', 1, 'SliderStep', [1/(size(handles.swingData,1)-1) 0.1], ...
    'Units', 'normalized', 'Position', [0.05 0.3 0.9 0.05], ...
    'Callback', @(src,~) updateFrame(round(get(src,'Value'))));

handles.playBtn = uicontrol('Parent', handles.panel_playback, 'Style', 'togglebutton', ...
    'String', 'Play', 'Units', 'normalized', ...
    'Position', [0.3 0.2 0.4 0.08], ...
    'BackgroundColor', [0.4 0.8 0.4], ...
    'Callback', @(src,~) togglePlay(src));

% Offset Dropdown
uicontrol('Parent', handles.panel_playback, 'Style', 'text', ...
    'String', 'Eval Pt Offset (in):', ...
    'Units', 'normalized', 'Position', [0.05 0.6 0.9 0.05]);
offset_options = {'-1', '0', '+1'};
handles.offsetMenu = uicontrol('Parent', handles.panel_playback, 'Style', 'popupmenu', ...
    'String', offset_options, ...
    'Units', 'normalized', 'Position', [0.05 0.55 0.9 0.05], ...
    'Callback', @(src,~) updateOffset(src));

% Filtering Panel
handles.panel_filters = uipanel('Parent', fig, 'Title', 'Filtering Methods', ...
    'FontSize', 10, 'BackgroundColor', [0.8 1 0.8], ...
    'Units', 'normalized', 'Position', [0.01 0.42 0.25 0.5]);

filter_names = {'None', 'MovingAvg', 'SavitzkyGolay', ...
                'Butter6', 'Butter8', 'Butter10', 'Butter12', ...
                'QuinticSpline', 'Lowess'};

handles.filterChecks = gobjects(length(filter_names),1);
for i = 1:length(filter_names)
    handles.filterChecks(i) = uicontrol('Parent', handles.panel_filters, ...
        'Style', 'checkbox', 'String', filter_names{i}, ...
        'Units', 'normalized', ...
        'Position', [0.05, 0.9 - (i-1)*0.07, 0.9, 0.06], ...
        'Value', strcmp(filter_names{i}, 'None'), ...
        'Callback', @(src,~) selectFilter(src, filter_names{i}));
end

% Store and continue
guidata(fig, handles);
%% === Helper: Update Offset ===
function updateOffset(src)
    handles = guidata(src);
    val = get(src, 'Value');
    opts = get(src, 'String');
    inch_offset = str2double(opts{val});
    handles.evalOffset = inch_offset * 0.0254;  % Convert inches to meters
    guidata(src, handles);
    updateFrame(handles.frame);
end

%% === Helper: Select Filter ===
function selectFilter(src, selected)
    handles = guidata(src);
    for k = 1:length(handles.filterChecks)
        if handles.filterChecks(k) ~= src
            set(handles.filterChecks(k), 'Value', 0);
        end
    end
    handles.currentFilter = selected;
    guidata(src, handles);
    updateFrame(handles.frame);
end

%% === Apply Filter ===
function filtered = applyFilter(data, method)
    switch method
        case 'None'
            filtered = data;
        case 'MovingAvg'
            filtered = movmean(data, 5);
        case 'SavitzkyGolay'
            filtered = sgolayfilt(data, 3, 9);
        case {'Butter6','Butter8','Butter10','Butter12'}
            cutoff = str2double(extractAfter(method,'Butter'));
            [b,a] = butter(4, cutoff/30, 'low');
            filtered = filtfilt(b, a, data);
        case 'QuinticSpline'
            t = linspace(0,1,size(data,1));
            filtered = zeros(size(data));
            for j = 1:size(data,2)
                [pp,~] = spaps(t, data(:,j)', 1e-6);
                filtered(:,j) = fnval(pp, t)';
            end
        case 'Lowess'
            filtered = zeros(size(data));
            for j = 1:size(data,2)
                filtered(:,j) = smooth(data(:,j), 0.1, 'lowess');
            end
        otherwise
            error('Unknown filter: %s', method);
    end
end

%% === Compute Kinematics ===
function [acc, alpha, omega] = computeKinematics(pos, ori, t)
    dt = mean(diff(t));
    vel = gradient(pos, dt);
    acc = gradient(vel, dt);
    omega = gradient(ori, dt);
    alpha = gradient(omega, dt);
end

%% === Compute Dynamics ===
function [F, Tau, evalPt, shaftDir, cgPt] = computeDynamics(pos, ori, frame, offset, I, m)
    % Extract shaft direction and define eval point and CG
    midhands = pos(frame,1:3);
    clubface = pos(frame,4:6);
    shaftVec = clubface - midhands;
    shaftDir = shaftVec / norm(shaftVec);

    % Define evaluation point (where F is applied)
    evalPt = midhands + shaftDir * offset;

    % Define CG (10 cm behind clubface)
    cgPt = clubface - shaftDir * 0.10;

    % Filtered orientation and position
    [accel, alpha, omega] = computeKinematics(pos(:,1:3), ori, linspace(0,1,size(pos,1)));

    % Force = mass * linear acceleration
    F = m * accel(frame, :);

    % Moment arm from eval point to CG
    r = cgPt - evalPt;

    % Torque = I*alpha + omega x (I*omega)
    Tau_ang = (I * alpha(frame,:)' + cross(omega(frame,:)', I*omega(frame,:)'))';
    Tau_force = cross(r, F);  % Moment due to force
    Tau = Tau_ang - Tau_force;
end
function updateFrame(frame)
    handles = guidata(gcf);
    handles.frame = frame;
    data = handles.swingData;
    pos = data(:,1:6); % Assume [MPx MPy MPz CFx CFy CFz]
    ori = data(:,7:9); % Orientation data assumed (if present)

    % Apply current filter
    pos_filtered = applyFilter(pos, handles.currentFilter);
    ori_filtered = applyFilter(ori, handles.currentFilter);

    % Compute dynamics
    [F, Tau, evalPt, shaftDir, cgPt] = computeDynamics(pos_filtered, ori_filtered, ...
                                                       frame, handles.evalOffset, ...
                                                       handles.I, handles.mass);

    % Club visuals
    midhands = pos_filtered(frame,1:3);
    clubface = pos_filtered(frame,4:6);

    % Plot shaft line
    if ~isfield(handles, 'shaftLine')
        handles.shaftLine = plot3(handles.ax, NaN, NaN, NaN, 'k-', 'LineWidth', 2);
    end
    set(handles.shaftLine, 'XData', [midhands(1) clubface(1)], ...
                           'YData', [midhands(2) clubface(2)], ...
                           'ZData', [midhands(3) clubface(3)]);

    % Force vector
    if ~isfield(handles, 'forceVec')
        handles.forceVec = quiver3(handles.ax, 0,0,0,0,0,0, 'r', 'LineWidth', 2, 'AutoScale', 'off');
    end
    set(handles.forceVec, 'XData', evalPt(1), 'YData', evalPt(2), 'ZData', evalPt(3), ...
                          'UData', F(1), 'VData', F(2), 'WData', F(3));

    % Torque vector
    if ~isfield(handles, 'torqueVec')
        handles.torqueVec = quiver3(handles.ax, 0,0,0,0,0,0, 'b', 'LineWidth', 2, 'AutoScale', 'off');
    end
    set(handles.torqueVec, 'XData', evalPt(1), 'YData', evalPt(2), 'ZData', evalPt(3), ...
                           'UData', Tau(1), 'VData', Tau(2), 'WData', Tau(3));

    % Update axis limits based on full motion path (only first time or if needed)
    if ~isfield(handles, 'fixedAxes')
        allpts = [pos_filtered(:,1:3); pos_filtered(:,4:6)];
        margin = 0.3;
        xlim(handles.ax, [min(allpts(:,1))-margin, max(allpts(:,1))+margin]);
        ylim(handles.ax, [min(allpts(:,2))-margin, max(allpts(:,2))+margin]);
        zlim(handles.ax, [min(allpts(:,3))-margin, max(allpts(:,3))+margin]);
        handles.fixedAxes = true;
    end

    % If recording
    if isfield(handles, 'recording') && handles.recording && isfield(handles, 'videoObj')
        frameCap = getframe(handles.ax);
        writeVideo(handles.videoObj, frameCap);
    end

    guidata(gcf, handles);
end
%% === Toggle Playback ===
function togglePlay(src)
    handles = guidata(src);
    set(src, 'String', 'Pause');
    handles.playing = true;
    guidata(src, handles);

    while handles.playing && ishandle(handles.slider)
        i = round(get(handles.slider, 'Value'));
        if i >= size(handles.swingData,1)
            i = 1;
        end
        set(handles.slider, 'Value', i+1);
        updateFrame(i+1);
        pause(0.03);  % 30 fps

        handles = guidata(src);  % Refresh in loop
        if get(src, 'Value') == 0
            handles.playing = false;
            set(src, 'String', 'Play');
            guidata(src, handles);
            break;
        end
    end
end

%% === Toggle Recording ===
function toggleRecord(~, ~)
    handles = guidata(gcf);
    if ~handles.recording
        [file, path] = uiputfile('*.mp4', 'Save Recording As...');
        if isequal(file,0), return; end
        handles.videoObj = VideoWriter(fullfile(path, file), 'MPEG-4');
        handles.videoObj.FrameRate = 30;
        open(handles.videoObj);
        handles.recording = true;
    else
        close(handles.videoObj);
        handles.recording = false;
    end
    guidata(gcf, handles);
end

%% === Calculate Button Callback ===
function calculateSwingPlane(~, ~)
    handles = guidata(gcf);
    frameImpact = round(handles.keyFrames(1));  % Assume row 1 = impact frame
    N_before = 5;  % Could be GUI user setting
    N_after = 5;

    range = max(1, frameImpact - N_before) : min(size(handles.swingData,1), frameImpact + N_after);
    clubPath = handles.swingData(range, 4:6);  % clubface XYZ

    % Fit plane with PCA
    [~, ~, V] = svd(bsxfun(@minus, clubPath, mean(clubPath,1)));
    normalVec = V(:,3);
    handles.swingPlaneNormal = normalVec;
    handles.swingPlanePoint = mean(clubPath,1);

    % Redraw current frame with new torque projection
    updateFrame(handles.frame);
    guidata(gcf, handles);
end
%% === Add Calculate Button to Playback Panel ===
uicontrol('Parent', handles.panel_playback, 'Style', 'pushbutton', ...
    'String', 'Calculate Plane & Torques', ...
    'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.08], ...
    'FontSize', 9, ...
    'Callback', @calculateSwingPlane);

%% === Projected Torque Plot (2D) Window ===
handles.torqueFig = figure('Name', 'Projected Torque on Swing Plane', ...
    'Color', [1 1 1], ...
    'Position', [1500, 300, 500, 300]);
handles.torqueAxes = axes('Parent', handles.torqueFig);
xlabel(handles.torqueAxes, 'Frame');
ylabel(handles.torqueAxes, 'Projected Torque (Nm)');
grid(handles.torqueAxes, 'on');

%% === Update Frame to Show Projected Torque on Plot ===
function updateFrame(frame)
    handles = guidata(gcf);
    handles.frame = frame;

    % Same filtering and dynamic calc as before
    data = handles.swingData;
    pos = data(:,1:6);
    ori = data(:,7:9);
    pos_filtered = applyFilter(pos, handles.currentFilter);
    ori_filtered = applyFilter(ori, handles.currentFilter);
    [F, Tau, evalPt, shaftDir, cgPt] = computeDynamics(pos_filtered, ori_filtered, ...
                                                       frame, handles.evalOffset, ...
                                                       handles.I, handles.mass);

    % Plot updates: Shaft + Vectors
    midhands = pos_filtered(frame,1:3);
    clubface = pos_filtered(frame,4:6);
    if isfield(handles, 'shaftLine')
        set(handles.shaftLine, 'XData', [midhands(1), clubface(1)], ...
                               'YData', [midhands(2), clubface(2)], ...
                               'ZData', [midhands(3), clubface(3)]);
    end

    set(handles.forceVec, 'XData', evalPt(1), 'YData', evalPt(2), 'ZData', evalPt(3), ...
                          'UData', F(1), 'VData', F(2), 'WData', F(3));
    set(handles.torqueVec, 'XData', evalPt(1), 'YData', evalPt(2), 'ZData', evalPt(3), ...
                           'UData', Tau(1), 'VData', Tau(2), 'WData', Tau(3));

    % Compute projection onto swing plane (if available)
    if isfield(handles, 'swingPlaneNormal')
        Tau_proj = Tau - dot(Tau, handles.swingPlaneNormal) * handles.swingPlaneNormal;
        handles.torqueHistory(frame,:) = Tau_proj;
        plot(handles.torqueAxes, vecnorm(handles.torqueHistory,2,2), 'LineWidth', 2);
    end

    guidata(gcf, handles);
end
