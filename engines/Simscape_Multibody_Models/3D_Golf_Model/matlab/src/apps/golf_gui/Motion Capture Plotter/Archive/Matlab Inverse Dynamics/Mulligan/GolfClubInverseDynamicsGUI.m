function GolfClubInverseDynamicsGUI()
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
