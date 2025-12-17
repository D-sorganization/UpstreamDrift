% GolfClubInverseDynamicsGUI.m - Complete Assembled Script
% All components merged: GUI layout, playback, filtering, inverse dynamics, recording

function GolfClubInverseDynamicsGUI()
close all; clc;
fig = figure('Name', 'Golf Club Inverse Dynamics GUI', 'NumberTitle', 'off', ...
    'Color', [0.9 1 0.9], 'Position', [100, 100, 1400, 800]);

handles = struct();
handles.recording = false;
[file, path] = uigetfile('*.xlsx', 'Select the swing data file');
if isequal(file,0), disp('User canceled file selection'); return; end
fullpath = fullfile(path, file);
[~, sheets] = xlsfinfo(fullpath);
prov1_data = readmatrix(fullpath, 'Sheet', sheets{1});
wiffle_data = readmatrix(fullpath, 'Sheet', sheets{2});
handles.swing.ProV1 = prov1_data;
handles.swing.Wiffle = wiffle_data;
handles.time = linspace(0, 1, size(prov1_data,1));
handles.currentBall = 'ProV1';
handles.currentFilter = 'None';
handles.evalOffset = 0;
handles.frame = 1;
handles.mass = 0.2;
handles.I = diag([0.001, 0.002, 0.0015]);

handles.ax = axes('Parent', fig, 'Position', [0.3 0.1 0.65 0.8]);
axis(handles.ax, 'equal'); grid(handles.ax, 'on'); view(handles.ax, 3);
xlabel(handles.ax, 'X (m)'); ylabel(handles.ax, 'Y (m)'); zlabel(handles.ax, 'Z (m)');
hold(handles.ax, 'on');
handles.club = plot3(handles.ax, NaN, NaN, NaN, 'ko-', 'LineWidth', 2);
handles.forceVec = quiver3(handles.ax, 0,0,0,0,0,0, 'r', 'LineWidth', 2, 'AutoScale', 'off');
handles.torqueVec = quiver3(handles.ax, 0,0,0,0,0,0, 'b', 'LineWidth', 2, 'AutoScale', 'off');

handles.panel = uipanel('Parent', fig, 'Title', 'Controls', 'FontSize', 10, ...
    'BackgroundColor', [0.8 1 0.8], 'Units', 'normalized', 'Position', [0.01 0.1 0.25 0.8]);
uicontrol('Parent', handles.panel, 'Style', 'text', 'String', 'Ball Type', 'Units', 'normalized', 'Position', [0.05 0.9 0.9 0.05]);
handles.ballMenu = uicontrol('Parent', handles.panel, 'Style', 'popupmenu', 'String', {'ProV1', 'Wiffle'}, 'Units', 'normalized', 'Position', [0.05 0.85 0.9 0.05], 'Callback', @(src,~) updateBall(src));

filters = {'None', 'MovingAvg', 'SavitzkyGolay', 'Butter6', 'Butter8', 'Butter10', 'Butter12', 'QuinticSpline'};
handles.filterChecks = gobjects(length(filters),1);
for i = 1:length(filters)
    handles.filterChecks(i) = uicontrol('Parent', handles.panel, 'Style', 'checkbox', 'String', filters{i}, 'Units', 'normalized', 'Position', [0.05, 0.75 - (i-1)*0.05, 0.9, 0.05], 'Value', strcmp(filters{i},'None'), 'Callback', @(src,~) selectFilter(src, filters{i}));
end

uicontrol('Parent', handles.panel, 'Style', 'text', 'String', 'Eval Pt Offset (in):', 'Units', 'normalized', 'Position', [0.05 0.35 0.9 0.05]);
offset_options = {'-1', '0', '+1'};
handles.offsetMenu = uicontrol('Parent', handles.panel, 'Style', 'popupmenu', 'String', offset_options, 'Units', 'normalized', 'Position', [0.05 0.3 0.9 0.05], 'Callback', @(src,~) updateOffset(src));

handles.playBtn = uicontrol('Parent', handles.panel, 'Style', 'togglebutton', 'String', 'Play', 'Units', 'normalized', 'Position', [0.3 0.15 0.4 0.08], 'Callback', @(src,~) playToggle(src));
handles.slider = uicontrol('Parent', handles.panel, 'Style', 'slider', 'Min', 1, 'Max', size(prov1_data,1), 'Value', 1, 'SliderStep', [1/(size(prov1_data,1)-1) 0.1], 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.05], 'Callback', @(src,~) updateFrame(round(get(src,'Value'))));
handles.recordBtn = uicontrol('Parent', handles.panel, 'Style', 'togglebutton', 'String', 'Record', 'Units', 'normalized', 'Position', [0.05 0.2 0.9 0.05], 'BackgroundColor', [1.0 0.6 0.0], 'Callback', @(src,~) toggleRecord(src));

guidata(fig, handles);
updateFrame(1);
end

% Insert additional function definitions here: applySmoothing, updateFrame, playToggle, etc.
% Supporting functions for GolfClubInverseDynamicsGUI

function updateBall(src)
handles = guidata(src);
options = get(src, 'String');
selection = options{get(src, 'Value')};
handles.currentBall = selection;
guidata(src, handles);
updateFrame(handles.frame);
end

function selectFilter(src, selectedFilter)
handles = guidata(src);
for i = 1:length(handles.filterChecks)
    if handles.filterChecks(i) ~= src
        set(handles.filterChecks(i), 'Value', 0);
    else
        set(handles.filterChecks(i), 'Value', 1);
    end
end
handles.currentFilter = selectedFilter;
guidata(src, handles);
updateFrame(handles.frame);
end

function updateOffset(src)
handles = guidata(src);
offsets = get(src, 'String');
offset_in = str2double(offsets{get(src, 'Value')});
handles.evalOffset = offset_in * 0.0254;
guidata(src, handles);
updateFrame(handles.frame);
end

function toggleRecord(src)
handles = guidata(src);
if get(src, 'Value')
    [file, path] = uiputfile('*.mp4', 'Save Swing Recording As...');
    if isequal(file,0)
        set(src, 'Value', 0);
        return;
    end
    filename = fullfile(path, file);
    handles.videoObj = VideoWriter(filename, 'MPEG-4');
    handles.videoObj.FrameRate = 60;
    open(handles.videoObj);
    handles.recording = true;
    set(src, 'String', 'Stop Recording', 'BackgroundColor', [1.0 0.4 0.4]);
else
    handles.recording = false;
    if isfield(handles, 'videoObj') && ~isempty(handles.videoObj)
        close(handles.videoObj);
    end
    handles.videoObj = [];
    set(src, 'String', 'Record', 'BackgroundColor', [1.0 0.6 0.0]);
end
guidata(src, handles);
end

function playToggle(src)
handles = guidata(src);
if get(src, 'Value')
    set(src, 'String', 'Pause');
    while get(src, 'Value') && ishandle(src)
        handles = guidata(src);
        i = handles.frame;
        i = mod(i, size(handles.swing.(handles.currentBall),1)) + 1;
        handles.frame = i;
        guidata(src, handles);
        set(handles.slider, 'Value', i);
        updateFrame(i);
        pause(1/60);
    end
    set(src, 'String', 'Play');
else
    set(src, 'String', 'Play');
end
end

function updateFrame(frame)
handles = guidata(gcf);
handles.frame = frame;
data = handles.swing.(handles.currentBall);

if size(data,2) < 7
    pos_raw = data(:,2:4) / 100;
    ori = zeros(size(pos_raw));
else
    pos_raw = data(:,2:4) / 100;
    ori_raw = data(:,5:7);
    ori = applySmoothing(ori_raw, handles.currentFilter);
end

pos = applySmoothing(pos_raw, handles.currentFilter);
[accel, alpha, omega] = computeKinematics(pos, ori, handles.time);
r_offset = [0, 0, handles.evalOffset];
[F, Tau] = computeDynamics(accel(frame,:), alpha(frame,:), omega(frame,:), handles.I, handles.mass, r_offset);

set(handles.club, 'XData', pos(frame,1), 'YData', pos(frame,2), 'ZData', pos(frame,3));
mp = mean([pos(frame, :); pos(frame, :) + [0, 0, -0.3]], 1); % rough midpoint between hands
set(handles.forceVec, 'XData', mp(1), 'YData', mp(2), 'ZData', mp(3), ...
    'UData', F(1), 'VData', F(2), 'WData', F(3));

set(handles.torqueVec, 'XData', mp(1), 'YData', mp(2), 'ZData', mp(3), ...
    'UData', Tau(1), 'VData', Tau(2), 'WData', Tau(3));


if handles.recording
    frame_capture = getframe(handles.ax);
    writeVideo(handles.videoObj, frame_capture);
end
guidata(gcf, handles);
end

function [accel, alpha, omega] = computeKinematics(pos, ori, t)
dt = mean(diff(t));
vel = gradient(pos, dt);
accel = gradient(vel, dt);
omega = gradient(ori, dt);
alpha = gradient(omega, dt);
end

function [F, Tau] = computeDynamics(accel, alpha, omega, I, m, r)
F = m * accel;
Tau = I * alpha' + cross(omega', I * omega') + cross(r', F');
Tau = Tau';
end

function smoothData = applySmoothing(rawData, method)
switch method
    case 'None'
        smoothData = rawData;
    case 'MovingAvg'
        smoothData = movmean(rawData, 5);
    case 'SavitzkyGolay'
        smoothData = sgolayfilt(rawData, 3, 9);
    case {'Butter6','Butter8','Butter10','Butter12'}
        order = 4;
        cutoff = str2double(extractAfter(method, 'Butter'));
        [b,a] = butter(order, cutoff/30, 'low');
        smoothData = filtfilt(b, a, rawData);
    case 'QuinticSpline'
    t = linspace(0, 1, size(rawData,1));
    smoothData = zeros(size(rawData));
    for j = 1:size(rawData,2)
        [pp, ~] = spaps(t, rawData(:,j)', 1e-6);
        smoothData(:,j) = fnval(pp, t)';
    end
    otherwise
        error('Unknown smoothing method: %s', method);
end
end
