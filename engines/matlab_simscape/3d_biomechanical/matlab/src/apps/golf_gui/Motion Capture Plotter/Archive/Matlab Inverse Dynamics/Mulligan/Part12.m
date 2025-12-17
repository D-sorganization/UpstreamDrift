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
