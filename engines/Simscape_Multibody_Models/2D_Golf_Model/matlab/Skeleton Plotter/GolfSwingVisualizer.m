% Save this code as GolfSwingVisualizer.m
classdef GolfSwingVisualizer < handle
    % GolfSwingVisualizer Creates a GUI to visualize 3D golf swing data.
    % (Version: Velocity face normal, custom colors, adjusted ball pos & lighting)
    %
    %   Usage:
    %       viz = GolfSwingVisualizer(BASEQ_table, ZTCFQ_table, DELTAQ_table);
    %
    %   Inputs:
    %       BASEQ_table, ZTCFQ_table, DELTAQ_table: MATLAB tables containing
    %                            swing data (must have identical row counts).

    % == Public Properties ==
    properties (Access = public)
        NumFrames       % Total number of frames (rows) in the BASEQ table
    end

    % == Private Properties ==
    properties (Access = private)
        Config, DataSets, MaxForceMag, MaxTorqueMag
        Handles, PlotHandles
        CurrentFrame, IsPlaying, IsRecording, VideoWriterObj, PlaybackTimer
    end

    % == Public Methods ==
    methods (Access = public)
        % --- Constructor ---
        function obj = GolfSwingVisualizer(BASEQ_table, ZTCFQ_table, DELTAQ_table)
            % 1. Basic Input Validation
            if nargin < 3 || ~istable(BASEQ_table) || ~istable(ZTCFQ_table) || ~istable(DELTAQ_table)
                error('GolfSwingVisualizer:InvalidInput', 'Requires three MATLAB table inputs (BASEQ_table, ZTCFQ_table, DELTAQ_table).');
            end
            % Validate BASEQ first to get NumFrames
            try
                obj.validateInputData(BASEQ_table, 'BASEQ_table');
                obj.NumFrames = height(BASEQ_table);
                if obj.NumFrames < 2 % Need at least 2 frames for velocity calculation
                    warning('GolfSwingVisualizer:NotEnoughFrames', 'Requires at least 2 frames for velocity-based face normal. Face normal may default to fallback.');
                end
                if obj.NumFrames == 0
                    error('GolfSwingVisualizer:NoFrames', 'Input table BASEQ_table contains no rows (frames).');
                end
            catch ME
                rethrow(ME);
            end
            % Validate other tables (ensure same number of frames)
             try
                obj.validateInputData(ZTCFQ_table, 'ZTCFQ_table');
                obj.validateInputData(DELTAQ_table, 'DELTAQ_table');
             catch ME
                 rethrow(ME);
             end

            % 2. Setup Configuration
            obj.setupConfig();

            % 3. Store Data and Calculate Initial Values
            obj.DataSets = {
                struct('Name', 'BASE', 'Data', BASEQ_table, 'ColorForce', obj.Config.Colors.Force{1}, 'ColorTorque', obj.Config.Colors.Torque{1});
                struct('Name', 'ZTCF', 'Data', ZTCFQ_table, 'ColorForce', obj.Config.Colors.Force{2}, 'ColorTorque', obj.Config.Colors.Torque{2});
                struct('Name', 'DELTA', 'Data', DELTAQ_table,'ColorForce', obj.Config.Colors.Force{3}, 'ColorTorque', obj.Config.Colors.Torque{3})
            };
            obj.calculateScalingMagnitudes(); % Calculate max force/torque

            % 4. Initialize State
            obj.CurrentFrame = 1;
            obj.IsPlaying = false;
            obj.IsRecording = false;
            obj.VideoWriterObj = [];
            obj.PlaybackTimer = [];

            % 5. Create GUI and Plot Objects
            obj.createGUI(); % Creates Figure and Axes
            hold(obj.Handles.Axes, 'on'); % HOLD ON before creating plot objects
            obj.initializePlotObjects(); % Creates plot objects (surf, quiver)
            obj.setupAxes(); % Sets up axes appearance and limits

            % 6. Set Figure Close Request Callback
            set(obj.Handles.Figure, 'CloseRequestFcn', @obj.closeFigureCallback);

            % 7. Perform Initial Plot
            try
                obj.updatePlot();
            catch ME
                warning('GolfSwingVisualizer:InitialPlotError', 'Error during initial plot: %s', ME.message);
                 % Continue execution so figure is visible, but plot might be wrong
            end
        end % End Constructor

        % --- Destructor ---
        function delete(obj)
            obj.cleanupResources();
        end
    end % End Public Methods

    % == Private Methods ==
    methods (Access = private)
        % --- Initialization Helpers ---
        function validateInputData(obj, dataTable, tableName)
            % Basic validation for required columns and data types
            requiredCols = {'Buttx', 'Butty', 'Buttz', 'CHx', 'CHy', 'CHz', ...
                              'MPx', 'MPy', 'MPz', 'LWx', 'LWy', 'LWz', ...
                              'LEx', 'LEy', 'LEz', 'LSx', 'LSy', 'LSz', ...
                              'RWx', 'RWy', 'RWz', 'REx', 'REy', 'REz', ...
                              'RSx', 'RSy', 'RSz', 'HUBx', 'HUBy', 'HUBz', ...
                              'TotalHandForceGlobal', 'EquivalentMidpointCoupleGlobal'};
            presentCols = dataTable.Properties.VariableNames;
            for i = 1:length(requiredCols)
                colName = requiredCols{i};
                if ~ismember(colName, presentCols)
                    error('GolfSwingVisualizer:MissingColumn', 'Input table %s is missing required column: %s', tableName, colName);
                end
                colData = dataTable.(colName);
                if ~isnumeric(colData)
                     error('GolfSwingVisualizer:InvalidColumnType', 'Column %s in table %s must be numeric.', colName, tableName);
                end
                 % Check frame count consistency after BASEQ is processed
                 if ~strcmp(tableName, 'BASEQ_table') && isprop(obj,'NumFrames') && ~isempty(obj.NumFrames) && height(dataTable) ~= obj.NumFrames
                      error('GolfSwingVisualizer:FrameMismatch', 'Input table %s has %d rows, expected %d based on BASEQ_table.', tableName, height(dataTable), obj.NumFrames);
                 end
            end
             % Validate vector columns are Nx3
             vectorCols = {'TotalHandForceGlobal', 'EquivalentMidpointCoupleGlobal'};
             for i = 1:length(vectorCols)
                 colName = vectorCols{i};
                 if ismember(colName, presentCols)
                     colData = dataTable.(colName);
                     if size(colData, 2) ~= 3
                         error('GolfSwingVisualizer:InvalidColumnSize', 'Column %s in table %s must have 3 columns (Nx3).', colName, tableName);
                     end
                 end
             end
        end

        function setupConfig(obj)
            obj.Config = struct();
            % --- Colors ---
            % Body/Environment
            obj.Config.Colors.Skin = [0.9, 0.75, 0.65]; % Beige-like skin tone
            obj.Config.Colors.Shirt = [0.2, 0.4, 0.8];  % Blue shirt color
            obj.Config.Colors.Shaft = [0 0 0];          % Black shaft
            obj.Config.Colors.Clubhead = [0.6 0.6 0.6]; % Grey clubhead
            obj.Config.Colors.FaceNormal = [0 1 0];     % Green face normal
            obj.Config.Colors.Ground = [0.4, 0.6, 0.2];  % Greenish ground (used for cmap calc)
            obj.Config.Colors.Ball = [1 1 1];        % White ball
            % GUI Elements (Reverted)
            obj.Config.Colors.FigureBackground = [0.9, 1, 0.9]; % Light green
            obj.Config.Colors.AxesBackground = [1, 1, 0.8];   % Soft yellow
            obj.Config.Colors.PanelBackground = [0.8, 1, 0.8];  % Lighter green
            obj.Config.Colors.TextBackground = [1 1 1];      % White text bg
            obj.Config.Colors.RecordIdle = [1.0 0.6 0.0];    % Orange record idle
            obj.Config.Colors.RecordActive = [1.0 0.4 0.4];   % Red record active
            obj.Config.Colors.PlayButton = [0.4 0.8 0.4];    % Green play button
            % Vectors & Legend (Classic Colors)
            obj.Config.Colors.Force = {[1 0 0], [0 0 1], [0 0.5 0]};     % Red, Blue, Dark Green
            obj.Config.Colors.Torque = {[0.5 0 0.5], [0 0.5 0.5], [1 0.5 0]}; % Purple, Teal, Orange
            obj.Config.Colors.LegendText = obj.Config.Colors.Force; % Forces first
            obj.Config.Colors.LegendText(4:6) = obj.Config.Colors.Torque; % Then torques

            % --- Sizes ---
            inches_to_meters = 0.0254;
            obj.Config.Sizes.ClubheadLength = 4.5 * inches_to_meters; % Longer axis (along shaft)
            obj.Config.Sizes.ClubheadWidth = 3.5 * inches_to_meters;  % Shorter axes (face width/depth)
            obj.Config.Sizes.ShaftDiameter = 0.5 * inches_to_meters;
            obj.Config.Sizes.ForearmDiameter = 2.8 * inches_to_meters;
            obj.Config.Sizes.UpperarmDiameter = 3.5 * inches_to_meters;
            obj.Config.Sizes.ShoulderNeckDiameter = 4.5 * inches_to_meters;
            obj.Config.Sizes.BallDiameter = 1.68 * inches_to_meters; % Regulation golf ball diameter
            obj.Config.Sizes.PlotMargin = 0.3;
            obj.Config.Sizes.GroundPlaneZ = -0.6; % Ground plane height
            obj.Config.Sizes.VelocityEps = 1e-4; % Threshold for detecting zero velocity
            obj.Config.Sizes.ParallelEps = 1e-4; % Threshold for detecting parallel vectors

            % --- Labels & Text ---
            obj.Config.Font.Size = 10; obj.Config.Font.SizeSmall = 9;
            obj.Config.Labels.FigureName = 'Golf Swing Visualizer';
            obj.Config.Labels.CheckboxPanelTitle = 'Segments and Vectors';
            obj.Config.Labels.PlaybackPanelTitle = 'Playback and Scaling';
            obj.Config.Labels.ZoomPanelTitle = 'Zoom';
            obj.Config.Labels.LegendPanelTitle = 'Legend';
            obj.Config.Labels.Checkboxes = {'Force BASE', 'Force ZTCF', 'Force DELTA', 'Torque BASE', 'Torque ZTCF', 'Torque DELTA', 'Shaft & Club', 'Face Normal', 'Left Forearm', 'Left Upper Arm', 'Left Shoulder-Neck', 'Right Forearm', 'Right Upper Arm', 'Right Shoulder-Neck'};
            obj.Config.Labels.LegendEntries = {'BASE (Force)', 'ZTCF (Force)', 'DELTA (Force)', 'BASE (Torque)', 'ZTCF (Torque)', 'DELTA (Torque)'};
            % Mapping from label to checkbox index
            obj.Config.CheckboxMapping = struct('Force_BASE', 1, 'Force_ZTCF', 2, 'Force_DELTA', 3, 'Torque_BASE', 4, 'Torque_ZTCF', 5, 'Torque_DELTA', 6, 'Shaft_Club', 7, 'Face_Normal', 8, 'Left_Forearm', 9, 'Left_Upper_Arm', 10, 'Left_Shoulder_Neck', 11, 'Right_Forearm', 12, 'Right_Upper_Arm', 13, 'Right_Shoulder_Neck', 14);

            % --- Playback, Scaling, Zoom, Recording Config ---
            obj.Config.Playback.TimerPeriod = 0.033; % 30 fps base speed (1/30 s)
            obj.Config.Playback.MinSpeed = 0.1; obj.Config.Playback.MaxSpeed = 3.0; obj.Config.Playback.DefaultSpeed = 1.0;
            obj.Config.Scaling.MinVectorScale = 0.1; obj.Config.Scaling.MaxVectorScale = 9.0; obj.Config.Scaling.DefaultVectorScale = 1.0;
            obj.Config.Zoom.MinFactor = 0.1; obj.Config.Zoom.MaxFactor = 5.0; obj.Config.Zoom.DefaultFactor = 1.0;
            obj.Config.Recording.FrameRate = 30;
            obj.Config.Recording.DefaultFileName = 'golf_swing_recording.mp4';
            obj.Config.Recording.FileType = '*.mp4';
            obj.Config.Recording.FileDescription = 'Save Swing Recording As...';
        end

        function calculateScalingMagnitudes(obj)
            % Calculate max magnitudes from BASE data for consistent scaling
            baseTable = obj.DataSets{1}.Data;
            forceNorms = vecnorm(baseTable.TotalHandForceGlobal, 2, 2);
            torqueNorms = vecnorm(baseTable.EquivalentMidpointCoupleGlobal, 2, 2);
            % Use max ignoring NaNs, default to 1 if all NaN or zero
            obj.MaxForceMag = max(forceNorms, [], 'omitnan');
            if isempty(obj.MaxForceMag) || obj.MaxForceMag == 0 || isnan(obj.MaxForceMag); obj.MaxForceMag = 1; end
            obj.MaxTorqueMag = max(torqueNorms, [], 'omitnan');
            if isempty(obj.MaxTorqueMag) || obj.MaxTorqueMag == 0 || isnan(obj.MaxTorqueMag); obj.MaxTorqueMag = 1; end
        end

        function createGUI(obj)
            % Create the main figure and axes
            obj.Handles.Figure = figure('Name', obj.Config.Labels.FigureName, ...
                'NumberTitle', 'off', 'Color', obj.Config.Colors.FigureBackground, ...
                'Position', [100, 100, 1400, 800], 'Visible', 'off'); % Start invisible

            obj.Handles.Axes = axes('Parent', obj.Handles.Figure, ...
                'Color', obj.Config.Colors.AxesBackground, ...
                'Position', [0.25 0.05 0.6 0.9]); % Main plot area

            % Create control panels
            obj.createCheckboxPanel();
            obj.createPlaybackPanel();
            obj.createZoomPanel();
            obj.createLegendPanel();

            % Create view buttons, record button, etc.
            obj.createViewButtons();
            obj.createRecordButton();
            obj.createShowHideLegendButton();

            % Add Timestamp display
            obj.Handles.TimestampText = uicontrol('Parent', obj.Handles.Figure, ...
                'Style', 'text', 'String', 'Frame: 0 / Time: 0.00s', ...
                'Units', 'normalized', 'Position', [0.25 0.96 0.2 0.03], ... % Position above axes
                'BackgroundColor', obj.Config.Colors.FigureBackground, ...
                'FontSize', obj.Config.Font.SizeSmall, 'HorizontalAlignment', 'left');

            % Make figure visible after setup
            obj.Handles.Figure.Visible = 'on';
        end

        function createCheckboxPanel(obj)
             panelPos = [0.01 0.4 0.22 0.55];
            obj.Handles.PanelCheckboxes = uipanel('Parent', obj.Handles.Figure, 'Title', obj.Config.Labels.CheckboxPanelTitle, 'FontSize', obj.Config.Font.Size, 'BackgroundColor', obj.Config.Colors.PanelBackground, 'Units', 'normalized', 'Position', panelPos);
            numCheckboxes = length(obj.Config.Labels.Checkboxes);
            obj.Handles.Checkboxes = gobjects(numCheckboxes, 1);
            checkboxHeight = 1 / (numCheckboxes + 1); % Dynamic height
            startY = 1 - checkboxHeight * 0.9;
            for k = 1:numCheckboxes
                obj.Handles.Checkboxes(k) = uicontrol('Parent', obj.Handles.PanelCheckboxes, 'Style', 'checkbox', ...
                    'String', obj.Config.Labels.Checkboxes{k}, ...
                    'Units', 'normalized', 'Position', [0.05, startY - (k-1)*checkboxHeight, 0.9, checkboxHeight*0.8], ...
                    'BackgroundColor', obj.Config.Colors.PanelBackground, 'FontSize', obj.Config.Font.SizeSmall, ...
                    'Value', 1, 'Callback', @obj.checkboxCallback); % Default to checked
            end
        end

        function createPlaybackPanel(obj)
             panelPos = [0.01 0.05 0.22 0.35];
             obj.Handles.PanelPlayback = uipanel('Parent', obj.Handles.Figure, 'Title', obj.Config.Labels.PlaybackPanelTitle, 'FontSize', obj.Config.Font.Size, 'BackgroundColor', obj.Config.Colors.PanelBackground, 'Units', 'normalized', 'Position', panelPos);

             % Slider creation using helper
             obj.Handles.SpeedSlider = obj.createSliderWithLabel(obj.Handles.PanelPlayback, 'Playback Speed', [0.05 0.8 0.9 0.18], obj.Config.Playback.MinSpeed, obj.Config.Playback.MaxSpeed, obj.Config.Playback.DefaultSpeed, @(src, ~) obj.speedSliderCallback(src.Value));
             obj.Handles.ScaleSlider = obj.createSliderWithLabel(obj.Handles.PanelPlayback, 'Vector Scale', [0.05 0.6 0.9 0.18], obj.Config.Scaling.MinVectorScale, obj.Config.Scaling.MaxVectorScale, obj.Config.Scaling.DefaultVectorScale, @obj.scaleSliderCallback);

             % Frame Slider - ensure steps are valid
             sliderStep = [1/(obj.NumFrames-1), 10/(obj.NumFrames-1)];
             if obj.NumFrames <= 1; sliderStep = [1, 1]; end % Avoid division by zero/negative
             if any(isnan(sliderStep)) || any(isinf(sliderStep)); sliderStep = [0.01, 0.1]; end % Fallback steps

             obj.Handles.FrameSlider = obj.createSliderWithLabel(obj.Handles.PanelPlayback, 'Frame', [0.05 0.4 0.9 0.18], 1, max(1, obj.NumFrames), 1, @obj.frameSliderCallback, sliderStep); % Ensure Max >= Min

             % Play/Pause Button
             obj.Handles.PlayPauseButton = uicontrol('Parent', obj.Handles.PanelPlayback, 'Style', 'togglebutton', ...
                'String', 'Play', 'Units', 'normalized', 'Position', [0.25 0.1 0.5 0.2], ...
                'FontSize', obj.Config.Font.Size, 'BackgroundColor', obj.Config.Colors.PlayButton, ...
                'Callback', @obj.togglePlayPause);
        end

        function sliderHandle = createSliderWithLabel(obj, parent, labelText, position, minVal, maxVal, initialVal, callback, sliderStep)
            % Helper to create a slider with a text label above it
            labelHeightRatio = 0.4; sliderHeightRatio = 0.6;
            labelPos = [position(1), position(2) + position(4)*sliderHeightRatio, position(3), position(4)*labelHeightRatio];
            sliderPos = [position(1), position(2), position(3), position(4)*sliderHeightRatio];

            uicontrol('Parent', parent, 'Style', 'text', 'String', labelText, 'Units', 'normalized', ...
                'Position', labelPos, 'BackgroundColor', obj.Config.Colors.PanelBackground, ...
                'FontSize', obj.Config.Font.SizeSmall, 'HorizontalAlignment', 'center');

            if nargin < 9; sliderStep = [0.01, 0.1]; end % Default slider steps

            sliderHandle = uicontrol('Parent', parent, 'Style', 'slider', ...
                'Min', minVal, 'Max', maxVal, 'Value', initialVal, ...
                'Units', 'normalized', 'Position', sliderPos, ...
                'BackgroundColor', obj.Config.Colors.TextBackground, ...
                'FontSize', obj.Config.Font.SizeSmall, ...
                'SliderStep', sliderStep, 'Callback', callback);
        end

        function createZoomPanel(obj)
            panelPos = [0.01 0.95 0.22 0.04]; % Adjusted Y position slightly
            obj.Handles.PanelZoom = uipanel('Parent', obj.Handles.Figure, 'Title', obj.Config.Labels.ZoomPanelTitle, 'FontSize', obj.Config.Font.Size, 'BackgroundColor', obj.Config.Colors.PanelBackground, 'Units', 'normalized', 'Position', panelPos);
            obj.Handles.ZoomSlider = uicontrol('Parent', obj.Handles.PanelZoom, 'Style', 'slider', ...
                'Min', obj.Config.Zoom.MinFactor, 'Max', obj.Config.Zoom.MaxFactor, 'Value', obj.Config.Zoom.DefaultFactor, ...
                'Units', 'normalized', 'Position', [0.05 0.1 0.9 0.8], ...
                'BackgroundColor', obj.Config.Colors.TextBackground, ...
                'FontSize', obj.Config.Font.SizeSmall, 'Callback', @obj.zoomSliderCallback);
        end

        function createLegendPanel(obj)
             panelPos = [0.86 0.35 0.12 0.3];
            obj.Handles.PanelLegend = uipanel('Parent', obj.Handles.Figure, 'Title', obj.Config.Labels.LegendPanelTitle, 'FontSize', obj.Config.Font.Size, 'BackgroundColor', obj.Config.Colors.TextBackground, 'Units', 'normalized', 'Position', panelPos, 'Visible', 'on'); % Initially visible

            numEntries = length(obj.Config.Labels.LegendEntries);
            entryHeight = 1 / (numEntries + 1); % Dynamic height
            startY = 1 - entryHeight * 0.9;
            for k = 1:numEntries
                uicontrol('Parent', obj.Handles.PanelLegend, 'Style', 'text', ...
                    'String', obj.Config.Labels.LegendEntries{k}, ...
                    'ForegroundColor', obj.Config.Colors.LegendText{k}, ...
                    'BackgroundColor', obj.Config.Colors.TextBackground, ...
                    'Units', 'normalized', 'Position', [0.05, startY - (k-1)*entryHeight, 0.9, entryHeight*0.8], ...
                    'FontSize', obj.Config.Font.SizeSmall, 'HorizontalAlignment', 'left');
            end
        end

        function createViewButtons(obj)
            buttonWidth = 0.12; buttonHeight = 0.05; buttonX = 0.86; buttonSpacing = 0.06; initialY = 0.92;
            viewTypes = {'Face-On', 'Down-the-Line', 'Top-Down', 'Isometric'};
            viewCodes = {'faceon', 'downline', 'topdown', 'iso'};

            for k = 1:length(viewTypes)
                buttonY = initialY - (k-1) * buttonSpacing;
                uicontrol('Parent', obj.Handles.Figure, 'Style', 'pushbutton', 'String', viewTypes{k}, ...
                    'Units', 'normalized', 'Position', [buttonX, buttonY, buttonWidth, buttonHeight], ...
                    'FontSize', obj.Config.Font.Size, 'Callback', @(src, evt) obj.setViewCallback(viewCodes{k}), ...
                    'BackgroundColor', [0.8 0.8 0.8]);
            end
             % Store Y position for placing next button below
             obj.Handles.LastViewButtonY = initialY - length(viewTypes) * buttonSpacing;
        end

         function createShowHideLegendButton(obj)
            buttonWidth = 0.12; buttonHeight = 0.05; buttonX = 0.86;
            buttonY = obj.Handles.LastViewButtonY; % Position below last view button
            obj.Handles.ToggleLegendButton = uicontrol('Parent', obj.Handles.Figure, 'Style', 'togglebutton', ...
                'String', 'Hide Legend', 'Value', 1, ... % Start visible
                'Units', 'normalized', 'Position', [buttonX, buttonY, buttonWidth, buttonHeight], ...
                'BackgroundColor', [0.8 0.8 0.8], 'FontSize', obj.Config.Font.Size, ...
                'Callback', @obj.toggleLegendVisibility);
        end

        function createRecordButton(obj)
            obj.Handles.RecordButton = uicontrol('Parent', obj.Handles.Figure, 'Style', 'togglebutton', ...
                'String', 'Record', 'Units', 'normalized', 'Position', [0.86 0.01 0.12 0.04], ...
                'BackgroundColor', obj.Config.Colors.RecordIdle, 'FontSize', obj.Config.Font.Size, ...
                'Callback', @obj.toggleRecord);
        end

        function initializePlotObjects(obj)
            % Create all plot handles (surf, quiver) and store them
            ax = obj.Handles.Axes;
            obj.PlotHandles = struct();

            % --- Club --- (Using updated colors from setupConfig)
            obj.PlotHandles.ShaftCylinder = surf(ax, nan(21,2), nan(21,2), nan(21,2), ...
                'FaceColor', obj.Config.Colors.Shaft, 'EdgeColor', 'none', ...
                'Tag', 'Shaft', 'Visible', 'off');
            obj.PlotHandles.ClubheadShape = surf(ax, nan(16,31), nan(16,31), nan(16,31), ... % Match NaN size for hemisphere base
                'FaceColor', obj.Config.Colors.Clubhead, 'EdgeColor', 'none', ...
                'Tag', 'Clubhead', 'Visible', 'off');

            % --- Vectors ---
            obj.PlotHandles.ForceQuivers = gobjects(1, 3);
            obj.PlotHandles.TorqueQuivers = gobjects(1, 3);
            for k = 1:3
                obj.PlotHandles.ForceQuivers(k) = quiver3(ax, 0,0,0, 0,0,0, ...
                    'Color', obj.Config.Colors.Force{k}, 'LineWidth', 2, 'MaxHeadSize', 0.5, ...
                    'AutoScale', 'off', 'Visible', 'off', 'Tag', ['Force_', obj.DataSets{k}.Name]);
                obj.PlotHandles.TorqueQuivers(k) = quiver3(ax, 0,0,0, 0,0,0, ...
                    'Color', obj.Config.Colors.Torque{k}, 'LineWidth', 2, 'MaxHeadSize', 0.5, ...
                    'AutoScale', 'off', 'Visible', 'off', 'Tag', ['Torque_', obj.DataSets{k}.Name]);
            end
            obj.PlotHandles.FaceNormalQuiver = quiver3(ax, 0,0,0, 0,0,0, ...
                'Color', obj.Config.Colors.FaceNormal, 'LineWidth', 2, 'MaxHeadSize', 0.5, ...
                'AutoScale', 'off', 'Visible', 'off', 'Tag', 'FaceNormal');

            % --- Body Segments --- (Correct color assignment confirmed)
            segmentNames = {'Left_Forearm', 'Left_Upper_Arm', 'Left_Shoulder_Neck', ...
                            'Right_Forearm', 'Right_Upper_Arm', 'Right_Shoulder_Neck'};
            segmentColors = {obj.Config.Colors.Skin, obj.Config.Colors.Shirt, obj.Config.Colors.Shirt, ...
                             obj.Config.Colors.Skin, obj.Config.Colors.Shirt, obj.Config.Colors.Shirt};
            for k = 1:length(segmentNames)
                name = segmentNames{k};
                color = segmentColors{k};
                obj.PlotHandles.([name, '_Cylinder']) = surf(ax, nan(21,2), nan(21,2), nan(21,2), ...
                    'FaceColor', color, 'EdgeColor', 'none', 'Visible', 'off', 'Tag', [name, '_Cyl']);
                obj.PlotHandles.([name, '_Sphere']) = surf(ax, nan(21), nan(21), nan(21), ...
                    'FaceColor', color, 'EdgeColor', 'none', 'Visible', 'off', 'Tag', [name, '_Sph']);
            end
             obj.PlotHandles.Hub_Sphere = surf(ax, nan(21), nan(21), nan(21), ...
                 'FaceColor', obj.Config.Colors.Shirt, 'EdgeColor', 'none', 'Visible', 'off', 'Tag', 'Hub_Sph');

           % --- Environment --- (Ground plane color calculated directly) ---
            % Ground Plane
            [xlims, ylims, ~] = obj.calculatePlotLimits(obj.DataSets{1}.Data); % Get initial range estimate
            xRange = diff(xlims); yRange = diff(ylims);
            groundX = linspace(xlims(1)-xRange*0.2, xlims(2)+xRange*0.2, 20); % Larger plane
            groundY = linspace(ylims(1)-yRange*0.2, ylims(2)+yRange*0.2, 20);
            [groundMeshX, groundMeshY] = meshgrid(groundX, groundY);
            groundMeshZ = ones(size(groundMeshX)) * obj.Config.Sizes.GroundPlaneZ;
            s = rng;
            rng(1); % Ensure reproducible ground texture
            groundTexture = rand(size(groundMeshZ)); % Random noise texture (0 to 1)
            rng(s);

            % Define the desired colormap for the ground *only*
            groundCmap = [linspace(0.3,0.5,256)', linspace(0.5,0.7,256)', linspace(0.1,0.3,256)'];
            numColors = size(groundCmap, 1);

            % Map the texture values to RGB colors using the groundCmap
            texMin = min(groundTexture(:));
            texMax = max(groundTexture(:));
            if texMax <= texMin % Avoid division by zero/NaN if texture is flat
                indices = ones(size(groundTexture), 'uint16'); % Use integer type
            else
                % Scale texture values to [0, 1] then map to colormap indices
                normTexture = (groundTexture - texMin) / (texMax - texMin);
                indices = uint16(round(1 + normTexture * (numColors - 1))); % Faster mapping
            end
            % indices = max(uint16(1), min(indices, uint16(numColors))); % Clamp indices - already handled by round/uint16

            % Create M x N x 3 RGB CData matrix efficiently
            groundCDataRGB = ind2rgb(indices, groundCmap); % Use ind2rgb for fast conversion

            % Create the ground surface using the pre-calculated RGB CData
            obj.PlotHandles.GroundPlane = surf(ax, groundMeshX, groundMeshY, groundMeshZ, groundCDataRGB, ... % Provide RGB CData
                 'FaceColor', 'texturemap', 'EdgeColor', 'none', ...
                 'Tag', 'GroundPlane', 'Visible', 'on');

            % DO NOT SET AXES COLORMAP HERE (Fixes body color issue)

            % Golf Ball (Using updated Y position)
            ballYPos = -1.63; % <<<< Y POSITION SET HERE <<<<
            ballPos = [0, ballYPos, obj.Config.Sizes.GroundPlaneZ + obj.Config.Sizes.BallDiameter/2]; % Sit on ground
            ballRadius = obj.Config.Sizes.BallDiameter / 2;
            [bx, by, bz] = sphere(20);
            obj.PlotHandles.GolfBall = surf(ax, ballPos(1) + bx * ballRadius, ...
                                             ballPos(2) + by * ballRadius, ...
                                             ballPos(3) + bz * ballRadius, ...
                 'FaceColor', obj.Config.Colors.Ball, 'EdgeColor', 'none', ... % Uses white from setupConfig
                 'Tag', 'GolfBall', 'Visible', 'on');
            % --- End Environment Section ---
        end

        function setupAxes(obj)
            % Configure the appearance of the main axes
            ax = obj.Handles.Axes;
            axis(ax, 'equal'); % Ensure correct aspect ratio
            grid(ax, 'on');
            xlabel(ax, 'X (m)'); ylabel(ax, 'Y (m)'); zlabel(ax, 'Z (m)');
            view(ax, 3); % Default 3D view

            % Calculate and set axis limits based on data range + ground/ball
            [xlims, ylims, zlims] = obj.calculatePlotLimits(obj.DataSets{1}.Data);

            % Ensure limits are valid numbers
            if any(isnan(xlims)) || any(isinf(xlims)); xlims = [-1 1]; end
            if any(isnan(ylims)) || any(isinf(ylims)); ylims = [-1 2]; end % Adjust default Y limits for ball
            if any(isnan(zlims)) || any(isinf(zlims)); zlims = [-1 2]; end % Adjust default Z limits

             % Explicitly include ground plane Z, slightly below for visibility
             % And ensure Z max is sufficiently above the highest point (e.g., top of swing)
            zlims(1) = min(zlims(1), obj.Config.Sizes.GroundPlaneZ - 0.1);
            zlims(2) = max(zlims(2), 1.8); % Ensure reasonable height for golfer

            xlim(ax, xlims); ylim(ax, ylims); zlim(ax, zlims);
            axis(ax, 'manual'); % Lock limits after setting them

            % --- Lighting and Material Properties --- (Adjusted Lighting)
            set(ax, 'AmbientLightColor', [0.2 0.2 0.2]); % Darker ambient light << ADJUSTED

            if isempty(findobj(ax, 'Type', 'light')) % Add light if none exists
                % camlight(ax, 'headlight'); % Original
                camlight(ax, 'right'); % Position light to the right << ADJUSTED
            end
            lighting(ax, 'gouraud'); % Smooth lighting (phong is more detailed but slower)
            material(ax, 'dull');    % Non-shiny material appearance << KEY FOR LESS REFLECTION
            shading(ax, 'interp');   % Interpolated shading for smoothness
            % --- End Lighting Section ---
        end

        function [xlims, ylims, zlims] = calculatePlotLimits(obj, dataTable)
            % Calculate plot limits based on all marker data ranges + ball/ground
             all_x_data = [dataTable.Buttx; dataTable.CHx; dataTable.MPx; dataTable.LWx; dataTable.LEx; dataTable.LSx; dataTable.RWx; dataTable.REx; dataTable.RSx; dataTable.HUBx];
             all_y_data = [dataTable.Butty; dataTable.CHy; dataTable.MPy; dataTable.LWy; dataTable.LEy; dataTable.LSy; dataTable.RWy; dataTable.REy; dataTable.RSy; dataTable.HUBy];
             all_z_data = [dataTable.Buttz; dataTable.CHz; dataTable.MPz; dataTable.LWz; dataTable.LEz; dataTable.LSz; dataTable.RWz; dataTable.REz; dataTable.RSz; dataTable.HUBz];

             % Include golf ball and ground Z in limit calculations
             ballX = 0; ballY = 1.63; ballZ = obj.Config.Sizes.GroundPlaneZ + obj.Config.Sizes.BallDiameter/2; % Use updated Y
             all_x_data = [all_x_data; ballX];
             all_y_data = [all_y_data; ballY];
             all_z_data = [all_z_data; ballZ; obj.Config.Sizes.GroundPlaneZ]; % Ball Z and Ground Z

             minX = min(all_x_data(:), [], 'omitnan'); maxX = max(all_x_data(:), [], 'omitnan');
             minY = min(all_y_data(:), [], 'omitnan'); maxY = max(all_y_data(:), [], 'omitnan');
             minZ = min(all_z_data(:), [], 'omitnan'); maxZ = max(all_z_data(:), [], 'omitnan');

             % Handle cases where data might be all NaN or single points
             if isempty(minX) || isnan(minX); minX=0; end; if isempty(maxX) || isnan(maxX); maxX=0; end
             if isempty(minY) || isnan(minY); minY=0; end; if isempty(maxY) || isnan(maxY); maxY=0; end
             if isempty(minZ) || isnan(minZ); minZ=0; end; if isempty(maxZ) || isnan(maxZ); maxZ=0; end

             % Ensure a minimum range and add margin
             minRange = 0.2; % Minimum span for any axis
             xRange = max(maxX - minX, minRange);
             yRange = max(maxY - minY, minRange);
             zRange = max(maxZ - minZ, minRange);

             margin = obj.Config.Sizes.PlotMargin;
             xlims = [minX - margin*xRange, maxX + margin*xRange];
             ylims = [minY - margin*yRange, maxY + margin*yRange];
             zlims = [minZ - margin*zRange, maxZ + margin*zRange];
        end

        % --- Update Logic ---
        function updatePlot(obj, ~, ~)
            % Main function to update all plot elements for the CurrentFrame
            if ~isvalid(obj) || ~ishandle(obj.Handles.Figure); return; end % Check validity

            frameIdx = obj.CurrentFrame;
            baseTable = obj.DataSets{1}.Data; % Use BASE data for kinematics

            % --- Update Timestamp ---
            currentTime = (frameIdx - 1) / obj.Config.Recording.FrameRate;
            if ishandle(obj.Handles.TimestampText)
                set(obj.Handles.TimestampText, 'String', sprintf('Frame: %d / Time: %.2fs', frameIdx, currentTime));
            end

            % --- Get Point Coordinates for the current frame ---
            points = struct();
            try
                points.Butt = [baseTable.Buttx(frameIdx), baseTable.Butty(frameIdx), baseTable.Buttz(frameIdx)];
                points.Clubhead = [baseTable.CHx(frameIdx), baseTable.CHy(frameIdx), baseTable.CHz(frameIdx)];
                points.Midpoint = [baseTable.MPx(frameIdx), baseTable.MPy(frameIdx), baseTable.MPz(frameIdx)];
                points.LWrist = [baseTable.LWx(frameIdx), baseTable.LWy(frameIdx), baseTable.LWz(frameIdx)];
                points.LElbow = [baseTable.LEx(frameIdx), baseTable.LEy(frameIdx), baseTable.LEz(frameIdx)];
                points.LShoulder = [baseTable.LSx(frameIdx), baseTable.LSy(frameIdx), baseTable.LSz(frameIdx)];
                points.RWrist = [baseTable.RWx(frameIdx), baseTable.RWy(frameIdx), baseTable.RWz(frameIdx)];
                points.RElbow = [baseTable.REx(frameIdx), baseTable.REy(frameIdx), baseTable.REz(frameIdx)];
                points.RShoulder = [baseTable.RSx(frameIdx), baseTable.RSy(frameIdx), baseTable.RSz(frameIdx)];
                points.Hub = [baseTable.HUBx(frameIdx), baseTable.HUBy(frameIdx), baseTable.HUBz(frameIdx)];

                % Check for critical NaNs that prevent basic drawing / calculations
                criticalPoints = [points.Butt; points.Clubhead; points.Midpoint]; % Add more if needed by logic
                if any(isnan(criticalPoints(:)))
                    warning('GolfSwingVisualizer:NaNPoints', 'NaN detected in fundamental points for frame %d. Hiding dynamic elements.', frameIdx);
                    obj.hideDynamicPlotElements(); % Hide dynamic elements
                     if ishandle(obj.Handles.TimestampText)
                        set(obj.Handles.TimestampText, 'String', sprintf('Frame: %d / Time: %.2fs - NaN Data', frameIdx, currentTime));
                     end
                    return;
                end
            catch ME
                 warning('GolfSwingVisualizer:DataAccessError', 'Error accessing point data at frame %d: %s', frameIdx, ME.message);
                 obj.hideDynamicPlotElements(); % Hide dynamic elements on error
                 return;
            end

            % --- Calculate Vectors and Scales ---
            shaftVec = points.Clubhead - points.Butt;
            shaftLength = norm(shaftVec);
            if isnan(shaftLength) || shaftLength < 1e-6
                 shaftLength = 1e-6; % Avoid zero/NaN length
                 shaftDir = [0 0 1]; % Default direction
            else
                shaftDir = shaftVec / shaftLength;
            end

            vectorScale = get(obj.Handles.ScaleSlider, 'Value'); % User-defined scaling

            % --- Calculate Face Normal (Velocity-Based) ---
            velCH = [NaN NaN NaN]; % Initialize
            if frameIdx > 1
                prevCH = obj.getPreviousPoint('Clubhead', frameIdx);
                if ~any(isnan(prevCH)) && ~all(points.Clubhead == prevCH) % Check if position actually changed
                    velCH = points.Clubhead - prevCH;
                end
            end

            faceNormal = [1 0 0]; % Default fallback
            faceNormalValid = false;
            velDir = [NaN NaN NaN];

            if ~any(isnan(velCH)) && norm(velCH) > obj.Config.Sizes.VelocityEps
                 velDir = velCH / norm(velCH);
                 % Ensure shaftDir and velDir are valid and not parallel
                 if ~any(isnan(shaftDir)) && norm(cross(shaftDir, velDir)) > obj.Config.Sizes.ParallelEps
                     % Calculate normal using triple cross product: n = cross(cross(v, s), s);
                     faceNormal_temp = cross(cross(velDir, shaftDir), shaftDir);
                     norm_fn = norm(faceNormal_temp);
                     if norm_fn > 1e-6
                         faceNormal = faceNormal_temp / norm_fn;
                         faceNormalValid = true;
                     end
                 end
            end

            % If velocity method failed (frame 1, zero vel, parallel), use fallback
            if ~faceNormalValid
                 globalZ = [0 0 1];
                 if ~any(isnan(shaftDir)) && norm(cross(shaftDir, globalZ)) > obj.Config.Sizes.ParallelEps
                     faceNormal_temp = cross(shaftDir, globalZ); % Normal is horizontal projection
                     faceNormal = faceNormal_temp / norm(faceNormal_temp);
                 else % Shaft is parallel to Z or invalid
                     faceNormal = [1 0 0]; % Use global X as normal
                 end
            end

            % --- Update Plot Objects ---
            % Club (Shaft + Head)
            clubVisibleCheckbox = get(obj.Handles.Checkboxes(obj.Config.CheckboxMapping.Shaft_Club), 'Value');
            obj.updateCylinder(obj.PlotHandles.ShaftCylinder, points.Butt, points.Clubhead, obj.Config.Sizes.ShaftDiameter, clubVisibleCheckbox);
            obj.updateClubheadShape(points.Clubhead, shaftVec, clubVisibleCheckbox); % Use elongated clubhead function

            % Face Normal
            fnVisibleCheckbox = get(obj.Handles.Checkboxes(obj.Config.CheckboxMapping.Face_Normal), 'Value');
            fnScale = shaftLength * 0.2; % Scale face normal length relative to shaft length
            obj.updateQuiver(obj.PlotHandles.FaceNormalQuiver, points.Clubhead, faceNormal, fnScale, fnVisibleCheckbox);

            % Force/Torque Vectors
            for k = 1:length(obj.DataSets)
                dataSet = obj.DataSets{k};
                dataTable = dataSet.Data;
                try
                    forceVec = dataTable.TotalHandForceGlobal(frameIdx,:);
                    torqueVec = dataTable.EquivalentMidpointCoupleGlobal(frameIdx,:);
                    if any(isnan(forceVec)) || any(isnan(torqueVec))
                        forceVec = [NaN NaN NaN]; torqueVec = [NaN NaN NaN];
                    end
                catch ME
                    warning('GolfSwingVisualizer:DataAccessError', 'Error accessing vector data for %s at frame %d: %s', dataSet.Name, frameIdx, ME.message);
                    forceVec = [NaN NaN NaN]; torqueVec = [NaN NaN NaN]; % Treat as NaN on error
                end
                forceVisibleCheckbox = get(obj.Handles.Checkboxes(obj.Config.CheckboxMapping.(['Force_', dataSet.Name])), 'Value');
                forceScale = shaftLength / obj.MaxForceMag * vectorScale;
                obj.updateQuiver(obj.PlotHandles.ForceQuivers(k), points.Midpoint, forceVec, forceScale, forceVisibleCheckbox);
                torqueVisibleCheckbox = get(obj.Handles.Checkboxes(obj.Config.CheckboxMapping.(['Torque_', dataSet.Name])), 'Value');
                torqueScale = shaftLength / obj.MaxTorqueMag * vectorScale;
                obj.updateQuiver(obj.PlotHandles.TorqueQuivers(k), points.Midpoint, torqueVec, torqueScale, torqueVisibleCheckbox);
            end

            % Body Segments
            obj.updateBodySegment('Left_Forearm', points.LWrist, points.LElbow, obj.Config.Sizes.ForearmDiameter);
            obj.updateBodySegment('Left_Upper_Arm', points.LElbow, points.LShoulder, obj.Config.Sizes.UpperarmDiameter);
            obj.updateBodySegment('Left_Shoulder_Neck', points.LShoulder, points.Hub, obj.Config.Sizes.ShoulderNeckDiameter);
            obj.updateBodySegment('Right_Forearm', points.RWrist, points.RElbow, obj.Config.Sizes.ForearmDiameter);
            obj.updateBodySegment('Right_Upper_Arm', points.RElbow, points.RShoulder, obj.Config.Sizes.UpperarmDiameter);
            obj.updateBodySegment('Right_Shoulder_Neck', points.RShoulder, points.Hub, obj.Config.Sizes.ShoulderNeckDiameter);

             lhVisibleCheckbox = get(obj.Handles.Checkboxes(obj.Config.CheckboxMapping.Left_Shoulder_Neck), 'Value');
             rhVisibleCheckbox = get(obj.Handles.Checkboxes(obj.Config.CheckboxMapping.Right_Shoulder_Neck), 'Value');
             hubSphereVisible = (lhVisibleCheckbox || rhVisibleCheckbox);
             obj.updateSphere(obj.PlotHandles.Hub_Sphere, points.Hub, obj.Config.Sizes.ShoulderNeckDiameter, hubSphereVisible);

            % Recording Frame
            if obj.IsRecording && ~isempty(obj.VideoWriterObj) && isvalid(obj.VideoWriterObj)
                try
                    frame = getframe(obj.Handles.Figure);
                    writeVideo(obj.VideoWriterObj, frame);
                catch ME
                    warning('GolfSwingVisualizer:RecordingError', 'Failed to write video frame: %s', ME.message);
                    obj.stopRecording(); % Stop if error occurs
                end
            end

            drawnow limitrate; % Update display efficiently
        end

        function prevPoint = getPreviousPoint(obj, pointName, currentFrameIdx)
            % Helper to safely get coordinates from the previous frame
            prevPoint = [NaN NaN NaN];
            if currentFrameIdx > 1
                prevFrameIdx = currentFrameIdx - 1;
                baseTable = obj.DataSets{1}.Data; % Assume kinematics from BASE
                xCol = [pointName, 'x'];
                yCol = [pointName, 'y'];
                zCol = [pointName, 'z'];
                try
                    if ismember(xCol, baseTable.Properties.VariableNames) && ...
                       ismember(yCol, baseTable.Properties.VariableNames) && ...
                       ismember(zCol, baseTable.Properties.VariableNames)
                        prevPoint = [baseTable.(xCol)(prevFrameIdx), ...
                                     baseTable.(yCol)(prevFrameIdx), ...
                                     baseTable.(zCol)(prevFrameIdx)];
                    end
                catch ME
                    % warning('GolfSwingVisualizer:PrevPointAccessErr', 'Error accessing previous point %s: %s', pointName, ME.message); % Reduce noise
                    prevPoint = [NaN NaN NaN]; % Return NaN on error
                end
            end
        end

        function updateBodySegment(obj, segmentName, pt1, pt2, diameter)
            % Updates a single body segment (cylinder and start sphere)
            checkboxIndex = obj.Config.CheckboxMapping.(segmentName);
            isVisibleCheckbox = get(obj.Handles.Checkboxes(checkboxIndex), 'Value');

            cylHandle = obj.PlotHandles.([segmentName, '_Cylinder']);
            sphHandle = obj.PlotHandles.([segmentName, '_Sphere']);

            if ~ishandle(cylHandle) || ~ishandle(sphHandle); return; end

            obj.updateCylinder(cylHandle, pt1, pt2, diameter, isVisibleCheckbox);
            obj.updateSphere(sphHandle, pt1, diameter, isVisibleCheckbox);
        end

        function updateCylinder(obj, hSurf, pt1, pt2, diameter, isVisibleCheckbox)
            % Updates the position, orientation, and visibility of a cylinder surface
             if ~ishandle(hSurf); return; end

             if any(isnan(pt1)) || any(isinf(pt1)) || any(isnan(pt2)) || any(isinf(pt2))
                 set(hSurf, 'Visible', 'off', 'XData', nan(21,2), 'YData', nan(21,2), 'ZData', nan(21,2)); return;
             end
             if ~isVisibleCheckbox
                 set(hSurf, 'Visible', 'off'); return;
             end
             vec = pt2 - pt1;
             height = norm(vec);
             if height < 1e-6
                 set(hSurf, 'Visible', 'off', 'XData', nan(21,2), 'YData', nan(21,2), 'ZData', nan(21,2)); return;
             end

             [cyl_x, cyl_y, cyl_z] = cylinder(diameter/2, 20);
             cyl_pts = [cyl_x(:)'; cyl_y(:)'; cyl_z(:)'];

             z_axis = [0; 0; 1];
             dir = vec(:) / height;
             v_cross = cross(z_axis, dir);
             cos_angle = dot(z_axis, dir);

             if abs(cos_angle + 1.0) < 1e-6 % 180 degrees
                 if abs(dir(1)) > 1e-6 || abs(dir(2)) > 1e-6
                     axis_rot = cross(z_axis, dir);
                 else
                     axis_rot = [1; 0; 0];
                 end
                 angle_rad = pi;
                 axang = [axis_rot' angle_rad];
                  try
                      R = axang2rotm(axang);
                  catch
                       ax = axis_rot(1); ay = axis_rot(2); az = axis_rot(3);
                       R = [2*ax^2-1  2*ax*ay   2*ax*az;
                            2*ay*ax   2*ay^2-1  2*ay*az;
                            2*az*ax   2*az*ay   2*az^2-1];
                      if ~isdeployed
                          warning('GolfSwingVisualizer:MissingToolbox', 'Robotics/Navigation Toolbox potentially missing. Using manual 180deg rotation fallback for cylinder.');
                      end
                  end
             elseif abs(cos_angle - 1.0) < 1e-6 % 0 degrees
                 R = eye(3);
             else % General case
                 v_skew = [0 -v_cross(3) v_cross(2); v_cross(3) 0 -v_cross(1); -v_cross(2) v_cross(1) 0];
                 R = eye(3) + v_skew + v_skew^2 * (1/(1+cos_angle));
             end

             cyl_pts(3,:) = cyl_pts(3,:) * height;
             cyl_pts_rot = R * cyl_pts;
             Xc = reshape(cyl_pts_rot(1,:) + pt1(1), size(cyl_x));
             Yc = reshape(cyl_pts_rot(2,:) + pt1(2), size(cyl_y));
             Zc = reshape(cyl_pts_rot(3,:) + pt1(3), size(cyl_z));

             set(hSurf, 'XData', Xc, 'YData', Yc, 'ZData', Zc, 'Visible', 'on');
        end

        function updateSphere(obj, hSurf, center, diameter, isVisibleCheckbox)
            % Updates position and visibility of a sphere surface
             if ~ishandle(hSurf); return; end

             if any(isnan(center)) || any(isinf(center))
                 set(hSurf, 'Visible', 'off', 'XData', nan(21), 'YData', nan(21), 'ZData', nan(21)); return;
             end
             if ~isVisibleCheckbox
                 set(hSurf, 'Visible', 'off'); return;
             end

             [sx, sy, sz] = sphere(20);
             radius = diameter / 2;
             X = center(1) + sx * radius;
             Y = center(2) + sy * radius;
             Z = center(3) + sz * radius;

             set(hSurf, 'Visible', 'on', 'XData', X, 'YData', Y, 'ZData', Z);
        end

        function updateClubheadShape(obj, clubheadPos, shaftVec, isVisibleCheckbox)
             % Updates the elongated clubhead shape
             hSurf = obj.PlotHandles.ClubheadShape;
             if ~ishandle(hSurf); return; end

             if any(isnan(clubheadPos)) || any(isinf(clubheadPos)) || any(isnan(shaftVec)) || any(isinf(shaftVec))
                 set(hSurf, 'Visible', 'off', 'XData', nan(16,31), 'YData', nan(16,31), 'ZData', nan(16,31)); return;
             end
             shaftNorm = norm(shaftVec);
             if isnan(shaftNorm) || shaftNorm < 1e-6
                 set(hSurf, 'Visible', 'off', 'XData', nan(16,31), 'YData', nan(16,31), 'ZData', nan(16,31)); return;
             end
             if ~isVisibleCheckbox
                 set(hSurf, 'Visible', 'off'); return;
             end

             rad_along_shaft = obj.Config.Sizes.ClubheadLength / 2;
             rad_perp_shaft = obj.Config.Sizes.ClubheadWidth / 2;

             [theta, phi] = meshgrid(linspace(0, 2*pi, 30), linspace(0, pi/2, 15));
             hx = cos(theta).*sin(phi);
             hy = sin(theta).*sin(phi);
             hz = cos(phi);

             scaled_hx = hx * rad_perp_shaft;
             scaled_hy = hy * rad_perp_shaft;
             scaled_hz = hz * rad_along_shaft;

             pts = [scaled_hx(:)'; scaled_hy(:)'; scaled_hz(:)'];

             shaftDir = shaftVec / shaftNorm;
             localZ = -shaftDir(:);
             globalUp = [0; 0; 1];
             localX = cross(globalUp, localZ);
             if norm(localX) < 1e-6
                 localX = cross([0; 1; 0], localZ);
             end
             localX = localX / norm(localX);
             localY = cross(localZ, localX);

             R = [localX, localY, localZ];

             rotatedPts = R * pts;
             nx = reshape(rotatedPts(1,:), size(hx)) + clubheadPos(1);
             ny = reshape(rotatedPts(2,:), size(hy)) + clubheadPos(2);
             nz = reshape(rotatedPts(3,:), size(hz)) + clubheadPos(3);

             set(hSurf, 'XData', nx, 'YData', ny, 'ZData', nz, 'Visible', 'on');
        end

        function updateQuiver(obj, hQuiver, origin, vector, scaleFactor, isVisibleCheckbox)
            % Updates position, direction, scale, and visibility of a quiver object
            if ~ishandle(hQuiver); return; end

            vectorNorm = norm(vector);
            if any(isnan(origin)) || any(isinf(origin)) || any(isnan(vector)) || any(isinf(vector)) || ...
               isnan(scaleFactor) || isinf(scaleFactor) || vectorNorm < 1e-9 || abs(scaleFactor) < 1e-9
                set(hQuiver, 'Visible', 'off', 'XData', NaN, 'YData', NaN, 'ZData', NaN, 'UData', NaN, 'VData', NaN, 'WData', NaN); return;
            end
            if ~isVisibleCheckbox
                set(hQuiver, 'Visible', 'off'); return;
            end

            set(hQuiver, 'Visible', 'on', ...
                'XData', origin(1), 'YData', origin(2), 'ZData', origin(3), ...
                'UData', vector(1) * scaleFactor, 'VData', vector(2) * scaleFactor, 'WData', vector(3) * scaleFactor);
        end

        function hideDynamicPlotElements(obj)
            % Hides plot elements that change frame-to-frame
            fields = fieldnames(obj.PlotHandles);
            for i = 1:length(fields)
                fieldName = fields{i};
                if strcmp(fieldName, 'GroundPlane') || strcmp(fieldName, 'GolfBall')
                    continue; % Skip static elements
                end
                handleGroup = obj.PlotHandles.(fieldName);
                if isgraphics(handleGroup)
                    set(handleGroup(isgraphics(handleGroup)), 'Visible', 'off');
                end
            end
        end

        % --- Callbacks ---
        function frameSliderCallback(obj, src, ~)
            if ~isvalid(obj); return; end
            obj.CurrentFrame = round(src.Value);
            if ~obj.IsPlaying
                 obj.updatePlot();
            end
        end

        function scaleSliderCallback(obj, ~, ~)
             if ~isvalid(obj); return; end
             obj.updatePlot();
        end

        function speedSliderCallback(obj, newSpeedValue)
             if ~isvalid(obj); return; end
             if newSpeedValue <= 0; newSpeedValue = 0.01; end

             if obj.IsPlaying && ~isempty(obj.PlaybackTimer) && isvalid(obj.PlaybackTimer)
                 newPeriod = obj.Config.Playback.TimerPeriod / newSpeedValue;
                 if newPeriod > 0
                     try
                         obj.PlaybackTimer.Period = newPeriod;
                         if strcmp(obj.PlaybackTimer.Running, 'on')
                             stop(obj.PlaybackTimer);
                             start(obj.PlaybackTimer);
                         end
                     catch ME_timer
                         warning('GolfSwingVisualizer:TimerPeriodError', 'Could not set timer period: %s', ME_timer.message);
                     end
                 end
             end
         end

        function checkboxCallback(obj, ~, ~)
             if ~isvalid(obj); return; end
             obj.updatePlot();
        end

        function zoomSliderCallback(obj, src, ~)
            if ~isvalid(obj) || ~ishandle(obj.Handles.Axes); return; end
            ax = obj.Handles.Axes;
            xlim_orig = xlim(ax); ylim_orig = ylim(ax); zlim_orig = zlim(ax);
            [az, el] = view(ax);

            currentZoom = camzoom(ax);
            desiredZoomFactor = src.Value;
            if currentZoom == 0; currentZoom = 1; end
            relativeZoom = desiredZoomFactor / currentZoom;

            camzoom(ax, relativeZoom);

            xlim(ax, xlim_orig); ylim(ax, ylim_orig); zlim(ax, zlim_orig);
            view(ax, az, el);
        end

        function togglePlayPause(obj, src, ~)
             if ~isvalid(obj); return; end
            if src.Value == 1
                obj.startPlayback();
            else
                obj.stopPlayback();
            end
        end

        function startPlayback(obj)
             if ~isvalid(obj) || obj.IsPlaying; return; end
             obj.IsPlaying = true;
             if ishandle(obj.Handles.PlayPauseButton)
                 set(obj.Handles.PlayPauseButton, 'String', 'Pause', 'Value', 1);
             end

             timerSpeed = get(obj.Handles.SpeedSlider, 'Value');
             if timerSpeed <=0; timerSpeed = obj.Config.Playback.DefaultSpeed; end
             timerPeriod = obj.Config.Playback.TimerPeriod / timerSpeed;

             if isempty(obj.PlaybackTimer) || ~isvalid(obj.PlaybackTimer)
                 obj.PlaybackTimer = timer(...
                     'ExecutionMode', 'fixedRate', ...
                     'Period', max(0.001, timerPeriod), ...
                     'TimerFcn', @obj.playbackTimerCallback, ...
                     'StopFcn', @obj.playbackTimerStopCallback, ...
                     'ErrorFcn', @obj.playbackTimerErrorCallback);
             else
                try
                    if strcmp(obj.PlaybackTimer.Running, 'on'); stop(obj.PlaybackTimer); end
                    obj.PlaybackTimer.Period = max(0.001, timerPeriod);
                catch ME_timer
                     warning('GolfSwingVisualizer:TimerSetError', 'Could not set timer properties: %s', ME_timer.message);
                     obj.IsPlaying = false;
                     if ishandle(obj.Handles.PlayPauseButton); set(obj.Handles.PlayPauseButton, 'String', 'Play', 'Value', 0); end
                     return;
                end
             end

             try
                 if ~strcmp(obj.PlaybackTimer.Running, 'on')
                    start(obj.PlaybackTimer);
                 end
             catch ME_timer
                 warning('GolfSwingVisualizer:TimerStartError', 'Could not start timer: %s', ME_timer.message);
                 obj.IsPlaying = false;
                 if ishandle(obj.Handles.PlayPauseButton); set(obj.Handles.PlayPauseButton, 'String', 'Play', 'Value', 0); end
             end
        end

        function stopPlayback(obj)
            if ~isvalid(obj) || ~obj.IsPlaying; return; end
            obj.IsPlaying = false;
             if ishandle(obj.Handles.PlayPauseButton)
                 set(obj.Handles.PlayPauseButton, 'String', 'Play', 'Value', 0);
             end

            if ~isempty(obj.PlaybackTimer) && isvalid(obj.PlaybackTimer) && strcmp(obj.PlaybackTimer.Running, 'on')
                try
                    stop(obj.PlaybackTimer);
                catch ME_timer
                     warning('GolfSwingVisualizer:TimerStopError', 'Could not stop timer: %s', ME_timer.message);
                end
            end
        end

        function playbackTimerCallback(obj, ~, ~)
            if ~isvalid(obj) || ~obj.IsPlaying || ~ishandle(obj.Handles.Figure)
                obj.stopPlayback();
                return;
            end

            try
                if obj.CurrentFrame < obj.NumFrames
                    obj.CurrentFrame = obj.CurrentFrame + 1;
                else
                    obj.CurrentFrame = 1; % Loop
                end

                if ishandle(obj.Handles.FrameSlider)
                    set(obj.Handles.FrameSlider, 'Value', obj.CurrentFrame);
                end
                obj.updatePlot(); % Update graphics
            catch ME_plot
                 warning('GolfSwingVisualizer:PlotCallbackError', 'Error during plot update in timer: %s', ME_plot.message);
                 obj.stopPlayback(); % Stop playback if plotting fails
            end
        end

         function playbackTimerStopCallback(obj, ~, ~)
             if isvalid(obj) && obj.IsPlaying
                 obj.IsPlaying = false;
                 if ishandle(obj.Handles.PlayPauseButton)
                     set(obj.Handles.PlayPauseButton, 'String', 'Play', 'Value', 0);
                 end
             end
         end

         function playbackTimerErrorCallback(obj, ~, event)
             warning('GolfSwingVisualizer:TimerError', 'Error during playback timer execution: %s', event.Data.message);
             if isvalid(obj)
                 obj.stopPlayback();
             end
         end

        function toggleRecord(obj, src, ~)
            if ~isvalid(obj); return; end
            if src.Value == 1
                obj.startRecording();
            else
                obj.stopRecording();
            end
        end

        function startRecording(obj)
            if ~isvalid(obj) || obj.IsRecording; return; end

            defaultPath = fullfile(pwd, obj.Config.Recording.DefaultFileName);
            [file, path] = uiputfile(obj.Config.Recording.FileType, obj.Config.Recording.FileDescription, defaultPath);

            if isequal(file, 0) || isequal(path, 0)
                 if ishandle(obj.Handles.RecordButton); set(obj.Handles.RecordButton, 'Value', 0); end
                 return;
            end
            filename = fullfile(path, file);

            try
                obj.VideoWriterObj = VideoWriter(filename, 'MPEG-4');
                obj.VideoWriterObj.FrameRate = obj.Config.Recording.FrameRate;
                open(obj.VideoWriterObj);

                obj.IsRecording = true;
                if ishandle(obj.Handles.RecordButton)
                    set(obj.Handles.RecordButton, 'String', 'Stop Recording', 'BackgroundColor', obj.Config.Colors.RecordActive, 'Value', 1);
                end

                if ~obj.IsPlaying
                    obj.updatePlot();
                end

            catch ME
                errordlg(sprintf('Failed to open video file for writing:\n%s', ME.message), 'Recording Error');
                obj.stopRecording();
            end
        end

        function stopRecording(obj)
             if ~isvalid(obj); return; end
             wasRecording = obj.IsRecording;
             obj.IsRecording = false;

             if ~isempty(obj.VideoWriterObj)
                 try
                     close(obj.VideoWriterObj);
                     if wasRecording; fprintf('Recording saved to: %s\n', obj.VideoWriterObj.Filename); end
                 catch ME
                     warning('GolfSwingVisualizer:VideoCloseError', 'Error closing video file: %s', ME.message);
                 end
             end
             obj.VideoWriterObj = [];

             if ~isempty(obj.Handles) && isfield(obj.Handles, 'RecordButton') && ishandle(obj.Handles.RecordButton)
                 set(obj.Handles.RecordButton, 'String', 'Record', 'BackgroundColor', obj.Config.Colors.RecordIdle, 'Value', 0);
             end
        end

        function setViewCallback(obj, viewType)
            if ~isvalid(obj) || ~ishandle(obj.Handles.Axes); return; end
            ax = obj.Handles.Axes;

            switch lower(viewType)
                case 'faceon' % View from positive Y
                    view(ax, [0 0]); camup(ax, [0 0 1]);
                case 'downline' % View from positive X (looking down -X axis)
                    view(ax, [-90 0]); camup(ax, [0 0 1]); % Flipped view
                case 'topdown' % View from positive Z
                    view(ax, [0 90]); camup(ax, [0 1 0]); % Y is up when looking top-down
                case 'iso' % Standard isometric
                    view(ax, [-45 30]); camup(ax, [0 0 1]);
            end
             % Reset zoom on view change
             if ishandle(obj.Handles.ZoomSlider)
                 set(obj.Handles.ZoomSlider, 'Value', 1.0);
                 camzoom(ax, 1.0); % Apply zoom reset
             else
                 camzoom(ax,1.0); % Apply zoom reset even if slider deleted
             end
        end

        function toggleLegendVisibility(obj, src, ~)
            if ~isvalid(obj) || ~isfield(obj.Handles,'PanelLegend') || ~ishandle(obj.Handles.PanelLegend); return; end
            if src.Value == 1 % Button down -> Show Legend
                obj.Handles.PanelLegend.Visible = 'on';
                src.String = 'Hide Legend';
            else % Button up -> Hide Legend
                obj.Handles.PanelLegend.Visible = 'off';
                src.String = 'Show Legend';
            end
        end

        function closeFigureCallback(obj, ~, ~)
            % Callback for when the user tries to close the figure window
            figHandle = [];
            if isvalid(obj) && ~isempty(obj.Handles) && isfield(obj.Handles, 'Figure') && ishandle(obj.Handles.Figure)
                 figHandle = obj.Handles.Figure;
            end

            try
                if isvalid(obj)
                    obj.cleanupResources(); % Attempt cleanup first
                end
            catch ME_cleanup
                 warning('GolfSwingVisualizer:CleanupError', 'Error during resource cleanup on close: %s', ME_cleanup.message);
            end

            % Force deletion of the figure window if it still exists
            if ~isempty(figHandle) && ishandle(figHandle)
                delete(figHandle);
            end

            % Invalidate object handles to prevent errors if obj persists
            if isvalid(obj)
                obj.Handles = [];
                obj.PlotHandles = [];
                % delete(obj) % Optional: remove obj from workspace too
            end
        end

        function cleanupResources(obj)
            % Clean up timers, video writers, etc., before deletion/closure
            % fprintf('Cleaning up GolfSwingVisualizer resources...\n');

            obj.stopPlayback(); % Safely stops timer
            obj.stopRecording(); % Safely stops recording and closes file

             if ~isempty(obj.PlaybackTimer) && isvalid(obj.PlaybackTimer)
                 try; delete(obj.PlaybackTimer); catch; end
             end
             obj.PlaybackTimer = [];

            % No need to explicitly delete graphics handles if figure is closing
            obj.Handles = [];
            obj.PlotHandles = [];

            % fprintf('Resources cleaned.\n');
        end

    end % End Private Methods
end % End Class Definition
