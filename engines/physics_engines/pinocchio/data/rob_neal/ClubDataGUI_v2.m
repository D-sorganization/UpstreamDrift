% ClubDataGUI_v2.m
% Enhanced GUI for viewing golf club data from multiple MAT files

function ClubDataGUI_v2()

    % --- Configuration ---
    matFiles = {
        'TW_wiffle.mat',
        'TW_ProV1.mat',
        'GW_wiffle.mat',
        'GW_ProV1.mat',
        'Browse...'
    };
    defaultFile = matFiles{1};

    % --- Load Initial Data ---
    [data, params, initialLoadOk, initialErr] = safeLoad(defaultFile);
    if ~initialLoadOk
        warning('Initial data load failed: %s', initialErr);
        [data, params] = placeholderData();
    end

    % --- GUI Setup ---
    fig = figure('Name', 'Club Shaft Viewer', 'NumberTitle', 'off', 'Color', 'w', 'Position', [100 100 1200 700], ...
        'CloseRequestFcn', @onClose);
    handles.fig = fig;
    handles.ax = axes('Parent', fig, 'Position', [0.32 0.1 0.66 0.85]);
    xlabel('X'); ylabel('Y'); zlabel('Z'); grid on; daspect([1 1 1]); hold on;

    % Control panel
    panel = uipanel('Parent', fig, 'Title', 'Controls', 'FontSize', 10, 'Units', 'normalized', 'Position', [0.01 0.01 0.3 0.98]);

    % File selection dropdown

    handles.fileMenu = uicontrol(panel, 'Style', 'popupmenu', 'String', matFiles, 'Units', 'normalized', 'Position', [0.1 0.94 0.8 0.035], ...
        'Callback', @(src,~) changeFile(src, handles));

    % Playback speed
    uicontrol(panel, 'Style', 'text', 'String', 'Playback Speed:', 'Units', 'normalized', 'Position', [0.1 0.68 0.8 0.05]);
    handles.speedSlider = uicontrol(panel, 'Style', 'slider', 'Min', 30, 'Max', 480, 'Value', 240, 'SliderStep', [1/450 0.1], ...
        'Units', 'normalized', 'Position', [0.1 0.74 0.8 0.025]);


    handles.viewMenu = uicontrol(panel, 'Style', 'popupmenu', 'String', {'Isometric','Face-On','Down-the-Line','Top-Down'}, 'Units', 'normalized', 'Position', [0.1 0.86 0.8 0.035], 'Callback', @(src,~) updateView(src, handles));

    % Play/Pause
    handles.playBtn = uicontrol(panel, 'Style', 'togglebutton', 'String', 'Play', 'Units', 'normalized', 'Position', [0.35 0.79 0.3 0.03], 'BackgroundColor', [0.2 0.8 0.2], 'Callback', @(src,~) toggleTimer(src, guidata(src)));

    % Frame slider
    handles.frameSlider = uicontrol(fig, 'Style', 'slider', 'Min', 1, 'Max', length(data.time), 'Value', 1, 'Units', 'normalized', 'Position', [0.35 0.01 0.6 0.02], 'Callback', @(src,~) updateFrameSlider(src, guidata(src)));
    if ~initialLoadOk
        set(handles.frameSlider, 'Enable', 'off');
        set(handles.playBtn, 'Enable', 'off');
    end

    % --- Data Plot Init ---
    handles.data = data;
    handles.params = params;
    handles.currentFile = defaultFile;
    handles.frame = 1;
    handles.timer = timer('ExecutionMode', 'fixedSpacing', 'BusyMode', 'drop', ...
        'TimerFcn', @(~,~) playbackCallback(fig), 'Period', 0.01);

    handles.quivers = gobjects(2, 3);
    colors = {'r','g','b'};
    for j = 1:3
        handles.quivers(1,j) = quiver3(handles.ax, 0, 0, 0, 0, 0, 0, colors{j}, 'LineWidth', 2);
        handles.quivers(2,j) = quiver3(handles.ax, 0, 0, 0, 0, 0, 0, colors{j}, 'LineWidth', 2, 'LineStyle', '--');
    end


    % --- Continue GUI Setup ---
    handles.line = plot3(handles.ax, NaN, NaN, NaN, 'k-', 'LineWidth', 2);
    % Trace toggles and impact marker toggle
    handles.showTraceHands = uicontrol(panel, 'Style', 'checkbox', 'String', 'Hands Trace', 'Value', 0, 'Units', 'normalized', 'Position', [0.1 0.525 0.8 0.02]);
    handles.showTraceClub = uicontrol(panel, 'Style', 'checkbox', 'String', 'Club Trace', 'Value', 0, 'Units', 'normalized', 'Position', [0.1 0.50 0.8 0.02]);
    handles.showImpact = uicontrol(panel, 'Style', 'checkbox', 'String', 'Impact Marker', 'Value', 0, 'Units', 'normalized', 'Position', [0.1 0.475 0.8 0.02]);

    % Trace lines
    handles.traceClub = plot3(handles.ax, NaN, NaN, NaN, 'b:', 'LineWidth', 1);
    handles.traceHands = plot3(handles.ax, NaN, NaN, NaN, 'k:', 'LineWidth', 1);

    % Impact marker
    impIdx = params.impact_frame;
    impA = data.midhands_xyz(impIdx,:);
    impB = data.clubface_xyz(impIdx,:);
    handles.impactLine = plot3(handles.ax, [impA(1), impB(1)], [impA(2), impB(2)], [impA(3), impB(3)], 'm--', 'LineWidth', 2, 'Visible', 'off');

    % Velocity and acceleration quivers
    handles.velQuiver = quiver3(handles.ax, 0, 0, 0, 0, 0, 0, 'c', 'LineWidth', 1.5);
    handles.accQuiver = quiver3(handles.ax, 0, 0, 0, 0, 0, 0, 'm', 'LineWidth', 1.5);

    handles.showVelocity = uicontrol(panel, 'Style', 'checkbox', 'String', 'Velocity', 'Value', 0, 'Units', 'normalized', 'Position', [0.1 0.425 0.8 0.02]);
    handles.showAcceleration = uicontrol(panel, 'Style', 'checkbox', 'String', 'Acceleration', 'Value', 0, 'Units', 'normalized', 'Position', [0.1 0.4 0.8 0.02]);

    % Add toggles for midhands velocity and acceleration
    handles.showHandVelocity = uicontrol(panel, 'Style', 'checkbox', 'String', 'Hand Velocity', 'Value', 0, 'Units', 'normalized', 'Position', [0.1 0.375 0.8 0.02]);
    handles.showHandAcceleration = uicontrol(panel, 'Style', 'checkbox', 'String', 'Hand Acceleration', 'Value', 0, 'Units', 'normalized', 'Position', [0.1 0.35 0.8 0.02]);

    % Add legend toggle
    handles.showLegend = uicontrol(panel, 'Style', 'checkbox', 'String', 'Show Legend', 'Value', 0, 'Units', 'normalized', 'Position', [0.1 0.3 0.8 0.02], 'Callback', @(src,dummy) toggleLegend(src, dummy));

    % Legend objects
    handles.legendItems = {
        handles.quivers(1,1), 'Club X';
        handles.quivers(1,2), 'Club Y';
        handles.quivers(1,3), 'Club Z';
        handles.quivers(2,1), 'Hand X';
        handles.quivers(2,2), 'Hand Y';
        handles.quivers(2,3), 'Hand Z';
        handles.velQuiver, 'Club Velocity';
        handles.accQuiver, 'Club Acceleration'
    };
    handles.legendHandle = legend(handles.ax, 'off');

    % Midhand velocity/acceleration quivers
    handles.velHand = quiver3(handles.ax, 0, 0, 0, 0, 0, 0, 'g', 'LineWidth', 1.5);
    handles.accHand = quiver3(handles.ax, 0, 0, 0, 0, 0, 0, 'y', 'LineWidth', 1.5);

    % Extend legendItems
    handles.legendItems(end+1:end+2, :) = {
        handles.velHand, 'Hand Velocity';
        handles.accHand, 'Hand Acceleration'
    };

    % Manual vector scaling controls (placed at the bottom)
    labelWidth = 0.35;
    inputWidth = 0.35;
    inputHeight = 0.03;
    labelHeight = 0.027;
    startX = 0.1;
    inputX = startX + labelWidth + 0.02;
    spacing = 0.04;
    startBottom = 0.05;  % Very bottom position

    uicontrol(panel, 'Style', 'text', 'String', 'Unit Vector Scale:', 'Units', 'normalized', 'Position', [startX startBottom+2*spacing labelWidth labelHeight], 'HorizontalAlignment', 'left');
    handles.unitScaleEdit = uicontrol(panel, 'Style', 'edit', 'String', '0.1', 'Units', 'normalized', 'Position', [inputX startBottom+2*spacing inputWidth inputHeight], ...
        'Callback', @(src,~) validateUnitScale(src));

    uicontrol(panel, 'Style', 'text', 'String', 'Velocity Scale:', 'Units', 'normalized', 'Position', [startX startBottom+spacing labelWidth labelHeight], 'HorizontalAlignment', 'left');
    handles.velScaleEdit = uicontrol(panel, 'Style', 'edit', 'String', '1', 'Units', 'normalized', 'Position', [inputX startBottom+spacing inputWidth inputHeight], ...
        'Callback', @(src,~) validateVelScale(src));

    uicontrol(panel, 'Style', 'text', 'String', 'Acceleration Scale:', 'Units', 'normalized', 'Position', [startX startBottom labelWidth labelHeight], 'HorizontalAlignment', 'left');
    handles.accScaleEdit = uicontrol(panel, 'Style', 'edit', 'String', '1', 'Units', 'normalized', 'Position', [inputX startBottom inputWidth inputHeight], ...
        'Callback', @(src,~) validateAccScale(src));

    % Individual axis toggles (original positions)
    axis_labels = {'Club X', 'Club Y', 'Club Z', 'Hand X', 'Hand Y', 'Hand Z'};
    spacingAxisToggles = 0.025;
    for idx = 1:6
        col = mod(idx-1, 2);
        row = floor((idx-1)/2);
        xpos = 0.1 + 0.45 * col;
        ypos = 0.6 - spacingAxisToggles * row;
        handles.axisToggles(idx) = uicontrol(panel, 'Style', 'checkbox', 'String', axis_labels{idx}, 'Value', 1, ...
            'Units', 'normalized', 'Position', [xpos, ypos, 0.4, 0.035]);
    end

    % Keyframe jump dropdown (original position)
    keyLabels = {'Address', 'TopOfBackswing', 'Impact', 'Finish'};

    handles.keyMenu = uicontrol(panel, 'Style', 'popupmenu', 'String', keyLabels, 'Units', 'normalized', 'Position', [0.1 0.9 0.8 0.035], ...
        'Callback', @(src,~) jumpToKeyframe(src));

    % Frame info label (original position)
    handles.frameLabel = uicontrol(panel, 'Style', 'text', 'String', 'Frame: 1    Time: 0.000 s', 'Units', 'normalized', 'Position', [0.1 0.68 0.8 0.025]);

    % Frame navigation buttons (original positions)
    uicontrol(panel, 'Style', 'pushbutton', 'String', '< Frame', 'Units', 'normalized', 'Position', [0.1 0.79 0.25 0.03], 'Callback', @(src,~) stepFrame(-1));
    uicontrol(panel, 'Style', 'pushbutton', 'String', 'Frame >', 'Units', 'normalized', 'Position', [0.65 0.79 0.25 0.03], 'Callback', @(src,~) stepFrame(1));

    % Text labels for vector magnitudes
    handles.velClubLabel = text(handles.ax, 0, 0, 0, '', 'Color', 'c', 'FontSize', 9);
    handles.accClubLabel = text(handles.ax, 0, 0, 0, '', 'Color', 'm', 'FontSize', 9);
    handles.velHandLabel = text(handles.ax, 0, 0, 0, '', 'Color', 'g', 'FontSize', 9);
    handles.accHandLabel = text(handles.ax, 0, 0, 0, '', 'Color', 'y', 'FontSize', 9);

    % Ghost shaft line from midhands in negative Z direction
    handles.ghostLine = plot3(handles.ax, NaN, NaN, NaN, 'k--', 'LineWidth', 1.5);

    % Ghost shaft toggle
    handles.showGhost = uicontrol(panel, 'Style', 'checkbox', 'String', 'Ghost Shaft', 'Value', 1, 'Units', 'normalized', 'Position', [0.1 0.325 0.8 0.02]);

    % Shaft tip deflection label
    handles.deflectionLabel = uicontrol(panel, 'Style', 'text', 'String', 'Tip Deflection: -- m', 'Units', 'normalized', 'Position', [0.1 0.3 0.4 0.02]);

    guidata(fig, handles);

    % Apply initial view and plot first frame when data is valid
    guidata(fig, handles);
    set(handles.viewMenu, 'Value', 2);
    view(handles.ax, [180 0]);
    if initialLoadOk
        animateFrame(handles);
        updateAxisLimits(handles);
    else
        warndlg('Default file could not be loaded. Please use Browse... to select a data file.', 'Load Warning');
    end
end

function toggleTimer(src, handles)
    if get(src, 'Value')
        handles.timer.Period = 1 / get(handles.speedSlider, 'Value');
        set(src, 'String', 'Pause', 'BackgroundColor', [1 0.4 0.4]);
        start(handles.timer);
    else
        stop(handles.timer);
        set(src, 'String', 'Play', 'BackgroundColor', [0.2 0.8 0.2]);
    end
    guidata(src, handles);
end

function playbackCallback(fig)
    if ~ishandle(fig), return; end
    handles = guidata(fig);
    handles.frame = handles.frame + 1;
    if handles.frame > length(handles.data.time)
        handles.frame = 1;
    end
    set(handles.frameSlider, 'Value', handles.frame);
    guidata(fig, handles);
    animateFrame(handles);
end

function onClose(src, ~)
    handles = guidata(src);
    try, stop(handles.timer); delete(handles.timer); end
    delete(src);
end

function [data, params] = loadData(filename)
    S = load(filename);

    if ~isfield(S, 'data') || ~isfield(S, 'params')
        error('Loaded file %s does not contain required ''data'' and ''params'' structures.', filename);
    end

    data = S.data;
    params = S.params;

    required_fields = {"time", "midhands_xyz", "clubface_xyz", "midhands_dircos", "clubface_dircos"};
    missing = required_fields(~isfield(data, required_fields));
    if ~isempty(missing)
        error('Data structure missing required fields: %s', strjoin(missing, ', '));
    end

    % Quintic spline smoothing of clubface and midhands
    t = data.time(:);
    for name = ["clubface_xyz", "midhands_xyz"]
        P = data.(name);
        pp = spline(t, P');  % natural spline fit
        vel = ppval(fnder(pp, 1), t)';
        acc = ppval(fnder(pp, 2), t)';
        data.([char(name) '_vel']) = vel;
        data.([char(name) '_acc']) = acc;
    end
end

function [data, params, ok, errMsg] = safeLoad(filename)
    ok = true;
    errMsg = '';
    try
        [data, params] = loadData(filename);
    catch ME
        ok = false;
        errMsg = ME.message;
        data = [];
        params = [];
    end
end

function [data, params] = placeholderData()
    data.time = 0;
    data.midhands_xyz = zeros(1, 3);
    data.clubface_xyz = zeros(1, 3);
    data.midhands_dircos = zeros(1, 9);
    data.clubface_dircos = zeros(1, 9);
    data.midhands_xyz_vel = zeros(1, 3);
    data.clubface_xyz_vel = zeros(1, 3);
    data.midhands_xyz_acc = zeros(1, 3);
    data.clubface_xyz_acc = zeros(1, 3);
    params.Address = 1;
    params.impact_frame = 1;
end

function animateFrame(handles)
    i = handles.frame;
    A = handles.data.midhands_xyz(i,:);
    B = handles.data.clubface_xyz(i,:);
    shaftLen = norm(B - A);

    unitScale = str2double(get(handles.unitScaleEdit, 'String'));
    if isnan(unitScale) || unitScale <= 0
        unitScale = 0.1;
        set(handles.unitScaleEdit, 'String', '0.1');
    end

    velScale = str2double(get(handles.velScaleEdit, 'String'));
    if isnan(velScale) || velScale <= 0
        velScale = 1;
        set(handles.velScaleEdit, 'String', '1');
    end

    accScale = str2double(get(handles.accScaleEdit, 'String'));
    if isnan(accScale) || accScale <= 0
        accScale = 1;
        set(handles.accScaleEdit, 'String', '1');
    end

    set(handles.line, 'XData', [A(1) B(1)], 'YData', [A(2) B(2)], 'ZData', [A(3) B(3)]);
    for j = 1:3
        clubVec = handles.data.clubface_dircos(i, (j-1)*3+1:(j-1)*3+3);
        handVec = handles.data.midhands_dircos(i, (j-1)*3+1:(j-1)*3+3);
        visible = ternary(get(handles.axisToggles(j), 'Value'), 'on', 'off');
        set(handles.quivers(1,j), 'XData', B(1), 'YData', B(2), 'ZData', B(3), 'UData', unitScale*clubVec(1), 'VData', unitScale*clubVec(2), 'WData', unitScale*clubVec(3), 'Visible', visible);
        visible = ternary(get(handles.axisToggles(j+3), 'Value'), 'on', 'off');
        set(handles.quivers(2,j), 'XData', A(1), 'YData', A(2), 'ZData', A(3), 'UData', unitScale*handVec(1), 'VData', unitScale*handVec(2), 'WData', unitScale*handVec(3), 'Visible', visible);
    end
    % Trace update
    if get(handles.showTraceHands, 'Value')
        set(handles.traceHands, 'XData', handles.data.midhands_xyz(1:i,1), 'YData', handles.data.midhands_xyz(1:i,2), 'ZData', handles.data.midhands_xyz(1:i,3));
    else
        set(handles.traceHands, 'XData', NaN, 'YData', NaN, 'ZData', NaN);
    end

    if get(handles.showTraceClub, 'Value')
        set(handles.traceClub, 'XData', handles.data.clubface_xyz(1:i,1), 'YData', handles.data.clubface_xyz(1:i,2), 'ZData', handles.data.clubface_xyz(1:i,3));
    else
        set(handles.traceClub, 'XData', NaN, 'YData', NaN, 'ZData', NaN);
    end

    % Impact marker visibility
    set(handles.impactLine, 'Visible', ternary(get(handles.showImpact, 'Value'), 'on', 'off'));

    % Velocity & acceleration vectors
    dt = mean(diff(handles.data.time));
    if get(handles.showVelocity, 'Value') && i > 1
        v = handles.data.clubface_xyz_vel(i,:);
        set(handles.velQuiver, 'XData', B(1), 'YData', B(2), 'ZData', B(3), 'UData', velScale*v(1), 'VData', velScale*v(2), 'WData', velScale*v(3), 'Visible', 'on');  % Apply scale
        set(handles.velClubLabel, 'Position', B + v * 0.1, 'String', sprintf('%.2f m/s', norm(v))); % No scale here
    else
        set(handles.velQuiver, 'Visible', 'off');
        set(handles.velClubLabel, 'String', '', 'Position', [0 0 0]);
    end

    if get(handles.showAcceleration, 'Value') && i > 2
        a = handles.data.clubface_xyz_acc(i,:);
        set(handles.accQuiver, 'XData', B(1), 'YData', B(2), 'ZData', B(3), 'UData', accScale*a(1), 'VData', accScale*a(2), 'WData', accScale*a(3), 'Visible', 'on');  % Apply scale
        set(handles.accClubLabel, 'Position', B + a * 0.1, 'String', sprintf('%.2f m/s²', norm(a))); % No scale here
    else
        set(handles.accQuiver, 'Visible', 'off');
        set(handles.accClubLabel, 'String', '', 'Position', [0 0 0]);
    end

    % Hand velocity vector
    if get(handles.showHandVelocity, 'Value') && i > 1
        vH = handles.data.midhands_xyz_vel(i,:);
        set(handles.velHand, 'XData', A(1), 'YData', A(2), 'ZData', A(3), 'UData', velScale*vH(1), 'VData', velScale*vH(2), 'WData', velScale*vH(3), 'Visible', 'on');  % Apply scale
        set(handles.velHandLabel, 'Position', A + vH * 0.1, 'String', sprintf('%.2f m/s', norm(vH))); % No scale here
    else
        set(handles.velHand, 'Visible', 'off');
        set(handles.velHandLabel, 'String', '', 'Position', [0 0 0]);
    end

    % Hand acceleration vector
    if get(handles.showHandAcceleration, 'Value') && i > 2
        aH = handles.data.midhands_xyz_acc(i,:);
        set(handles.accHand, 'XData', A(1), 'YData', A(2), 'ZData', A(3), 'UData', accScale*aH(1), 'VData', accScale*aH(2), 'WData', accScale*aH(3), 'Visible', 'on');  % Apply scale
        set(handles.accHandLabel, 'Position', A + aH * 0.1, 'String', sprintf('%.2f m/s²', norm(aH))); % No scale here
    else
        set(handles.accHand, 'Visible', 'off');
        set(handles.accHandLabel, 'String', '', 'Position', [0 0 0]);
    end

    guidata(handles.fig, handles);
    set(handles.frameLabel, 'String', sprintf('Frame: %d    Time: %.3f s', i, handles.data.time(i)));
    % Ghost shaft visualization (using Z-axis of midhands at address)
    if get(handles.showGhost, 'Value') && isfield(handles, 'ghostLine') && isfield(handles.params, 'Address')
        Zh = handles.data.midhands_dircos(i, 7:9);
        if ~isfield(handles, 'refZ') || ~isfield(handles, 'refLen')
            handles.refZ = handles.data.midhands_dircos(handles.params.Address, 7:9);
            addrA = handles.data.midhands_xyz(handles.params.Address, :);
            addrB = handles.data.clubface_xyz(handles.params.Address, :);
            handles.refLen = norm(addrB - addrA);
        end
        if isfield(handles, 'refZ') && isfield(handles, 'refLen')
            ghostEnd = A - Zh * handles.refLen;
            set(handles.ghostLine, 'XData', [A(1), ghostEnd(1)], 'YData', [A(2), ghostEnd(2)], 'ZData', [A(3), ghostEnd(3)], 'Visible', 'on');
            deflection = norm(B - ghostEnd);
            set(handles.deflectionLabel, 'String', sprintf('Tip Deflection: %.3f m', deflection));
        end
    else
        set(handles.ghostLine, 'Visible', 'off');
        set(handles.deflectionLabel, 'String', 'Tip Deflection: -- m');
    end

    drawnow;


function updateAxisLimits(handles)
    if ~isfield(handles, 'data') || ~isstruct(handles.data)
        return;
    end

    if ~isfield(handles.data, 'midhands_xyz') || isempty(handles.data.midhands_xyz) || ...
            ~isfield(handles.data, 'clubface_xyz') || isempty(handles.data.clubface_xyz)
        return;
    end

    allPts = [handles.data.midhands_xyz; handles.data.clubface_xyz];
    minVals = min(allPts, [], 1);
    maxVals = max(allPts, [], 1);
    rangePad = 0.1 * norm(maxVals - minVals);
    xlim(handles.ax, [minVals(1)-rangePad, maxVals(1)+rangePad]);
    ylim(handles.ax, [minVals(2)-rangePad, maxVals(2)+rangePad]);
    zlim(handles.ax, [minVals(3)-rangePad, maxVals(3)+rangePad]);
end


function updateFrameSlider(src, handles)
    handles.frame = round(get(src, 'Value'));
    guidata(src, handles);
    animateFrame(handles);
end

function stepFrame(step)
    fig = gcbf;
    handles = guidata(fig);
    newFrame = max(1, min(handles.frame + step, length(handles.data.time)));
    handles.frame = newFrame;
    set(handles.frameSlider, 'Value', newFrame);
    guidata(fig, handles);
    animateFrame(handles);
end

function updateView(src, handles)
    val = get(src, 'Value');
    switch val
        case 1, view(handles.ax, [45 20]);  % Isometric (from +Y side, slight elevation)
        case 2, view(handles.ax, [180 0]);  % Face-On (looking downrange)
        case 3, view(handles.ax, [270 0]); % Down-the-Line (from golfer's right)
        case 4, view(handles.ax, [0 90]); % Top-Down
    end
end
function changeFile(src, handles)
    handles = guidata(src);
    files = get(src, 'String');
    idx = get(src, 'Value');
    selected = files{idx};

    if strcmp(selected, 'Browse...')
        [file, path] = uigetfile({'*.mat', 'MAT-files (*.mat)'}, 'Select Motion Capture Data');
        if isequal(file, 0)
            set(src, 'Value', safeFileIndex(files, handles.currentFile));
            return;
        end
        selected = fullfile(path, file);
        if ~any(strcmp(files, selected))
            files = [files(1:end-1); {selected}; {'Browse...'}];
            set(src, 'String', files);
        end
        idx = find(strcmp(files, selected), 1);
        set(src, 'Value', idx);
    end

    try
        [data, params] = loadData(selected);
    catch ME
        errordlg(sprintf('Failed to load %s:\n%s', selected, ME.message), 'Load Error');
        set(src, 'Value', safeFileIndex(files, handles.currentFile));
        return;
    end

    handles.data = data;
    handles.params = params;
    handles.currentFile = selected;
    handles.frame = 1;
    set(handles.frameSlider, 'Min', 1, 'Max', length(data.time), 'Value', 1, 'Enable', 'on');
    if isfield(handles, 'playBtn') && isgraphics(handles.playBtn)
        set(handles.playBtn, 'Enable', 'on', 'Value', 0, 'String', 'Play', 'BackgroundColor', [0.2 0.8 0.2]);
        stop(handles.timer);
    end
    guidata(src, handles);
    animateFrame(handles);
    updateAxisLimits(handles);
end

function val = ternary(cond, t, f)
    if cond, val = t; else, val = f; end
end

function toggleLegend(src, ~)
    handles = guidata(src);
    if get(src, 'Value')
        items = {};
        labels = {};
        for i = 1:size(handles.legendItems, 1)
            if isgraphics(handles.legendItems{i,1})
                items{end+1} = handles.legendItems{i,1};
                labels{end+1} = handles.legendItems{i,2};
            end
        end
        handles.legendHandle = legend(handles.ax, items, labels);
    else
        legend(handles.ax, 'off');
    end
    guidata(src, handles);
end

function validateUnitScale(src)
    val = str2double(get(src, 'String'));
    if isnan(val) || val <= 0
        val = 0.1;
    end
    val = min(max(val, 0.001), 1000);
    set(src, 'String', num2str(val));
    handles = guidata(src);
    animateFrame(handles);
end

function validateVelScale(src)
    val = str2double(get(src, 'String'));
    if isnan(val) || val <= 0
        val = 1;
    end
    val = min(max(val, 0.001), 1000);
    set(src, 'String', num2str(val));
    handles = guidata(src);
    animateFrame(handles);
end

function validateAccScale(src)
    val = str2double(get(src, 'String'));
    if isnan(val) || val <= 0
        val = 1;
    end
    val = min(max(val, 0.001), 1000);
    set(src, 'String', num2str(val));
    handles = guidata(src);
    animateFrame(handles);
end

function idx = safeFileIndex(files, currentFile)
    idx = find(strcmp(files, currentFile), 1);
    if isempty(idx)
        idx = 1;
    end
end

function jumpToKeyframe(src)
    handles = guidata(src);
    labels = get(src, 'String');
    idx = get(src, 'Value');
    field = labels{idx};
    if isfield(handles.params, field)
        handles.frame = handles.params.(field);
        set(handles.frameSlider, 'Value', handles.frame);
        guidata(src, handles);
        animateFrame(handles);
    end
end
end
