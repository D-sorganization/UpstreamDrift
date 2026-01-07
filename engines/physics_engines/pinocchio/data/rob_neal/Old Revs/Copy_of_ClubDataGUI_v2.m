% ClubDataGUI_v2.m
% Enhanced GUI for viewing golf club data from multiple MAT files

function ClubDataGUI_v2()

    % --- Configuration ---
    matFiles = {
        'TW_wiffle.mat',
        'TW_ProV1.mat',
        'GW_wiffle.mat',
        'GW_ProV1.mat'
    };
    defaultFile = matFiles{1};

    % --- Load Initial Data ---
    [data, params] = loadData(defaultFile);

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

    % --- Data Plot Init ---
    handles.data = data;
    handles.params = params;
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


    % Extend legendItems
    handles.legendItems(end+1:end+4, :) = {
        handles.velQuiver, 'Club Velocity';
        handles.accQuiver, 'Club Acceleration';
        handles.velHand, 'Hand Velocity';
        handles.accHand, 'Hand Acceleration'
    };
    handles.velHand = quiver3(handles.ax, 0, 0, 0, 0, 0, 0, 'g', 'LineWidth', 1.5);
    handles.accHand = quiver3(handles.ax, 0, 0, 0, 0, 0, 0, 'y', 'LineWidth', 1.5);

    % Manual vector scaling
    uicontrol(panel, 'Style', 'text', 'String', 'Vector Scale:', 'Units', 'normalized', 'Position', [0.1 0.05 0.8 0.04]);
    handles.scaleEdit = uicontrol(panel, 'Style', 'edit', 'String', '0.1', 'Units', 'normalized', 'Position', [0.1 0.01 0.8 0.04], ...
        'Callback', @(src,~) validateScale(src));

    % Individual axis toggles
    axis_labels = {'Club X', 'Club Y', 'Club Z', 'Hand X', 'Hand Y', 'Hand Z'};
    spacing = 0.025;
    for idx = 1:6
        col = mod(idx-1, 2);
        row = floor((idx-1)/2);
        xpos = 0.1 + 0.45 * col;
        ypos = 0.6 - spacing * row;  % moved down significantly (~3 lines worth)
        handles.axisToggles(idx) = uicontrol(panel, 'Style', 'checkbox', 'String', axis_labels{idx}, 'Value', 1, ...
            'Units', 'normalized', 'Position', [xpos, ypos, 0.4, 0.035]);
    end

    % Keyframe jump dropdown
    keyLabels = {'Address', 'TopOfBackswing', 'Impact', 'Finish'};

    handles.keyMenu = uicontrol(panel, 'Style', 'popupmenu', 'String', keyLabels, 'Units', 'normalized', 'Position', [0.1 0.9 0.8 0.035], ...
        'Callback', @(src,~) jumpToKeyframe(src));

    % Frame info label
    handles.frameLabel = uicontrol(panel, 'Style', 'text', 'String', 'Frame: 1    Time: 0.000 s', 'Units', 'normalized', 'Position', [0.1 0.68 0.8 0.025]);

    % Frame navigation buttons
    uicontrol(panel, 'Style', 'pushbutton', 'String', '< Frame', 'Units', 'normalized', 'Position', [0.1 0.79 0.25 0.03], 'Callback', @(src,~) stepFrame(-1));
    uicontrol(panel, 'Style', 'pushbutton', 'String', 'Frame >', 'Units', 'normalized', 'Position', [0.65 0.79 0.25 0.03], 'Callback', @(src,~) stepFrame(1));

    % Text labels for vector magnitudes
    handles.velClubLabel = text(handles.ax, 0, 0, 0, '', 'Color', 'c', 'FontSize', 4);
    handles.accClubLabel = text(handles.ax, 0, 0, 0, '', 'Color', 'm', 'FontSize', 4);
    handles.velHandLabel = text(handles.ax, 0, 0, 0, '', 'Color', 'g', 'FontSize', 4);
    handles.accHandLabel = text(handles.ax, 0, 0, 0, '', 'Color', 'y', 'FontSize', 4);

    % Ghost shaft line from midhands in negative Z direction
    handles.ghostLine = plot3(handles.ax, NaN, NaN, NaN, 'k--', 'LineWidth', 1.5);

    % Ghost shaft toggle
    handles.showGhost = uicontrol(panel, 'Style', 'checkbox', 'String', 'Ghost Shaft', 'Value', 1, 'Units', 'normalized', 'Position', [0.1 0.325 0.8 0.02]);

    % Shaft tip deflection label
    handles.deflectionLabel = uicontrol(panel, 'Style', 'text', 'String', 'Tip Deflection: -- m', 'Units', 'normalized', 'Position', [0.1 0.3 0.4 0.02]);

guidata(fig, handles);

    % Apply initial view and plot first frame
    guidata(fig, handles);
    set(handles.viewMenu, 'Value', 2);
    view(handles.ax, [180 0]);
    animateFrame(handles);

    % Set axis limits to fit the motion
    allPts = [handles.data.midhands_xyz; handles.data.clubface_xyz];
    minVals = min(allPts, [], 1);
    maxVals = max(allPts, [], 1);
    rangePad = 0.1 * norm(maxVals - minVals);
    xlim(handles.ax, [minVals(1)-rangePad, maxVals(1)+rangePad]);
    ylim(handles.ax, [minVals(2)-rangePad, maxVals(2)+rangePad]);
    zlim(handles.ax, [minVals(3)-rangePad, maxVals(3)+rangePad]);
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
    data = S.data;
    params = S.params;

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

function animateFrame(handles)
    i = handles.frame;
    A = handles.data.midhands_xyz(i,:);
    B = handles.data.clubface_xyz(i,:);
    shaftLen = norm(B - A);
    scaleStr = get(handles.scaleEdit, 'String');
    scale = str2double(scaleStr);
    if isempty(scaleStr) || isnan(scale) || scale <= 0
        scale = 0.2 * shaftLen;
    end

    set(handles.line, 'XData', [A(1) B(1)], 'YData', [A(2) B(2)], 'ZData', [A(3) B(3)]);
    for j = 1:3
        clubVec = handles.data.clubface_dircos(i, (j-1)*3+1:(j-1)*3+3);
        handVec = handles.data.midhands_dircos(i, (j-1)*3+1:(j-1)*3+3);
        visible = ternary(get(handles.axisToggles(j), 'Value'), 'on', 'off');
        set(handles.quivers(1,j), 'XData', B(1), 'YData', B(2), 'ZData', B(3), 'UData', scale*clubVec(1), 'VData', scale*clubVec(2), 'WData', scale*clubVec(3), 'Visible', visible);
        visible = ternary(get(handles.axisToggles(j+3), 'Value'), 'on', 'off');
        set(handles.quivers(2,j), 'XData', A(1), 'YData', A(2), 'ZData', A(3), 'UData', scale*handVec(1), 'VData', scale*handVec(2), 'WData', scale*handVec(3), 'Visible', visible);
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
        set(handles.velQuiver, 'XData', B(1), 'YData', B(2), 'ZData', B(3), 'UData', v(1), 'VData', v(2), 'WData', v(3), 'Visible', 'on');
        set(handles.velClubLabel, 'Position', B + v * 1.1, 'String', sprintf('%.2f m/s', norm(v)));
    else
        set(handles.velQuiver, 'Visible', 'off');
        set(handles.velClubLabel, 'String', '', 'Position', [0 0 0]);
    end

    if get(handles.showAcceleration, 'Value') && i > 2
        a = handles.data.clubface_xyz_acc(i,:);
        set(handles.accQuiver, 'XData', B(1), 'YData', B(2), 'ZData', B(3), 'UData', a(1), 'VData', a(2), 'WData', a(3), 'Visible', 'on');
        set(handles.accClubLabel, 'Position', B + a * 1.1, 'String', sprintf('%.2f m/s²', norm(a)));
    else
        set(handles.accQuiver, 'Visible', 'off');
        set(handles.accClubLabel, 'String', '', 'Position', [0 0 0]);
    end

    % Hand velocity vector
    if get(handles.showHandVelocity, 'Value') && i > 1
        vH = handles.data.midhands_xyz_vel(i,:);
        set(handles.velHand, 'XData', A(1), 'YData', A(2), 'ZData', A(3), 'UData', vH(1), 'VData', vH(2), 'WData', vH(3), 'Visible', 'on');
        set(handles.velHandLabel, 'Position', A + vH * 1.1, 'String', sprintf('%.2f m/s', norm(vH)));
    else
        set(handles.velHand, 'Visible', 'off');
        set(handles.velHandLabel, 'String', '', 'Position', [0 0 0]);
    end

    % Hand acceleration vector
    if get(handles.showHandAcceleration, 'Value') && i > 2
        aH = handles.data.midhands_xyz_acc(i,:);
        set(handles.accHand, 'XData', A(1), 'YData', A(2), 'ZData', A(3), 'UData', aH(1), 'VData', aH(2), 'WData', aH(3), 'Visible', 'on');
        set(handles.accHandLabel, 'Position', A + aH * 1.1, 'String', sprintf('%.2f m/s²', norm(aH)));
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
    files = get(src, 'String');
    selected = files{get(src, 'Value')};
    [data, params] = loadData(selected);
    handles.data = data;
    handles.params = params;
    handles.frame = 1;
    set(handles.frameSlider, 'Min', 1, 'Max', length(data.time), 'Value', 1);
    guidata(src, handles);
    animateFrame(handles);
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

function validateScale(src)
    val = str2double(get(src, 'String'));
    if isnan(val) || val <= 0
        set(src, 'String', '0.1');
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
