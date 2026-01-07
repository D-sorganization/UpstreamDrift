function ClubDataGUIRev0_fixed()
% ClubDataGUI: Load and animate golf club data with proper column decoding

% Auto-load data file
defaultFile = 'Wiffle_ProV1_club_3D_data.xlsx';
if ~isfile(defaultFile)
    errordlg('Data file not found.');
    return;
end

[~, sheets] = xlsfinfo(defaultFile);
fig = figure('Name', 'Clubface Animation GUI', 'NumberTitle', 'off', ...
    'Color', [1 1 1], 'Position', [100 100 1200 700]);

handles.ax = axes('Parent', fig, 'Position', [0.3 0.25 0.65 0.7]);
axis equal; grid on; view(handles.ax, [180 0]);
xlabel('X'); ylabel('Y'); zlabel('Z'); hold on;

handles.panel = uipanel('Parent', fig, 'Title', 'Controls', 'FontSize', 10, ...
    'Units', 'normalized', 'Position', [0.01 0.25 0.25 0.7]);

handles.filepath = defaultFile;

uicontrol('Parent', handles.panel, 'Style', 'text', 'String', 'Worksheet', ...
    'Units', 'normalized', 'Position', [0.1 0.92 0.8 0.05]);
handles.sheetMenu = uicontrol('Parent', handles.panel, ...
    'Style', 'popupmenu', ...
    'String', sheets, ...
    'Units', 'normalized', ...
    'Position', [0.1 0.87 0.8 0.05], ...
    'Callback', @(src,~) plotSheet(src));
        'Units', 'normalized', 'Position', [0.1 0.80 - 0.05*v, 0.8, 0.04], ...
        'Callback', @(~,~) view(handles.ax, viewAngles{v}));
end
    'String', sheets, ...
    'Units', 'normalized', 'Position', [0.1 0.87 0.8 0.05], ...
    'Callback', @(src,~) plotSheet(src);

% Axis toggles
labels = {'Club X','Club Y','Club Z','Hand X','Hand Y','Hand Z'};
for i = 1:6
    handles.axisToggles(i) = uicontrol('Parent', handles.panel, 'Style', 'checkbox', ...
        'String', labels{i}, 'Value', 1, ...
        'Units', 'normalized', 'Position', [0.1 0.45 - 0.04*i 0.8 0.04]);
end

% Angle label and playback toggle
handles.angleLabel = uicontrol('Parent', handles.panel, 'Style', 'text', ...
    'String', 'X: --  Y: --  Z: --', 'Units', 'normalized', ...
    'Position', [0.1 0.12 0.8 0.05], 'FontSize', 10);

handles.playBtn = uicontrol('Parent', handles.panel, 'Style', 'togglebutton', ...
    'String', 'Play/Pause', 'Units', 'normalized', ...
    'Position', [0.1 0.06 0.8 0.05], 'Callback', @(src,~) togglePlayback(src));

handles.frameSlider = uicontrol('Parent', fig, 'Style', 'slider', ...
    'Min', 1, 'Max', 100, 'Value', 1, ...
    'SliderStep', [1/100 0.1], ...
    'Units', 'normalized', 'Position', [0.3 0.01 0.65 0.03], ...
    'Callback', @(src,~) updateFrameFromSlider(src));

handles.frame = 1;
handles.playing = false;

guidata(fig, handles);
try
    plotSheet(handles.sheetMenu);
catch ME
    warning('Failed to auto-load sheet: %s', ME.message);
end

function plotSheet(src)
    handles = guidata(src);
    sheetname = get(src, 'String');
    if iscell(sheetname)
        selected = sheetname{get(src,'Value')};
    elseif ischar(sheetname)
        selected = sheetname;
    else
        warning('Invalid sheet name format.');
        return;
    end
    selected = sheetname{get(src,'Value')};
elseif ischar(sheetname)
    selected = sheetname;
else
    warning('Invalid sheet name format.');
    return;
end

    data = readmatrix(handles.filepath, 'Sheet', selected, 'Range', 'A4:Z2000');

    handles.time = data(:,2);
    handles.midhands = data(:,3:5)/100;
    handles.clubface = data(:,15:17)/100;

    % Midhands direction cosines (F–N): columns 6–14
    Xh = data(:,6:8); Yh = data(:,9:11);
    if size(data,2) >= 14
        if size(data,2) >= 14 && all(data(:,12:14) == 0, 'all')
            Zh = cross(Xh, Yh, 2); Zh = Zh ./ vecnorm(Zh,2,2);
        else
            Zh = data(:,12:14);
        end
    else
        Zh = cross(Xh, Yh, 2); Zh = Zh ./ vecnorm(Zh,2,2);
    end

    % Clubface direction cosines (R–Z): columns 18–26
    Xc = data(:,18:20); Yc = data(:,21:23);
    if size(data,2) >= 26
        if all(data(:,24:26) == 0, 'all')
            Zc = cross(Xc, Yc, 2); Zc = Zc ./ vecnorm(Zc,2,2);
        else
            Zc = data(:,24:26);
        end
    else
        Zc = cross(Xc, Yc, 2); Zc = Zc ./ vecnorm(Zc,2,2);
    end

    handles.handAxes = cat(3, Xh, Yh, Zh);
    handles.clubAxes = cat(3, Xc, Yc, Zc);

    % === Fixed axis limits ===
    allX = [handles.midhands(:,1); handles.clubface(:,1)];
    allY = [handles.midhands(:,2); handles.clubface(:,2)];
    allZ = [handles.midhands(:,3); handles.clubface(:,3)];
    pad = 0.1;
    handles.xlim = [min(allX)-pad, max(allX)+pad];
    handles.ylim = [min(allY)-pad, max(allY)+pad];
    handles.zlim = [min(allZ)-pad, max(allZ)+pad];
    xlim(handles.ax, handles.xlim);
    ylim(handles.ax, handles.ylim);
    zlim(handles.ax, handles.zlim);
    cla(handles.ax);
    handles.lineObj = plot3(handles.ax, NaN, NaN, NaN, 'k-', 'LineWidth', 2);
    c = {'r','g','b'};
    for j = 1:3
        handles.(['quiverClub' c{j}]) = quiver3(handles.ax,0,0,0,0,0,0, c{j}, 'LineWidth', 1.5);
        handles.(['quiverHands' c{j}]) = quiver3(handles.ax,0,0,0,0,0,0, c{j}, 'LineWidth', 1.5, 'LineStyle','--');
    end

    handles.frame = 1;
    set(handles.frameSlider, 'Min', 1);
    set(handles.frameSlider, 'Max', size(handles.midhands,1));
    set(handles.frameSlider, 'Value', 1);
    set(handles.frameSlider, 'SliderStep', [1/(size(handles.midhands,1)-1) 0.1]);
    guidata(src, handles);
end

function togglePlayback(src)
    while get(src, 'Value') && ishandle(src)
        handles = guidata(src);
        if handles.frame > size(handles.midhands,1)
            handles.frame = 1;
        end
        animateFrame(handles);
        set(handles.frameSlider, 'Value', handles.frame);
        handles.frame = handles.frame + 1;
        guidata(src, handles);
        pause(1/240);
    end
end

function animateFrame(handles)
    i = handles.frame;
    if i > size(handles.midhands,1), return; end
    A = handles.midhands(i,:); B = handles.clubface(i,:);
    shaftLength = norm(B - A);
    scale = 0.1 * shaftLength;

    for j = 1:3
        vecC = handles.clubAxes(i,:,j); vecC = vecC / (norm(vecC) + 1e-8);
        vecH = handles.handAxes(i,:,j); vecH = vecH / (norm(vecH) + 1e-8);
        color = {'r','g','b'}; tag = color{j};
        set(handles.(['quiverClub' tag]), 'XData', B(1), 'YData', B(2), 'ZData', B(3), ...
            'UData', scale*vecC(1), 'VData', scale*vecC(2), 'WData', scale*vecC(3), ...
            'Visible', ternary(get(handles.axisToggles(j), 'Value'), 'on', 'off'));
        set(handles.(['quiverHands' tag]), 'XData', A(1), 'YData', A(2), 'ZData', A(3), ...
            'UData', scale*vecH(1), 'VData', scale*vecH(2), 'WData', scale*vecH(3), ...
            'Visible', ternary(get(handles.axisToggles(j+3), 'Value'), 'on', 'off'));
        angles(j) = acosd(dot(vecC, vecH));
    end
    set(handles.lineObj, 'XData', [A(1) B(1)], 'YData', [A(2) B(2)], 'ZData', [A(3) B(3)]);
    set(handles.angleLabel, 'String', sprintf('X: %.1f  Y: %.1f  Z: %.1f', angles));
    drawnow;
end

function updateFrameFromSlider(src)
    handles = guidata(src);
    handles.frame = round(get(src, 'Value'));
    guidata(src, handles);
    animateFrame(handles);
end

function val = ternary(cond, t, f)
    if cond, val = t; else, val = f; end
end
