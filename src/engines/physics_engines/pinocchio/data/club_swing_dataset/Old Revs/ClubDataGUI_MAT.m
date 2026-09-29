function ClubDataGUI_MAT()
% ClubDataGUI_MAT: Visualize golf club motion using preprocessed MAT files

datasets = {'TW_wiffle','TW_ProV1','GW_wiffle','GW_ProV1'};
data_paths = containers.Map(datasets, {
    'TW_wiffle_data.mat','TW_ProV1_data.mat',...
    'GW_wiffle_data.mat','GW_ProV1_data.mat'});

fig = figure('Name', 'Club Animation (MAT)', 'NumberTitle', 'off', ...
    'Color', [1 1 1], 'Position', [100 100 1200 700]);

handles.ax = axes('Parent', fig, 'Position', [0.3 0.25 0.65 0.7]);
axis equal; grid on;
xlabel('X'); ylabel('Y'); zlabel('Z'); hold on;

handles.panel = uipanel('Parent', fig, 'Title', 'Controls', 'FontSize', 10, ...
    'Units', 'normalized', 'Position', [0.01 0.25 0.25 0.7]);

% Dropdown to select dataset
handles.datasetMenu = uicontrol('Parent', handles.panel, 'Style', 'popupmenu', ...
    'String', datasets, ...
    'Units', 'normalized', 'Position', [0.1 0.92 0.8 0.05], ...
    'Callback', @(src,~) loadDataset(src));

% View buttons (radio-style checkboxes)
views = {'Face-On','Down-the-Line','Top-Down','Isometric'};
viewAngles = {[180 0],[0 0],[0 90],[-45 25]};
for i = 1:length(views)
    handles.viewChecks(i) = uicontrol('Parent', handles.panel, 'Style', 'checkbox', ...
        'String', views{i}, 'Units', 'normalized', ...
        'Position', [0.1 0.86 - 0.05*i 0.8 0.05], ...
        'Callback', @(src,~) setView(src, i));
end

% Playback and slider
handles.playBtn = uicontrol('Parent', handles.panel, 'Style', 'togglebutton', ...
    'String', 'Play/Pause', 'Units', 'normalized', ...
    'Position', [0.1 0.1 0.8 0.05], ...
    'Callback', @(src,~) togglePlayback());

handles.frameSlider = uicontrol('Parent', fig, 'Style', 'slider', ...
    'Min', 1, 'Max', 100, 'Value', 1, ...
    'SliderStep', [1/100 0.1], ...
    'Units', 'normalized', 'Position', [0.3 0.01 0.65 0.03], ...
    'Callback', @(src,~) updateFrame());

% Shared state
handles.frame = 1;
handles.playing = false;
handles.viewAngles = viewAngles;
handles.data = [];
handles.quivers = gobjects(1, 6);
guidata(fig, handles);

% === Subfunctions ===
    function loadDataset(src)
        handles = guidata(src);
        val = get(handles.datasetMenu, 'Value');
        choices = get(handles.datasetMenu, 'String');
        fname = data_paths(choices{val});
        if isfile(fname)
            S = load(fname);
            handles.data = S.data;
            handles.N = length(S.data.time);
            set(handles.frameSlider, 'Max', handles.N, 'Value', 1, ...
                'SliderStep', [1/handles.N 0.1]);
            handles.frame = 1;
            updateFrame();
            guidata(src, handles);
        else
            errordlg(['File not found: ' fname]);
        end
    end

    function togglePlayback()
        handles = guidata(gcf);
        handles.playing = ~handles.playing;
        guidata(gcf, handles);
        while handles.playing && ishandle(handles.ax)
            handles = guidata(gcf);
            if handles.frame >= handles.N
                handles.frame = 1;
            else
                handles.frame = handles.frame + 1;
            end
            set(handles.frameSlider, 'Value', handles.frame);
            updateFrame();
            pause(1/240);
        end
    end

    function updateFrame()
        handles = guidata(gcf);
        i = round(get(handles.frameSlider, 'Value'));
        handles.frame = i;
        A = handles.data.midhands_xyz(i,:);
        B = handles.data.clubface_xyz(i,:);
        shaftLength = norm(B - A);
        scale = 0.1 * shaftLength;

        % Axes
        Xh = handles.data.midhands_dircos(i,1:3);
        Yh = handles.data.midhands_dircos(i,4:6);
        Zh = handles.data.midhands_dircos(i,7:9);
        Xc = handles.data.clubface_dircos(i,1:3);
        Yc = handles.data.clubface_dircos(i,4:6);
        Zc = handles.data.clubface_dircos(i,7:9);

        cla(handles.ax);
        plot3(handles.ax, [A(1), B(1)], [A(2), B(2)], [A(3), B(3)], 'k-', 'LineWidth', 2);
        hold(handles.ax, 'on');
        colors = {'r','g','b'};
        for j = 1:3
            handles.quivers(j) = quiver3(handles.ax, B(1), B(2), B(3), ...
                scale * Xc(j), scale * Yc(j), scale * Zc(j), ...
                'Color', colors{j}, 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
            handles.quivers(j+3) = quiver3(handles.ax, A(1), A(2), A(3), ...
                scale * Xh(j), scale * Yh(j), scale * Zh(j), ...
                'Color', colors{j}, 'LineWidth', 1.5, 'LineStyle', '--', 'MaxHeadSize', 0.5);
        end
        axis(handles.ax, 'equal');
        drawnow;
        guidata(gcf, handles);
    end

    function setView(src, idx)
        handles = guidata(src);
        % Deselect all others
        for k = 1:length(handles.viewChecks)
            if k ~= idx
                set(handles.viewChecks(k), 'Value', 0);
            end
        end
        set(handles.viewChecks(idx), 'Value', 1);
        view(handles.ax, handles.viewAngles{idx});
    end
end
