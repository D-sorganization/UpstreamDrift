function animateTW_WiffleClub()
% Interactive animation of TW wiffle swing (midhands to clubface)

% === Prompt for file selection ===
[file, path] = uigetfile('*.xlsx', 'Select the swing data file');
if isequal(file, 0)
    disp('User canceled file selection.');
    return;
end
fullpath = fullfile(path, file);

% === Load data from TW_wiffle sheet ===
data = readmatrix(fullpath, 'Sheet', 'TW_wiffle', 'Range', 'B4:Q2000');
time = data(:,1);
midhands = data(:,2:4) / 100;     % columns C:D:E
clubface = data(:,14:16) / 100;   % columns O:P:Q

% === Setup figure and fixed view ===
fig = figure('Name', 'TW Wiffle Club Animation', 'Color', [1 1 1]);
ax = axes('Parent', fig); grid on; axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');
view(ax, [0 0]);  % face-on: looking down +X toward golfer

% Fixed plot limits based on entire trajectory
allX = [midhands(:,1); clubface(:,1)];
allY = [midhands(:,2); clubface(:,2)];
allZ = [midhands(:,3); clubface(:,3)];
padding = 0.1;

xlim(ax, [min(allX)-padding, max(allX)+padding]);
ylim(ax, [min(allY)-padding, max(allY)+padding]);
zlim(ax, [min(allZ)-padding, max(allZ)+padding]);

hold(ax, 'on');
hLine = plot3(NaN, NaN, NaN, 'k-', 'LineWidth', 3);

% === Animate ===
for i = 1:min(length(time), size(midhands,1))
    X = [midhands(i,1), clubface(i,1)];
    Y = [midhands(i,2), clubface(i,2)];
    Z = [midhands(i,3), clubface(i,3)];

    set(hLine, 'XData', X, 'YData', Y, 'ZData', Z);
    drawnow;
    pause(1/60);
end
end
