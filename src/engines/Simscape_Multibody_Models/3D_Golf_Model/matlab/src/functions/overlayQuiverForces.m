function h = overlayQuiverForces(Pcell, Vcell, varargin)
% OVERLAYQUIVERFORCES - Overlay multiple 3D quiver datasets with consistent scaling and sliders
%
% Usage:
%   h = overlayQuiverForces({P1, P2, ...}, {V1, V2, ...}, ...);
%   Each P and V must be Nx3 or cell arrays of [X,Y,Z] and [U,V,W] vectors.
%
% Options (Name-Value Pairs):
%   'Colors'       : Cell array of RGB triplets
%   'Tags'         : Cell array of legend labels
%   'Landmark'     : [x y z] to shift all origins
%   'ShellType'    : 'none' | 'sphere' | 'plane'
%   'ShellAlpha'   : Transparency for shell (default: 0.1)
%   'Title'        : Plot title (default: 'Overlayed Vector Field')
%   'ShowSliders'  : true/false (default: true)

% ---------------- Input Handling ----------------
p = inputParser;
p.addParameter('Colors', {});
p.addParameter('Tags', {});
p.addParameter('Landmark', []);
p.addParameter('ShellType', 'none');
p.addParameter('ShellAlpha', 0.1);
p.addParameter('Title', 'Overlayed Vector Field');
p.addParameter('ShowSliders', true);
p.parse(varargin{:});
opts = p.Results;

numSets = length(Pcell);
assert(numSets == length(Vcell), 'Mismatched P/V cell input counts');

% Normalize inputs to Nx3 matrices
for i = 1:numSets
    if iscell(Pcell{i})
        Pcell{i} = [Pcell{i}{1}, Pcell{i}{2}, Pcell{i}{3}];
    end
    if iscell(Vcell{i})
        Vcell{i} = [Vcell{i}{1}, Vcell{i}{2}, Vcell{i}{3}];
    end
end

% ---------------- Compute Global Max Magnitude ----------------
allMags = [];
for i = 1:numSets
    V = Vcell{i};
    mags = sqrt(sum(V.^2, 2));
    allMags = [allMags; mags];
end
maxMag = max(allMags);
if maxMag == 0, maxMag = 1; end
sharedScale = 1 / maxMag;

% ---------------- Create Figure and Axes ----------------
h.fig = figure('Name','Overlayed Vector Field','Color','w');
h.ax = axes(h.fig); hold on; axis equal; grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title(opts.Title);
h.shell = [];
h.quivers = gobjects(numSets, 1);
h.origins = cell(numSets, 1);
h.vectors = cell(numSets, 1);

% ---------------- Optional Landmark Shift ----------------
for i = 1:numSets
    P = Pcell{i};
    V = Vcell{i};
    if ~isempty(opts.Landmark)
        P = P - opts.Landmark;
    end
    h.origins{i} = P;
    h.vectors{i} = V;
    color = [0 0 0];
    if i <= length(opts.Colors)
        color = opts.Colors{i};
    end
    tag = "";
    if i <= length(opts.Tags)
        tag = opts.Tags{i};
    end
    h.quivers(i) = quiver3(h.ax, NaN, NaN, NaN, NaN, NaN, NaN, sharedScale, 'LineWidth', 1.5, 'Color', color);
    if tag ~= ""
        h.quivers(i).DisplayName = tag;
    end
end

if ~isempty(opts.Tags)
    legend(h.ax, 'show');
end

% ---------------- Draw Shell Once ----------------
if ~strcmpi(opts.ShellType, 'none')
    allX = cell2mat(cellfun(@(p)p(:,1), h.origins, 'UniformOutput', false));
    allY = cell2mat(cellfun(@(p)p(:,2), h.origins, 'UniformOutput', false));
    allZ = cell2mat(cellfun(@(p)p(:,3), h.origins, 'UniformOutput', false));
    switch lower(opts.ShellType)
        case 'sphere'
            [sx, sy, sz] = sphere(30);
            r = max(range([allX; allY; allZ])) / 3;
            cx = mean(allX); cy = mean(allY); cz = mean(allZ);
            h.shell = surf(h.ax, r*sx+cx, r*sy+cy, r*sz+cz, 'FaceAlpha', opts.ShellAlpha, ...
                          'EdgeColor','none','FaceColor',[0.5 0.8 1]);
        case 'plane'
            [px, py] = meshgrid(linspace(min(allX),max(allX),20), ...
                                 linspace(min(allY),max(allY),20));
            zVal = mean(allZ);
            h.shell = surf(h.ax, px, py, zVal*ones(size(px)), 'FaceAlpha', opts.ShellAlpha, ...
                          'EdgeColor','none','FaceColor',[0.8 0.5 1]);
    end
end

% ---------------- Sliders (Optional) ----------------
h.scaleSlider = [];
h.densitySlider = [];
if opts.ShowSliders
    h.panel = uipanel(h.fig, 'Position', [0.01 0.01 0.98 0.10], 'Title', 'Vector Controls');
    h.scaleSlider = uicontrol(h.panel, 'Style','slider', 'Min', 0.1, 'Max', 3.0, ...
        'Value', 1.0, 'Units','normalized', 'Position',[0.02 0.55 0.96 0.4], ...
        'Callback', @(s,~) updateOverlayQuivers(h, sharedScale));
    h.densitySlider = uicontrol(h.panel, 'Style','slider', 'Min', 1, 'Max', 100, ...
        'Value', min(30, size(h.origins{1},1)), 'Units','normalized', 'Position',[0.02 0.05 0.96 0.4], ...
        'Callback', @(s,~) updateOverlayQuivers(h, sharedScale));
end

updateOverlayQuivers(h, sharedScale);
end

function updateOverlayQuivers(h, baseScale)
    scaleFactor = 1.0;
    if isfield(h, 'scaleSlider') && isvalid(h.scaleSlider)
        scaleFactor = get(h.scaleSlider, 'Value');
    end
    n = inf;
    if isfield(h, 'densitySlider') && isvalid(h.densitySlider)
        n = round(get(h.densitySlider, 'Value'));
    end
    for i = 1:length(h.quivers)
        if isvalid(h.quivers(i))
            X = h.origins{i};
            V = h.vectors{i};
            idx = round(linspace(1, size(X,1), min(n, size(X,1))));
            set(h.quivers(i), 'XData', X(idx,1), 'YData', X(idx,2), 'ZData', X(idx,3), ...
                              'UData', V(idx,1), 'VData', V(idx,2), 'WData', V(idx,3), ...
                              'AutoScaleFactor', baseScale * scaleFactor);
        end
    end
end
