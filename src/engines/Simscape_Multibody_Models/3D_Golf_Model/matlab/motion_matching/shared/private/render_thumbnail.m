function ax = render_thumbnail(parent_fig, position, plot_fcn, tag)
%RENDER_THUMBNAIL  Render a viz function into an inset axes on a parent figure.
%
%   AX = RENDER_THUMBNAIL(PARENT_FIG, POSITION, PLOT_FCN, TAG) calls the
%   provided zero-argument PLOT_FCN handle, which is expected to return a
%   handle to a new (off-screen) figure.  All axes from that figure are
%   copied into PARENT_FIG via copyobj, repositioned into the rectangle
%   defined by POSITION (a 1x4 normalized [x y w h] vector), and tagged
%   with TAG.  The source figure is then closed.
%
%   This is the DRY hinge that lets plot_fit_quality_card.m embed View 1
%   and View 2 thumbnails without duplicating their drawing code.
%
%   Preconditions:
%     - PARENT_FIG is a valid matlab.ui.Figure.
%     - POSITION is a 1x4 numeric in [0,1] with positive width/height.
%     - PLOT_FCN is a function handle returning a figure handle.
%     - TAG is a non-empty string.
%   Postconditions:
%     - AX is a (possibly multi-element) array of axes parented to
%       PARENT_FIG, each carrying the supplied TAG.
%
%   GitHub issue: #3991.
    arguments
        parent_fig (1,1) matlab.ui.Figure
        position   (1,4) double {mustBeReal, mustBeNonnegative}
        plot_fcn   (1,1) function_handle
        tag        (1,1) string {mustBeNonzeroLengthText}
    end
    assert(position(3) > 0 && position(4) > 0, ...
        "render_thumbnail:badPosition", ...
        "POSITION width and height must be positive.");

    src_fig = plot_fcn();
    cleaner = onCleanup(@() local_safe_close(src_fig));
    if ~isgraphics(src_fig, 'figure')
        error("render_thumbnail:badPlotFcn", ...
              "plot_fcn must return a figure handle.");
    end

    src_axes = findall(src_fig, 'Type', 'axes');
    if isempty(src_axes)
        ax = matlab.graphics.axis.Axes.empty;
        return;
    end

    new_axes = copyobj(src_axes, parent_fig);
    ax = local_layout_axes(new_axes, position, tag);

    delete(cleaner);  % triggers close of src_fig
end

% ---------------------------------------------------------------------
function ax = local_layout_axes(new_axes, position, tag)
%LOCAL_LAYOUT_AXES  Tile a set of copied axes into the inset rectangle.
%   For a single axis we fill the rectangle.  For an Nx1 stack (View 2)
%   we partition the height; for a 1xN row (View 1) we partition the
%   width based on the original Position.x ordering.
    n = numel(new_axes);
    if n == 1
        new_axes(1).Position = position;
        new_axes(1).Tag = char(tag);
        new_axes(1).FontSize = max(7, new_axes(1).FontSize - 2);
        ax = new_axes(1);
        return;
    end

    src_pos = arrayfun(@(a) a.Position, new_axes, 'UniformOutput', false);
    src_pos = vertcat(src_pos{:});
    is_stacked = std(src_pos(:, 1)) < std(src_pos(:, 2));

    [~, order] = sort(src_pos(:, 2));
    if ~is_stacked
        [~, order] = sort(src_pos(:, 1));
    end
    new_axes = new_axes(order);

    x = position(1); y = position(2); w = position(3); h = position(4);
    pad = 0.005;
    if is_stacked
        cell_h = (h - pad * (n - 1)) / n;
        for k = 1:n
            new_axes(k).Position = [x, y + (k - 1) * (cell_h + pad), w, cell_h];
            new_axes(k).Tag = char(tag);
            new_axes(k).FontSize = max(7, new_axes(k).FontSize - 2);
        end
    else
        cell_w = (w - pad * (n - 1)) / n;
        for k = 1:n
            new_axes(k).Position = [x + (k - 1) * (cell_w + pad), y, cell_w, h];
            new_axes(k).Tag = char(tag);
            new_axes(k).FontSize = max(7, new_axes(k).FontSize - 2);
        end
    end
    ax = new_axes;
end

function local_safe_close(fig_handle)
    try
        if isgraphics(fig_handle, 'figure')
            close(fig_handle);
        end
    catch
        % nothing to do; suppress so cleanup is best-effort
    end
end
