function setup_camera_lock(ax_left, ax_right, all_points)
%SETUP_CAMERA_LOCK  Lock camera view and axis limits across two 3D axes.
%
%   SETUP_CAMERA_LOCK(AX_LEFT, AX_RIGHT, ALL_POINTS) computes a tight
%   bounding box around every point in ALL_POINTS (Mx3) and applies the
%   identical x/y/z limits, view angle, and DataAspectRatio to AX_LEFT
%   and AX_RIGHT. It also installs a linkprop on the camera properties so
%   that interactive rotation in either axes drives the other.
%
%   This is the canonical implementation of the "shared camera" rule
%   from VISUALIZATION_SPEC.md View 1.
%
%   GitHub issue: #3989.
    arguments
        ax_left   (1,1) matlab.graphics.axis.Axes
        ax_right  (1,1) matlab.graphics.axis.Axes
        all_points      double {mustBeNonempty}
    end

    if size(all_points, 2) ~= 3
        error("setup_camera_lock:badShape", ...
              "ALL_POINTS must be Mx3, got %dx%d", ...
              size(all_points, 1), size(all_points, 2));
    end

    finite_mask = all(isfinite(all_points), 2);
    pts = all_points(finite_mask, :);
    if isempty(pts)
        pts = zeros(1, 3);
    end

    pad = 0.05;  % 5 % padding
    mn = min(pts, [], 1);
    mx = max(pts, [], 1);
    span = max(mx - mn, 1e-3);
    xl = [mn(1) - pad * span(1), mx(1) + pad * span(1)];
    yl = [mn(2) - pad * span(2), mx(2) + pad * span(2)];
    zl = [mn(3) - pad * span(3), mx(3) + pad * span(3)];

    for ax = [ax_left, ax_right]
        xlim(ax, xl);
        ylim(ax, yl);
        zlim(ax, zl);
        view(ax, 35, 25);
        daspect(ax, [1 1 1]);
        grid(ax, "on");
        box(ax, "on");
    end

    % Link camera properties for interactive sessions. Stored as appdata
    % on the parent figure so the link survives as long as the figure does.
    try
        link = linkprop([ax_left, ax_right], ...
            {'CameraPosition', 'CameraTarget', 'CameraUpVector', ...
             'CameraViewAngle', 'XLim', 'YLim', 'ZLim'});
        setappdata(ancestor(ax_left, 'figure'), 'CameraLink', link);
    catch
        % linkprop unavailable in headless contexts: limits already match.
    end
end
