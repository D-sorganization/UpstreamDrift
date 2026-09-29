function h = draw_club_skeleton(ax, butt_xyz, head_xyz, color, opts)
%DRAW_CLUB_SKELETON  Draw club skeleton (butt -> clubhead) on axes AX.
%
%   h = DRAW_CLUB_SKELETON(AX, BUTT_XYZ, HEAD_XYZ, COLOR, OPTS) plots a
%   single line segment from BUTT_XYZ to HEAD_XYZ on the 3D axes AX with
%   the given COLOR (hex string or RGB triplet). Returns a struct of
%   handles with fields:
%
%     .skeleton  primitive line for the butt->head segment
%     .butt      scatter marker at the butt
%     .head      scatter marker at the clubhead
%
%   This helper exists so plot_trajectory_overlay and
%   animate_trajectory_overlay share identical primitive drawing
%   (DRY — see CODING_STANDARDS.md).
%
%   GitHub issue: #3989.
    arguments
        ax            (1,1) matlab.graphics.axis.Axes
        butt_xyz      (1,3) double {mustBeFinite}
        head_xyz      (1,3) double {mustBeFinite}
        color               % hex string or [r g b]
        opts          (1,1) struct = default_viz_options()
    end

    holdState = ishold(ax);
    hold(ax, "on");

    h = struct();
    h.skeleton = plot3(ax, ...
        [butt_xyz(1), head_xyz(1)], ...
        [butt_xyz(2), head_xyz(2)], ...
        [butt_xyz(3), head_xyz(3)], ...
        '-', 'Color', color, 'LineWidth', 2.0);

    h.butt = scatter3(ax, butt_xyz(1), butt_xyz(2), butt_xyz(3), ...
        36, 'MarkerEdgeColor', color, 'MarkerFaceColor', color);
    h.head = scatter3(ax, head_xyz(1), head_xyz(2), head_xyz(3), ...
        72, 'MarkerEdgeColor', color, 'MarkerFaceColor', color);

    if ~holdState
        hold(ax, "off");
    end

    % Suppress unused-variable warning for opts; kept in signature for
    % future styling overrides without a breaking API change.
    if isfield(opts, "fontsize_axes")
        ax.FontSize = opts.fontsize_axes;
    end
end
