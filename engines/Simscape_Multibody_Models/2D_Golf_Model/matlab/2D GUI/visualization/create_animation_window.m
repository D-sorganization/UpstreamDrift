function [fig, ax, animation_handles] = create_animation_window(config)
% CREATE_ANIMATION_WINDOW - Create the animation window for golf swing visualization
%
% Inputs:
%   config - Configuration structure from model_config()
%
% Returns:
%   fig - Figure handle
%   ax - Axes handle
%   animation_handles - Structure containing animation plot handles
%
% This function creates a figure with axes for displaying the golf swing animation

    % Create figure
    fig = figure('Name', [config.gui_title ' - Animation'], ...
                 'NumberTitle', 'off', ...
                 'Position', [100, 100, 800, 600], ...
                 'Color', config.colors.background, ...
                 'MenuBar', 'none', ...
                 'ToolBar', 'none', ...
                 'Resize', 'on');

    % Create axes
    ax = axes('Parent', fig, ...
              'Position', [0.1, 0.1, 0.8, 0.8], ...
              'Box', 'on', ...
              'GridLineStyle', ':', ...
              'GridAlpha', 0.3);

    % Set up axes properties
    hold(ax, 'on');
    grid(ax, 'on');
    xlabel(ax, 'X Position (m)', 'FontSize', config.plot_font_size);
    ylabel(ax, 'Y Position (m)', 'FontSize', config.plot_font_size);
    title(ax, '2D Golf Swing Animation', 'FontSize', config.plot_font_size + 2);

    % Set axis limits (will be updated with data)
    xlim(ax, [-2, 2]);
    ylim(ax, [-1, 1]);
    axis(ax, 'equal');

    % Create animation handles structure
    animation_handles = struct();

    % Create plot handles for different components
    animation_handles.club_shaft = plot(ax, NaN, NaN, 'b-', 'LineWidth', 3);
    animation_handles.club_head = plot(ax, NaN, NaN, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    animation_handles.hands = plot(ax, NaN, NaN, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
    animation_handles.arms = plot(ax, NaN, NaN, 'k-', 'LineWidth', 2);
    animation_handles.torso = plot(ax, NaN, NaN, 'k-', 'LineWidth', 2);

    % Create text handle for time display
    animation_handles.time_text = text(ax, 0.02, 0.98, 'Time: 0.000s', ...
                                      'Units', 'normalized', ...
                                      'FontSize', 12, ...
                                      'BackgroundColor', 'white', ...
                                      'EdgeColor', 'black');

    % Create legend
    legend(ax, [animation_handles.club_shaft, animation_handles.club_head, ...
                animation_handles.hands, animation_handles.arms], ...
           {'Club Shaft', 'Club Head', 'Hands', 'Arms'}, ...
           'Location', 'northeast', 'FontSize', 10);

    fprintf('✅ Animation window created successfully\n');

end
