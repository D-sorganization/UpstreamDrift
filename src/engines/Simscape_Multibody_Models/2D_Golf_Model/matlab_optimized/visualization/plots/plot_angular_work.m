function fig = plot_angular_work(data_table, dataset_name, fig_num, plot_cfg)
% PLOT_ANGULAR_WORK - Plot angular work for all joints
%
% Inputs:
%   data_table - Data table (BASEQ, ZTCFQ, DELTAQ, or ZVCFQ)
%   dataset_name - Name of dataset ('BASE', 'ZTCF', 'DELTA', 'ZVCF')
%   fig_num - Figure number
%   plot_cfg - Plot configuration from plot_config()
%
% Returns:
%   fig - Figure handle
%
% This function replaces:
%   - SCRIPT_101_PLOT_BaseData_AngularWork.m
%   - SCRIPT_301_PLOT_ZTCF_AngularWork.m
%   - SCRIPT_501_PLOT_DELTA_AngularWork.m
%
% Author: Optimized Golf Swing Analysis System
% Date: 2025

    arguments
        data_table table
        dataset_name (1,:) char
        fig_num (1,1) double
        plot_cfg struct
    end

    %% Create figure
    fig = figure(fig_num);
    clf(fig);
    set(fig, 'Position', [100, 100, plot_cfg.figure_width, plot_cfg.figure_height]);
    hold on;

    %% Plot data
    plot(data_table.Time, data_table.LSAngularWorkonArm, ...
        'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.LS, ...
        'DisplayName', 'LS Angular Work');

    plot(data_table.Time, data_table.RSAngularWorkonArm, ...
        'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.RS, ...
        'DisplayName', 'RS Angular Work');

    plot(data_table.Time, data_table.LEAngularWorkonForearm, ...
        'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.LE, ...
        'DisplayName', 'LE Angular Work');

    plot(data_table.Time, data_table.REAngularWorkonForearm, ...
        'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.RE, ...
        'DisplayName', 'RE Angular Work');

    plot(data_table.Time, data_table.LWAngularWorkonClub, ...
        'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.LW, ...
        'DisplayName', 'LW Angular Work');

    plot(data_table.Time, data_table.RWAngularWorkonClub, ...
        'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.RW, ...
        'DisplayName', 'RW Angular Work');

    %% Format plot
    xlabel('Time (s)', 'FontSize', plot_cfg.axis_font_size);
    ylabel('Work (J)', 'FontSize', plot_cfg.axis_font_size);

    if plot_cfg.show_grid
        grid on;
    end

    if plot_cfg.show_legend
        legend('Location', 'southeast', 'FontSize', plot_cfg.font_size);
    end

    title('Angular Work on Distal Segment', 'FontSize', plot_cfg.title_font_size);
    subtitle(dataset_name, 'FontSize', plot_cfg.font_size);

    hold off;

end
