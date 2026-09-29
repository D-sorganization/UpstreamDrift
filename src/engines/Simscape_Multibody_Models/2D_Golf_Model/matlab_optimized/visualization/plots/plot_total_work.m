function fig = plot_total_work(data_table, dataset_name, fig_num, plot_cfg)
% PLOT_TOTAL_WORK - Plot total work for all joints
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
    if ismember('LSTotalWork', data_table.Properties.VariableNames)
        % Use pre-calculated total work
        plot(data_table.Time, data_table.LSTotalWork, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.LS, ...
            'DisplayName', 'LS Total Work');
        plot(data_table.Time, data_table.RSTotalWork, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.RS, ...
            'DisplayName', 'RS Total Work');
        plot(data_table.Time, data_table.LETotalWork, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.LE, ...
            'DisplayName', 'LE Total Work');
        plot(data_table.Time, data_table.RETotalWork, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.RE, ...
            'DisplayName', 'RE Total Work');
        plot(data_table.Time, data_table.LWTotalWork, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.LW, ...
            'DisplayName', 'LW Total Work');
        plot(data_table.Time, data_table.RWTotalWork, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.RW, ...
            'DisplayName', 'RW Total Work');
    else
        % Calculate as sum of angular + linear
        plot(data_table.Time, data_table.LSAngularWorkonArm + data_table.LSLinearWorkonArm, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.LS, ...
            'DisplayName', 'LS Total Work');
        plot(data_table.Time, data_table.RSAngularWorkonArm + data_table.RSLinearWorkonArm, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.RS, ...
            'DisplayName', 'RS Total Work');
        plot(data_table.Time, data_table.LEAngularWorkonForearm + data_table.LELinearWorkonForearm, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.LE, ...
            'DisplayName', 'LE Total Work');
        plot(data_table.Time, data_table.REAngularWorkonForearm + data_table.RELinearWorkonForearm, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.RE, ...
            'DisplayName', 'RE Total Work');
        plot(data_table.Time, data_table.LWAngularWorkonClub + data_table.LHLinearWorkonClub, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.LW, ...
            'DisplayName', 'LW Total Work');
        plot(data_table.Time, data_table.RWAngularWorkonClub + data_table.RHLinearWorkonClub, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.RW, ...
            'DisplayName', 'RW Total Work');
    end

    %% Format plot
    xlabel('Time (s)', 'FontSize', plot_cfg.axis_font_size);
    ylabel('Work (J)', 'FontSize', plot_cfg.axis_font_size);

    if plot_cfg.show_grid
        grid on;
    end

    if plot_cfg.show_legend
        legend('Location', 'southeast', 'FontSize', plot_cfg.font_size);
    end

    title('Total Work at Each Joint', 'FontSize', plot_cfg.title_font_size);
    subtitle(dataset_name, 'FontSize', plot_cfg.font_size);

    hold off;

end
