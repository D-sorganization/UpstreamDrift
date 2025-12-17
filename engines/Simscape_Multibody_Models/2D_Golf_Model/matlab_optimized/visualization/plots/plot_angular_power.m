function fig = plot_angular_power(data_table, dataset_name, fig_num, plot_cfg)
% PLOT_ANGULAR_POWER - Plot angular power for all joints
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

    %% Plot data (using calculated power if available)
    if ismember('LSAngularPower', data_table.Properties.VariableNames)
        % Use pre-calculated power
        plot(data_table.Time, data_table.LSAngularPower, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.LS, ...
            'DisplayName', 'LS Angular Power');
        plot(data_table.Time, data_table.RSAngularPower, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.RS, ...
            'DisplayName', 'RS Angular Power');
        plot(data_table.Time, data_table.LEAngularPower, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.LE, ...
            'DisplayName', 'LE Angular Power');
        plot(data_table.Time, data_table.REAngularPower, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.RE, ...
            'DisplayName', 'RE Angular Power');
        plot(data_table.Time, data_table.LWAngularPower, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.LW, ...
            'DisplayName', 'LW Angular Power');
        plot(data_table.Time, data_table.RWAngularPower, ...
            'LineWidth', plot_cfg.line_width, 'Color', plot_cfg.colors.RW, ...
            'DisplayName', 'RW Angular Power');
    else
        % Calculate power as derivative of work
        dt = [diff(data_table.Time); data_table.Time(end) - data_table.Time(end-1)];

        LSP = [0; diff(data_table.LSAngularWorkonArm)] ./ dt;
        RSP = [0; diff(data_table.RSAngularWorkonArm)] ./ dt;
        LEP = [0; diff(data_table.LEAngularWorkonForearm)] ./ dt;
        REP = [0; diff(data_table.REAngularWorkonForearm)] ./ dt;
        LWP = [0; diff(data_table.LWAngularWorkonClub)] ./ dt;
        RWP = [0; diff(data_table.RWAngularWorkonClub)] ./ dt;

        plot(data_table.Time, LSP, 'LineWidth', plot_cfg.line_width, ...
            'Color', plot_cfg.colors.LS, 'DisplayName', 'LS Angular Power');
        plot(data_table.Time, RSP, 'LineWidth', plot_cfg.line_width, ...
            'Color', plot_cfg.colors.RS, 'DisplayName', 'RS Angular Power');
        plot(data_table.Time, LEP, 'LineWidth', plot_cfg.line_width, ...
            'Color', plot_cfg.colors.LE, 'DisplayName', 'LE Angular Power');
        plot(data_table.Time, REP, 'LineWidth', plot_cfg.line_width, ...
            'Color', plot_cfg.colors.RE, 'DisplayName', 'RE Angular Power');
        plot(data_table.Time, LWP, 'LineWidth', plot_cfg.line_width, ...
            'Color', plot_cfg.colors.LW, 'DisplayName', 'LW Angular Power');
        plot(data_table.Time, RWP, 'LineWidth', plot_cfg.line_width, ...
            'Color', plot_cfg.colors.RW, 'DisplayName', 'RW Angular Power');
    end

    %% Format plot
    xlabel('Time (s)', 'FontSize', plot_cfg.axis_font_size);
    ylabel('Power (W)', 'FontSize', plot_cfg.axis_font_size);

    if plot_cfg.show_grid
        grid on;
    end

    if plot_cfg.show_legend
        legend('Location', 'best', 'FontSize', plot_cfg.font_size);
    end

    title('Angular Power on Distal Segment', 'FontSize', plot_cfg.title_font_size);
    subtitle(dataset_name, 'FontSize', plot_cfg.font_size);

    hold off;

end
