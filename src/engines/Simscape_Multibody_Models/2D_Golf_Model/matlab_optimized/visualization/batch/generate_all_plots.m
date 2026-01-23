function generate_all_plots(BASE, ZTCF, DELTA, BASEQ, ZTCFQ, DELTAQ, ZVCFTableQ, sim_config, plot_cfg)
% GENERATE_ALL_PLOTS - Generate all plots for all datasets
%
% Inputs:
%   BASE, ZTCF, DELTA - Full resolution data tables
%   BASEQ, ZTCFQ, DELTAQ, ZVCFTableQ - Q-tables for plotting
%   sim_config - Simulation configuration
%   plot_cfg - Plot configuration
%
% This function replaces SCRIPT_AllPlots.m and all the individual master
% plot scripts by using a registry-based approach with parameterized
% plotting functions.
%
% Author: Optimized Golf Swing Analysis System
% Date: 2025

    arguments
        BASE table
        ZTCF table
        DELTA table
        BASEQ table
        ZTCFQ table
        DELTAQ table
        ZVCFTableQ table
        sim_config (1,1) struct
        plot_cfg (1,1) struct
    end

    if sim_config.verbose
        fprintf('📊 Generating all plots...\n');
    end

    %% Create output directories
    base_plot_dir = sim_config.plots_path;
    datasets = {'BASE', 'ZTCF', 'DELTA', 'ZVCF'};

    for i = 1:length(datasets)
        dataset_dir = fullfile(base_plot_dir, [datasets{i} '_Charts']);
        if ~isfolder(dataset_dir)
            mkdir(dataset_dir);
        end
    end

    %% Define plot functions
    plot_functions = {
        @plot_angular_work, 'Angular_Work', 1
        @plot_angular_power, 'Angular_Power', 2
        @plot_linear_work, 'Linear_Work', 3
        @plot_total_work, 'Total_Work', 4
    };

    %% Generate plots for each dataset
    data_tables = struct('BASE', BASEQ, 'ZTCF', ZTCFQ, 'DELTA', DELTAQ, 'ZVCF', ZVCFTableQ);

    total_plots = length(datasets) * size(plot_functions, 1);
    plot_count = 0;

    for dataset_idx = 1:length(datasets)
        dataset_name = datasets{dataset_idx};
        data_table = data_tables.(dataset_name);

        if isempty(data_table)
            continue;
        end

        if sim_config.verbose
            fprintf('\n   Generating plots for %s...\n', dataset_name);
        end

        % Get base figure number for this dataset
        base_fig_num = plot_cfg.fig_num.(dataset_name);
        output_dir = fullfile(base_plot_dir, [dataset_name '_Charts']);

        % Generate each plot type
        for plot_idx = 1:size(plot_functions, 1)
            plot_func = plot_functions{plot_idx, 1};
            plot_name = plot_functions{plot_idx, 2};
            plot_offset = plot_functions{plot_idx, 3};

            fig_num = base_fig_num + plot_offset;

            try
                % Generate plot
                fig = plot_func(data_table, dataset_name, fig_num, plot_cfg);

                % Save plot
                if sim_config.save_plots
                    for fmt_idx = 1:length(plot_cfg.export_formats)
                        fmt = plot_cfg.export_formats{fmt_idx};
                        filename = fullfile(output_dir, ...
                            sprintf('%s_Plot_%s.%s', dataset_name, plot_name, fmt));

                        switch fmt
                            case 'fig'
                                savefig(fig, filename);
                            case 'png'
                                saveas(fig, filename, 'png');
                            case 'pdf'
                                saveas(fig, filename, 'pdf');
                            case 'eps'
                                saveas(fig, filename, 'epsc');
                        end
                    end
                end

                % Close figure if configured
                if plot_cfg.close_after_save
                    close(fig);
                end

                plot_count = plot_count + 1;
                if sim_config.show_progress && mod(plot_count, 5) == 0
                    fprintf('   Progress: %d/%d plots generated\n', plot_count, total_plots);
                end

            catch ME
                warning('Failed to generate plot %s for %s: %s', ...
                    plot_name, dataset_name, ME.message);
            end
        end
    end

    if sim_config.verbose
        fprintf('\n✅ Plot generation complete: %d plots created\n', plot_count);
        fprintf('   Plots saved to: %s\n', base_plot_dir);
    end

end
