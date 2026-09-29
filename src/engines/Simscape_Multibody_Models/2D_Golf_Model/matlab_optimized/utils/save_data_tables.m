function save_data_tables(config, BASE, ZTCF, DELTA, BASEQ, ZTCFQ, DELTAQ, ZVCFTable, ZVCFTableQ, SummaryTable, ClubQuivers)
% SAVE_DATA_TABLES - Save all data tables to output directory
%
% Inputs:
%   config - Configuration structure
%   BASE, ZTCF, DELTA - Full resolution tables
%   BASEQ, ZTCFQ, DELTAQ - Q-tables for plotting
%   ZVCFTable, ZVCFTableQ - ZVCF tables
%   SummaryTable - Summary statistics
%   ClubQuivers - Club quiver data structure
%
% This function saves all generated tables to .mat files in the output
% directory specified in the configuration.
%
% Author: Optimized Golf Swing Analysis System
% Date: 2025

    arguments
        config struct
        BASE
        ZTCF
        DELTA
        BASEQ
        ZTCFQ
        DELTAQ
        ZVCFTable = []
        ZVCFTableQ = []
        SummaryTable = []
        ClubQuivers = []
    end

    if ~config.save_tables
        return;
    end

    if config.verbose
        fprintf('💾 Saving data tables...\n');
    end

    %% Create output directory
    try
        if exist(config.output_path, 'dir') ~= 7
            mkdir(config.output_path);
        end
    catch
        % Ignore directory creation errors
    end

    %% Save main tables
    save(fullfile(config.output_path, 'BASE.mat'), 'BASE', '-v7.3');
    save(fullfile(config.output_path, 'ZTCF.mat'), 'ZTCF', '-v7.3');
    save(fullfile(config.output_path, 'DELTA.mat'), 'DELTA', '-v7.3');

    if config.verbose
        fprintf('   Saved BASE.mat (%d rows)\n', height(BASE));
        fprintf('   Saved ZTCF.mat (%d rows)\n', height(ZTCF));
        fprintf('   Saved DELTA.mat (%d rows)\n', height(DELTA));
    end

    %% Save Q-tables
    save(fullfile(config.output_path, 'BASEQ.mat'), 'BASEQ', '-v7.3');
    save(fullfile(config.output_path, 'ZTCFQ.mat'), 'ZTCFQ', '-v7.3');
    save(fullfile(config.output_path, 'DELTAQ.mat'), 'DELTAQ', '-v7.3');

    if config.verbose
        fprintf('   Saved BASEQ.mat (%d rows)\n', height(BASEQ));
        fprintf('   Saved ZTCFQ.mat (%d rows)\n', height(ZTCFQ));
        fprintf('   Saved DELTAQ.mat (%d rows)\n', height(DELTAQ));
    end

    %% Save ZVCF tables if available
    if ~isempty(ZVCFTable)
        save(fullfile(config.output_path, 'ZVCFTable.mat'), 'ZVCFTable', '-v7.3');
        if config.verbose
            fprintf('   Saved ZVCFTable.mat (%d rows)\n', height(ZVCFTable));
        end
    end

    if ~isempty(ZVCFTableQ)
        save(fullfile(config.output_path, 'ZVCFTableQ.mat'), 'ZVCFTableQ', '-v7.3');
        if config.verbose
            fprintf('   Saved ZVCFTableQ.mat (%d rows)\n', height(ZVCFTableQ));
        end
    end

    %% Save summary table if available
    if ~isempty(SummaryTable)
        save(fullfile(config.output_path, 'SummaryTable.mat'), 'SummaryTable', '-v7.3');
        if config.verbose
            fprintf('   Saved SummaryTable.mat\n');
        end
    end

    %% Save club quiver data if available
    if ~isempty(ClubQuivers)
        if isfield(ClubQuivers, 'AlphaReversal')
            ClubQuiverAlphaReversal = ClubQuivers.AlphaReversal;
            save(fullfile(config.output_path, 'ClubQuiverAlphaReversal.mat'), ...
                'ClubQuiverAlphaReversal', '-v7.3');
        end
        if isfield(ClubQuivers, 'MaxCHS')
            ClubQuiverMaxCHS = ClubQuivers.MaxCHS;
            save(fullfile(config.output_path, 'ClubQuiverMaxCHS.mat'), ...
                'ClubQuiverMaxCHS', '-v7.3');
        end
        if isfield(ClubQuivers, 'ZTCFAlphaReversal')
            ClubQuiverZTCFAlphaReversal = ClubQuivers.ZTCFAlphaReversal;
            save(fullfile(config.output_path, 'ClubQuiverZTCFAlphaReversal.mat'), ...
                'ClubQuiverZTCFAlphaReversal', '-v7.3');
        end
        if isfield(ClubQuivers, 'DELTAAlphaReversal')
            ClubQuiverDELTAAlphaReversal = ClubQuivers.DELTAAlphaReversal;
            save(fullfile(config.output_path, 'ClubQuiverDELTAAlphaReversal.mat'), ...
                'ClubQuiverDELTAAlphaReversal', '-v7.3');
        end
        if config.verbose
            fprintf('   Saved club quiver data\n');
        end
    end

    if config.verbose
        fprintf('✅ All tables saved to: %s\n', config.output_path);
    end

end
