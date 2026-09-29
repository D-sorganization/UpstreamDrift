function [BASE, ZTCF, DELTA, BASEQ, ZTCFQ, DELTAQ, ZVCFTable, ZVCFTableQ, SummaryTable, ClubQuivers] = ...
    run_additional_processing(config, BASE, ZTCF, DELTA, BASEQ, ZTCFQ, DELTAQ)
% RUN_ADDITIONAL_PROCESSING - Run all additional processing scripts
%
% Inputs:
%   config - Configuration structure
%   BASE, ZTCF, DELTA, BASEQ, ZTCFQ, DELTAQ - Data tables
%
% Returns:
%   Enhanced tables with additional calculated quantities
%   ZVCFTable, ZVCFTableQ - Zero Velocity Counterfactual data
%   SummaryTable - Summary statistics
%   ClubQuivers - Quiver plot data at key events
%
% This function orchestrates the additional processing pipeline:
%   1. Calculate work and impulse (SCRIPT_UpdateCalcsforImpulseandWork)
%   2. Calculate total work and power (SCRIPT_TotalWorkandPowerCalculation)
%   3. Add club and hand path vectors (SCRIPT_CHPandMPPCalculation)
%   4. Generate summary statistics (SCRIPT_TableofValues)
%   5. Generate ZVCF data (SCRIPT_ZVCF_GENERATOR)
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
    end

    if config.verbose
        fprintf('🔬 Running additional processing pipeline...\n');
    end

    current_dir = pwd;

    try
        %% 1. Calculate work and impulse for ZTCF and DELTA
        if config.verbose
            fprintf('\n1️⃣  Calculating work and impulse...\n');
        end
        [ZTCF, DELTA] = calculate_work_impulse(ZTCF, DELTA, config);

        % Also update Q-tables
        [ZTCFQ, DELTAQ] = calculate_work_impulse(ZTCFQ, DELTAQ, config);

        %% 2. Calculate total work and power
        if config.verbose
            fprintf('\n2️⃣  Calculating total work and power...\n');
        end
        [BASE, BASEQ, ZTCF, ZTCFQ, DELTA, DELTAQ] = ...
            calculate_total_work_power(BASE, BASEQ, ZTCF, ZTCFQ, DELTA, DELTAQ, config);

        %% 3. Add club and hand path vectors
        if config.verbose
            fprintf('\n3️⃣  Adding club and hand path vectors...\n');
        end
        % OPTIMIZATION: Replaced legacy script with vectorized function
        [BASEQ, ZTCFQ, DELTAQ] = calculate_path_vectors(BASEQ, ZTCFQ, DELTAQ);

        %% 4. Generate summary table and club quivers
        if config.verbose
            fprintf('\n4️⃣  Generating summary statistics and quiver data...\n');
        end
        cd(config.legacy_scripts_path);
        SCRIPT_TableofValues;
        cd(current_dir);

        % Package club quiver results
        ClubQuivers = struct();
        if exist('ClubQuiverAlphaReversal', 'var')
            ClubQuivers.AlphaReversal = ClubQuiverAlphaReversal;
        end
        if exist('ClubQuiverMaxCHS', 'var')
            ClubQuivers.MaxCHS = ClubQuiverMaxCHS;
        end
        if exist('ClubQuiverZTCFAlphaReversal', 'var')
            ClubQuivers.ZTCFAlphaReversal = ClubQuiverZTCFAlphaReversal;
        end
        if exist('ClubQuiverDELTAAlphaReversal', 'var')
            ClubQuivers.DELTAAlphaReversal = ClubQuiverDELTAAlphaReversal;
        end

        %% 5. Generate ZVCF data
        if config.verbose
            fprintf('\n5️⃣  Generating ZVCF (Zero Velocity Counterfactual) data...\n');
        end
        cd(config.legacy_scripts_path);
        SCRIPT_ZVCF_GENERATOR;
        cd(current_dir);

        if config.verbose
            fprintf('\n✅ Additional processing complete\n');
        end

    catch ME
        cd(current_dir);
        rethrow(ME);
    end

    cd(current_dir);

end
