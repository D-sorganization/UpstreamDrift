function [BASE, ZTCF, DELTA, BASEQ, ZTCFQ, DELTAQ, ZVCFTable, ZVCFTableQ] = run_additional_processing(config, BASE, ZTCF, DELTA, BASEQ, ZTCFQ, DELTAQ)
% RUN_ADDITIONAL_PROCESSING - Run all additional processing scripts
%
% Inputs:
%   config - Configuration structure from model_config()
%   BASE, ZTCF, DELTA, BASEQ, ZTCFQ, DELTAQ - Data tables
%
% Returns:
%   Updated data tables with additional processing
%   ZVCFTable, ZVCFTableQ - ZVCF data tables
%
% This function runs all the additional processing scripts in the correct order

    % Change to scripts directory
    cd(config.scripts_path);

    fprintf('🔄 Running additional processing scripts...\n');

    % 1. Update calculations for impulse and work
    fprintf('   1. Updating impulse and work calculations...\n');
    SCRIPT_UpdateCalcsforImpulseandWork;

    % 2. Q table time change
    fprintf('   2. Processing Q table time changes...\n');
    SCRIPT_QTableTimeChange;

    % 3. Total work and power calculations
    fprintf('   3. Calculating total work and power...\n');
    SCRIPT_TotalWorkandPowerCalculation;

    % 4. Club and hand path calculations
    fprintf('   4. Calculating club and hand paths...\n');
    SCRIPT_CHPandMPPCalculation;

    % 5. Table of values generation
    fprintf('   5. Generating table of values...\n');
    SCRIPT_TableofValues;

    % 6. Generate ZVCF data
    fprintf('   6. Generating ZVCF data...\n');
    SCRIPT_ZVCF_GENERATOR;

    % 7. Generate all plots
    fprintf('   7. Generating plots...\n');
    SCRIPT_AllPlots;

    fprintf('✅ Additional processing completed successfully\n');

end
