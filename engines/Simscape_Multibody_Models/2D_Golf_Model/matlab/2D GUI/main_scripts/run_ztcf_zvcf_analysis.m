function [BASE, ZTCF, DELTA, ZVCFTable] = run_ztcf_zvcf_analysis()
% RUN_ZTCF_ZVCF_ANALYSIS - Main function to run complete ZTCF/ZVCF analysis
%
% Returns:
%   BASE, ZTCF, DELTA - Main data tables
%   ZVCFTable - ZVCF data table
%
% This function orchestrates the complete analysis pipeline:
%   1. Load configuration
%   2. Initialize model
%   3. Generate base data
%   4. Generate ZTCF data
%   5. Process data tables
%   6. Run additional processing
%   7. Save results
%
% Usage:
%   [BASE, ZTCF, DELTA, ZVCFTable] = run_ztcf_zvcf_analysis();

    fprintf('🚀 Starting ZTCF/ZVCF Analysis Pipeline\n');
    fprintf('=====================================\n\n');

    try
        % 1. Load configuration
        fprintf('📋 Loading configuration...\n');
        config = model_config();

        % 2. Initialize model
        fprintf('🔧 Initializing model...\n');
        mdlWks = initialize_model(config);

        % 3. Generate base data
        fprintf('📊 Generating base data...\n');
        BaseData = generate_base_data(config, mdlWks);

        % 4. Generate ZTCF data
        fprintf('🔄 Generating ZTCF data...\n');
        ZTCF = generate_ztcf_data(config, mdlWks, BaseData);

        % 5. Process data tables
        fprintf('⚙️  Processing data tables...\n');
        [BASE, ZTCF, DELTA, BASEQ, ZTCFQ, DELTAQ] = process_data_tables(config, BaseData, ZTCF);

        % 6. Run additional processing
        fprintf('🔬 Running additional processing...\n');
        [BASE, ZTCF, DELTA, BASEQ, ZTCFQ, DELTAQ, ZVCFTable, ZVCFTableQ] = ...
            run_additional_processing(config, BASE, ZTCF, DELTA, BASEQ, ZTCFQ, DELTAQ);

        % 7. Save results
        fprintf('💾 Saving results...\n');
        save_data_tables(config, BASE, ZTCF, DELTA, BASEQ, ZTCFQ, DELTAQ, ZVCFTable, ZVCFTableQ);

        % 8. Return to model directory
        cd(config.model_path);

        fprintf('\n🎉 ZTCF/ZVCF Analysis completed successfully!\n');
        fprintf('📈 Results saved to: %s\n', config.tables_path);

    catch ME
        fprintf('\n❌ Error during analysis: %s\n', ME.message);
        fprintf('📍 Error location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        rethrow(ME);
    end

end
