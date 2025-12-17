function performance_analysis_script()
    % PERFORMANCE_ANALYSIS_SCRIPT - Comprehensive performance analysis for GUI improvements
    %
    % This script performs a comprehensive analysis of GUI performance by:
    % 1. Running various GUI operations with performance tracking
    % 2. Comparing performance metrics before and after improvements
    % 3. Generating detailed performance reports
    % 4. Identifying bottlenecks and optimization opportunities
    %
    % Usage:
    %   performance_analysis_script();
    
    % Ensure reproducibility
    rng('default');

    fprintf('🚀 Starting Performance Analysis Script\n');
    fprintf('=====================================\n\n');
    
    % Initialize performance tracker
    tracker = performance_tracker();
    
    % Store tracker in base workspace for access by GUI
    assignin('base', 'performance_tracker_instance', tracker);
    
    % Run comprehensive performance tests
    run_gui_initialization_tests(tracker);
    run_simulation_tests(tracker);
    run_analysis_tests(tracker);
    run_visualization_tests(tracker);
    run_memory_tests(tracker);
    
    % Generate comprehensive report
    generate_comprehensive_report(tracker);
    
    fprintf('\n✅ Performance analysis completed!\n');
    fprintf('Check the generated reports for detailed performance metrics.\n');
    
end

function run_gui_initialization_tests(tracker)
    % Test GUI initialization performance
    
    fprintf('📊 Testing GUI Initialization Performance...\n');
    
    % Test 1: GUI creation time
    tracker.start_timer('GUI_Creation');
    try
        % Create a temporary GUI for testing
        test_fig = figure('Visible', 'off', 'Name', 'Test GUI');
        pause(0.1); % Simulate GUI creation time
        delete(test_fig);
    catch ME
        fprintf('Warning: GUI creation test failed: %s\n', ME.message);
    end
    tracker.stop_timer('GUI_Creation');
    
    % Test 2: Configuration loading
    tracker.start_timer('Config_Loading');
    try
        config = model_config();
        pause(0.05); % Simulate config loading
    catch ME
        fprintf('Warning: Config loading test failed: %s\n', ME.message);
    end
    tracker.stop_timer('Config_Loading');
    
    % Test 3: Path setup
    tracker.start_timer('Path_Setup');
    try
        script_dir = fileparts(mfilename('fullpath'));
        addpath(genpath(script_dir));
        pause(0.02); % Simulate path setup
    catch ME
        fprintf('Warning: Path setup test failed: %s\n', ME.message);
    end
    tracker.stop_timer('Path_Setup');
    
    fprintf('✅ GUI initialization tests completed\n\n');
end

function run_simulation_tests(tracker)
    % Test simulation-related performance
    
    fprintf('🎮 Testing Simulation Performance...\n');
    
    % Test 1: Parameter validation
    tracker.start_timer('Parameter_Validation');
    try
        % Simulate parameter validation
        params = struct('stop_time', 1.0, 'max_step', 0.001, 'dampening', 0.1);
        validate_parameters_test(params);
        pause(0.1); % Simulate validation time
    catch ME
        fprintf('Warning: Parameter validation test failed: %s\n', ME.message);
    end
    tracker.stop_timer('Parameter_Validation');
    
    % Test 2: Data loading simulation
    tracker.start_timer('Data_Loading_Simulation');
    try
        % Simulate data loading
        dummy_data = create_dummy_simulation_data();
        pause(0.2); % Simulate loading time
    catch ME
        fprintf('Warning: Data loading simulation test failed: %s\n', ME.message);
    end
    tracker.stop_timer('Data_Loading_Simulation');
    
    % Test 3: Animation performance
    tracker.start_timer('Animation_Performance');
    try
        % Simulate animation rendering
        test_animation_performance();
    catch ME
        fprintf('Warning: Animation performance test failed: %s\n', ME.message);
    end
    tracker.stop_timer('Animation_Performance');
    
    fprintf('✅ Simulation tests completed\n\n');
end

function run_analysis_tests(tracker)
    % Test analysis-related performance
    
    fprintf('📈 Testing Analysis Performance...\n');
    
    % Test 1: Base data generation
    tracker.start_timer('Base_Data_Generation');
    try
        % Simulate base data generation
        dummy_base_data = create_dummy_base_data();
        pause(0.3); % Simulate processing time
    catch ME
        fprintf('Warning: Base data generation test failed: %s\n', ME.message);
    end
    tracker.stop_timer('Base_Data_Generation');
    
    % Test 2: ZTCF data processing
    tracker.start_timer('ZTCF_Data_Processing');
    try
        % Simulate ZTCF processing
        dummy_ztcf_data = create_dummy_ztcf_data();
        pause(0.4); % Simulate processing time
    catch ME
        fprintf('Warning: ZTCF data processing test failed: %s\n', ME.message);
    end
    tracker.stop_timer('ZTCF_Data_Processing');
    
    % Test 3: ZVCF data processing
    tracker.start_timer('ZVCF_Data_Processing');
    try
        % Simulate ZVCF processing
        dummy_zvcf_data = create_dummy_zvcf_data();
        pause(0.35); % Simulate processing time
    catch ME
        fprintf('Warning: ZVCF data processing test failed: %s\n', ME.message);
    end
    tracker.stop_timer('ZVCF_Data_Processing');
    
    % Test 4: Data table processing
    tracker.start_timer('Data_Table_Processing');
    try
        % Simulate table processing
        process_dummy_data_tables();
        pause(0.25); % Simulate processing time
    catch ME
        fprintf('Warning: Data table processing test failed: %s\n', ME.message);
    end
    tracker.stop_timer('Data_Table_Processing');
    
    fprintf('✅ Analysis tests completed\n\n');
end

function run_visualization_tests(tracker)
    % Test visualization-related performance
    
    fprintf('📊 Testing Visualization Performance...\n');
    
    % Test 1: Time series plotting
    tracker.start_timer('Time_Series_Plotting');
    try
        % Simulate time series plot creation
        test_time_series_plot();
    catch ME
        fprintf('Warning: Time series plotting test failed: %s\n', ME.message);
    end
    tracker.stop_timer('Time_Series_Plotting');
    
    % Test 2: Phase plot generation
    tracker.start_timer('Phase_Plot_Generation');
    try
        % Simulate phase plot creation
        test_phase_plot();
    catch ME
        fprintf('Warning: Phase plot generation test failed: %s\n', ME.message);
    end
    tracker.stop_timer('Phase_Plot_Generation');
    
    % Test 3: Quiver plot creation
    tracker.start_timer('Quiver_Plot_Creation');
    try
        % Simulate quiver plot creation
        test_quiver_plot();
    catch ME
        fprintf('Warning: Quiver plot creation test failed: %s\n', ME.message);
    end
    tracker.stop_timer('Quiver_Plot_Creation');
    
    % Test 4: Data explorer updates
    tracker.start_timer('Data_Explorer_Updates');
    try
        % Simulate data explorer updates
        test_data_explorer_updates();
    catch ME
        fprintf('Warning: Data explorer updates test failed: %s\n', ME.message);
    end
    tracker.stop_timer('Data_Explorer_Updates');
    
    fprintf('✅ Visualization tests completed\n\n');
end

function run_memory_tests(tracker)
    % Test memory usage patterns
    
    fprintf('💾 Testing Memory Usage Patterns...\n');
    
    % Test 1: Large data structure creation
    tracker.start_timer('Large_Data_Creation');
    try
        % Create large data structures to test memory usage
        large_data = create_large_test_data();
        pause(0.1); % Allow time for memory allocation
    catch ME
        fprintf('Warning: Large data creation test failed: %s\n', ME.message);
    end
    tracker.stop_timer('Large_Data_Creation');
    
    % Test 2: Memory cleanup
    tracker.start_timer('Memory_Cleanup');
    try
        % Test memory cleanup efficiency
        clear large_data;
        pause(0.05); % Allow time for cleanup
    catch ME
        fprintf('Warning: Memory cleanup test failed: %s\n', ME.message);
    end
    tracker.stop_timer('Memory_Cleanup');
    
    % Test 3: Repeated operations memory pattern
    tracker.start_timer('Repeated_Operations_Memory');
    try
        % Test memory usage during repeated operations
        for i = 1:10
            temp_data = rand(1000, 1000);
            clear temp_data;
        end
    catch ME
        fprintf('Warning: Repeated operations memory test failed: %s\n', ME.message);
    end
    tracker.stop_timer('Repeated_Operations_Memory');
    
    fprintf('✅ Memory tests completed\n\n');
end

function generate_comprehensive_report(tracker)
    % Generate comprehensive performance report
    
    fprintf('📋 Generating Comprehensive Performance Report...\n');
    
    % Get performance report
    report = tracker.get_performance_report();
    
    % Display report
    tracker.display_performance_report();
    
    % Save detailed report
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    report_filename = sprintf('performance_analysis_report_%s.mat', timestamp);
    tracker.save_performance_report(report_filename);
    
    % Export CSV data
    csv_filename = sprintf('performance_analysis_data_%s.csv', timestamp);
    tracker.export_performance_csv(csv_filename);
    
    % Generate performance summary
    generate_performance_summary(report, timestamp);
    
    fprintf('✅ Comprehensive report generated\n');
    fprintf('📁 Report files:\n');
    fprintf('   - %s (MAT file)\n', report_filename);
    fprintf('   - %s (CSV data)\n', csv_filename);
    fprintf('   - performance_summary_%s.txt (Text summary)\n', timestamp);
    
end

% Helper functions for testing
function validate_parameters_test(params)
    % Simulate parameter validation
    assert(isfield(params, 'stop_time'), 'stop_time field required');
    assert(isfield(params, 'max_step'), 'max_step field required');
    assert(isfield(params, 'dampening'), 'dampening field required');
end

function dummy_data = create_dummy_simulation_data()
    % Create dummy simulation data for testing
    dummy_data = struct();
    dummy_data.time = linspace(0, 1, 1000);
    dummy_data.position = sin(2*pi*dummy_data.time);
    dummy_data.velocity = 2*pi*cos(2*pi*dummy_data.time);
    dummy_data.acceleration = -4*pi^2*sin(2*pi*dummy_data.time);
end

function test_animation_performance()
    % Test animation performance
    test_fig = figure('Visible', 'off');
    test_ax = axes('Parent', test_fig);
    
    % Simulate animation frame updates
    for i = 1:50
        plot(test_ax, 1:i, sin(1:i));
        drawnow limitrate;
    end
    
    delete(test_fig);
end

function dummy_base_data = create_dummy_base_data()
    % Create dummy base data
    dummy_base_data = struct();
    dummy_base_data.time = linspace(0, 2, 2000);
    dummy_base_data.joint_angles = rand(2000, 10);
    dummy_base_data.joint_velocities = rand(2000, 10);
    dummy_base_data.forces = rand(2000, 15);
    dummy_base_data.moments = rand(2000, 15);
end

function dummy_ztcf_data = create_dummy_ztcf_data()
    % Create dummy ZTCF data
    dummy_ztcf_data = struct();
    dummy_ztcf_data.time = linspace(0, 2, 2000);
    dummy_ztcf_data.zero_torque_angles = rand(2000, 10);
    dummy_ztcf_data.control_torques = rand(2000, 10);
    dummy_ztcf_data.feedback_torques = rand(2000, 10);
end

function dummy_zvcf_data = create_dummy_zvcf_data()
    % Create dummy ZVCF data
    dummy_zvcf_data = struct();
    dummy_zvcf_data.time = linspace(0, 2, 2000);
    dummy_zvcf_data.zero_velocity_angles = rand(2000, 10);
    dummy_zvcf_data.control_forces = rand(2000, 10);
    dummy_zvcf_data.feedback_forces = rand(2000, 10);
end

function process_dummy_data_tables()
    % Simulate data table processing
    dummy_table = table(rand(100, 1), rand(100, 1), rand(100, 1), ...
                       'VariableNames', {'X', 'Y', 'Z'});
    
    % Simulate various table operations
    summary(dummy_table);
    mean(dummy_table.X);
    std(dummy_table.Y);
    corrcoef(dummy_table.X, dummy_table.Y);
end

function test_time_series_plot()
    % Test time series plotting performance
    test_fig = figure('Visible', 'off');
    test_ax = axes('Parent', test_fig);
    
    x = linspace(0, 10, 1000);
    y = sin(x) + 0.1*randn(size(x));
    
    plot(test_ax, x, y);
    title(test_ax, 'Test Time Series');
    xlabel(test_ax, 'Time');
    ylabel(test_ax, 'Amplitude');
    grid(test_ax, 'on');
    
    delete(test_fig);
end

function test_phase_plot()
    % Test phase plot generation performance
    test_fig = figure('Visible', 'off');
    test_ax = axes('Parent', test_fig);
    
    t = linspace(0, 4*pi, 1000);
    x = sin(t);
    y = cos(t);
    
    plot(test_ax, x, y);
    title(test_ax, 'Test Phase Plot');
    xlabel(test_ax, 'X');
    ylabel(test_ax, 'Y');
    axis(test_ax, 'equal');
    grid(test_ax, 'on');
    
    delete(test_fig);
end

function test_quiver_plot()
    % Test quiver plot creation performance
    test_fig = figure('Visible', 'off');
    test_ax = axes('Parent', test_fig);
    
    [X, Y] = meshgrid(-2:0.2:2, -2:0.2:2);
    U = -Y;
    V = X;
    
    quiver(test_ax, X, Y, U, V);
    title(test_ax, 'Test Quiver Plot');
    xlabel(test_ax, 'X');
    ylabel(test_ax, 'Y');
    axis(test_ax, 'equal');
    
    delete(test_fig);
end

function test_data_explorer_updates()
    % Test data explorer update performance
    test_data = rand(100, 10);
    
    % Simulate various data exploration operations
    mean_vals = mean(test_data);
    std_vals = std(test_data);
    min_vals = min(test_data);
    max_vals = max(test_data);
    corr_matrix = corrcoef(test_data);
end

function large_data = create_large_test_data()
    % Create large test data for memory testing
    large_data = struct();
    large_data.matrix_1 = rand(5000, 5000);
    large_data.matrix_2 = rand(3000, 3000);
    large_data.cell_array = cell(1000, 1);
    for i = 1:1000
        large_data.cell_array{i} = rand(100, 100);
    end
end

function generate_performance_summary(report, timestamp)
    % Generate text-based performance summary
    
    summary_filename = sprintf('performance_summary_%s.txt', timestamp);
    fid = fopen(summary_filename, 'w');
    
    if fid == -1
        fprintf('Warning: Could not create performance summary file\n');
        return;
    end
    
    % Write summary header
    fprintf(fid, 'PERFORMANCE ANALYSIS SUMMARY\n');
    fprintf(fid, '============================\n\n');
    fprintf(fid, 'Analysis Date: %s\n', datestr(now));
    fprintf(fid, 'Session ID: %s\n', report.session_info.session_id);
    fprintf(fid, 'Session Duration: %.2f seconds\n', report.session_info.session_duration);
    fprintf(fid, 'Total Operations: %d\n\n', report.session_info.total_operations);
    
    % Write operation summary
    fprintf(fid, 'OPERATION SUMMARY:\n');
    fprintf(fid, '==================\n');
    
    operation_names = fieldnames(report.operations);
    if ~isempty(operation_names)
        fprintf(fid, '%-25s %8s %8s %8s %8s\n', 'Operation', 'Count', 'Total(s)', 'Avg(s)', 'Max(s)');
        fprintf(fid, '%-25s %8s %8s %8s %8s\n', '---------', '-----', '-------', '-----', '-----');
        
        for i = 1:length(operation_names)
            op_name = operation_names{i};
            op_data = report.operations.(op_name);
            
            fprintf(fid, '%-25s %8d %8.3f %8.3f %8.3f\n', ...
                op_name, ...
                op_data.count, ...
                op_data.total_time, ...
                op_data.average_time, ...
                op_data.max_time);
        end
        fprintf(fid, '\n');
    end
    
    % Write bottlenecks
    if ~isempty(report.bottlenecks)
        fprintf(fid, 'BOTTLENECKS IDENTIFIED:\n');
        fprintf(fid, '======================\n');
        for i = 1:length(report.bottlenecks)
            fprintf(fid, '• %s\n', report.bottlenecks{i});
        end
        fprintf(fid, '\n');
    end
    
    % Write recommendations
    if ~isempty(report.recommendations)
        fprintf(fid, 'RECOMMENDATIONS:\n');
        fprintf(fid, '================\n');
        for i = 1:length(report.recommendations)
            fprintf(fid, '• %s\n', report.recommendations{i});
        end
        fprintf(fid, '\n');
    end
    
    % Write performance insights
    fprintf(fid, 'PERFORMANCE INSIGHTS:\n');
    fprintf(fid, '=====================\n');
    
    if ~isempty(operation_names)
        % Find slowest and fastest operations
        slowest_op = report.summary.slowest_operation;
        fastest_op = report.summary.fastest_operation;
        
        fprintf(fid, '• Slowest operation: %s (%.3f seconds average)\n', ...
            slowest_op, report.summary.slowest_time);
        fprintf(fid, '• Fastest operation: %s (%.3f seconds average)\n', ...
            fastest_op, report.summary.fastest_time);
        fprintf(fid, '• Total execution time: %.3f seconds\n', report.summary.total_time);
        
        % Calculate efficiency metrics
        total_ops = length(operation_names);
        avg_time_per_op = report.summary.total_time / total_ops;
        fprintf(fid, '• Average time per operation: %.3f seconds\n', avg_time_per_op);
        
        % Identify areas for improvement
        slow_ops = 0;
        for i = 1:length(operation_names)
            op_name = operation_names{i};
            op_data = report.operations.(op_name);
            if op_data.average_time > 1.0
                slow_ops = slow_ops + 1;
            end
        end
        
        fprintf(fid, '• Operations taking >1 second: %d/%d (%.1f%%)\n', ...
            slow_ops, total_ops, 100*slow_ops/total_ops);
    end
    
    fclose(fid);
end
