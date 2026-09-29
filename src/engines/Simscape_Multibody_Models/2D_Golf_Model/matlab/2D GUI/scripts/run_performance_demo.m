function run_performance_demo()
    % RUN_PERFORMANCE_DEMO - Demonstration of performance tracking features
    %
    % This script demonstrates how to use the performance tracking system
    % to evaluate GUI improvements and identify optimization opportunities.
    %
    % Usage:
    %   run_performance_demo();
    
    % Ensure reproducibility
    rng('default');

    fprintf('🎯 Performance Tracking Demo\n');
    fprintf('============================\n\n');
    
    % Initialize performance tracker
    tracker = performance_tracker();
    
    fprintf('1. Running Performance Analysis Script...\n');
    fprintf('   This will test various GUI operations and generate reports.\n\n');
    
    % Run the comprehensive performance analysis
    performance_analysis_script();
    
    fprintf('\n2. Launching GUI with Performance Monitoring...\n');
    fprintf('   The GUI now includes a Performance Monitor tab.\n');
    fprintf('   Use it to track real-time performance during your workflow.\n\n');
    
    % Launch the GUI
    launch_gui();
    
    fprintf('\n3. Performance Tracking Features Available:\n');
    fprintf('   • Real-time performance metrics\n');
    fprintf('   • Memory usage monitoring\n');
    fprintf('   • Operation timing analysis\n');
    fprintf('   • Bottleneck identification\n');
    fprintf('   • Performance reports and exports\n');
    fprintf('   • Historical performance data\n\n');
    
    fprintf('4. How to Use Performance Tracking:\n');
    fprintf('   • Switch to the "🔍 Performance Monitor" tab\n');
    fprintf('   • Enable tracking to start monitoring\n');
    fprintf('   • Run your usual GUI operations\n');
    fprintf('   • View real-time charts and metrics\n');
    fprintf('   • Generate reports to analyze performance\n');
    fprintf('   • Export data for further analysis\n\n');
    
    fprintf('5. Performance Analysis Workflow:\n');
    fprintf('   a) Run operations with tracking enabled\n');
    fprintf('   b) Identify slow operations (>1 second)\n');
    fprintf('   c) Look for high memory usage patterns\n');
    fprintf('   d) Generate optimization recommendations\n');
    fprintf('   e) Implement improvements\n');
    fprintf('   f) Re-run analysis to measure improvements\n\n');
    
    fprintf('✅ Demo completed! Check the generated reports and GUI.\n');
    
end

function quick_performance_test()
    % QUICK_PERFORMANCE_TEST - Quick test of performance tracking
    
    fprintf('⚡ Quick Performance Test\n');
    fprintf('=========================\n\n');
    
    % Initialize tracker
    tracker = performance_tracker();
    
    % Test various operations
    operations = {
        'Data_Loading', @() pause(0.1);
        'Data_Processing', @() pause(0.2);
        'Plot_Generation', @() pause(0.15);
        'File_Export', @() pause(0.05);
        'Memory_Allocation', @() rand(1000, 1000);
    };
    
    for i = 1:size(operations, 1)
        op_name = operations{i, 1};
        op_func = operations{i, 2};
        
        fprintf('Testing: %s\n', op_name);
        tracker.start_timer(op_name);
        op_func();
        tracker.stop_timer(op_name);
    end
    
    % Display results
    fprintf('\n📊 Quick Test Results:\n');
    tracker.display_performance_report();
    
    % Save quick test report
    tracker.save_performance_report('quick_performance_test.mat');
    tracker.export_performance_csv('quick_performance_test.csv');
    
    fprintf('\n✅ Quick test completed! Check the generated files.\n');
end
