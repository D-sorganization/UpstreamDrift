classdef performance_tracker < handle
    % PERFORMANCE_TRACKER - Comprehensive performance monitoring for GUI operations
    %
    % This class provides detailed performance tracking including:
    % - Function execution times
    % - Memory usage monitoring
    % - CPU utilization
    % - Performance bottlenecks identification
    % - Historical performance data
    %
    % Usage:
    %   tracker = performance_tracker();
    %   tracker.start_timer('operation_name');
    %   % ... perform operation ...
    %   tracker.stop_timer('operation_name');
    %   results = tracker.get_performance_report();

    properties (Access = private)
        timers = containers.Map('KeyType', 'char', 'ValueType', 'any');
        memory_snapshots = containers.Map('KeyType', 'char', 'ValueType', 'any');
        cpu_usage = containers.Map('KeyType', 'char', 'ValueType', 'any');
        performance_history = struct();
        start_times = containers.Map('KeyType', 'char', 'ValueType', 'any');
        operation_counts = containers.Map('KeyType', 'char', 'ValueType', 'double');
        total_times = containers.Map('KeyType', 'char', 'ValueType', 'double');
        min_times = containers.Map('KeyType', 'char', 'ValueType', 'double');
        max_times = containers.Map('KeyType', 'char', 'ValueType', 'double');
        is_enabled = true;
        session_start_time = [];
        session_id = '';
    end

    methods
        function obj = performance_tracker()
            % Constructor - Initialize the performance tracker
            obj.session_start_time = tic;
            obj.session_id = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

            % Initialize containers
            obj.timers = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.memory_snapshots = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.cpu_usage = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.start_times = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.operation_counts = containers.Map('KeyType', 'char', 'ValueType', 'double');
            obj.total_times = containers.Map('KeyType', 'char', 'ValueType', 'double');
            obj.min_times = containers.Map('KeyType', 'char', 'ValueType', 'double');
            obj.max_times = containers.Map('KeyType', 'char', 'ValueType', 'double');

            fprintf('🔍 Performance tracker initialized (Session: %s)\n', obj.session_id);
        end

        function start_timer(obj, operation_name)
            % Start timing an operation
            if ~obj.is_enabled
                return;
            end

            % Record start time
            obj.start_times(operation_name) = tic;

            % Take memory snapshot
            obj.memory_snapshots(operation_name) = obj.get_memory_usage();

            % Record CPU usage
            obj.cpu_usage(operation_name) = obj.get_cpu_usage();

            fprintf('⏱️  Started timing: %s\n', operation_name);
        end

        function stop_timer(obj, operation_name)
            % Stop timing an operation and record results
            if ~obj.is_enabled || ~obj.start_times.isKey(operation_name)
                return;
            end

            % Calculate elapsed time
            elapsed_time = toc(obj.start_times(operation_name));

            % Update statistics
            if obj.operation_counts.isKey(operation_name)
                obj.operation_counts(operation_name) = obj.operation_counts(operation_name) + 1;
                obj.total_times(operation_name) = obj.total_times(operation_name) + elapsed_time;
                obj.min_times(operation_name) = min(obj.min_times(operation_name), elapsed_time);
                obj.max_times(operation_name) = max(obj.max_times(operation_name), elapsed_time);
            else
                obj.operation_counts(operation_name) = 1;
                obj.total_times(operation_name) = elapsed_time;
                obj.min_times(operation_name) = elapsed_time;
                obj.max_times(operation_name) = elapsed_time;
            end

            % Record final memory usage
            final_memory = obj.get_memory_usage();
            memory_delta = final_memory - obj.memory_snapshots(operation_name);

            % Store detailed results
            obj.timers(operation_name) = struct(...
                'elapsed_time', elapsed_time, ...
                'memory_start', obj.memory_snapshots(operation_name), ...
                'memory_end', final_memory, ...
                'memory_delta', memory_delta, ...
                'cpu_start', obj.cpu_usage(operation_name), ...
                'timestamp', now() ...
            );

            fprintf('⏱️  Completed: %s (%.3f seconds, Memory: %.2f MB)\n', ...
                operation_name, elapsed_time, memory_delta / 1024 / 1024);
        end

        function time_function(obj, operation_name, func_handle, varargin)
            % Time a function execution with automatic start/stop
            if ~obj.is_enabled
                result = func_handle(varargin{:});
                return;
            end

            obj.start_timer(operation_name);
            try
                result = func_handle(varargin{:});
                obj.stop_timer(operation_name);
            catch ME
                obj.stop_timer(operation_name);
                rethrow(ME);
            end
        end

        function report = get_performance_report(obj)
            % Generate comprehensive performance report
            if ~obj.is_enabled
                report = struct('message', 'Performance tracking is disabled');
                return;
            end

            session_duration = toc(obj.session_start_time);

            % Collect all operation names
            operation_names = obj.timers.keys();

            % Build detailed report
            report = struct();
            report.session_info = struct(...
                'session_id', obj.session_id, ...
                'session_duration', session_duration, ...
                'total_operations', length(operation_names), ...
                'timestamp', now() ...
            );

            % Operation details
            report.operations = struct();
            for i = 1:length(operation_names)
                op_name = operation_names{i};
                if obj.timers.isKey(op_name)
                    timer_data = obj.timers(op_name);

                    % Calculate statistics
                    count = obj.operation_counts(op_name);
                    total_time = obj.total_times(op_name);
                    avg_time = total_time / count;
                    min_time = obj.min_times(op_name);
                    max_time = obj.max_times(op_name);

                    report.operations.(op_name) = struct(...
                        'count', count, ...
                        'total_time', total_time, ...
                        'average_time', avg_time, ...
                        'min_time', min_time, ...
                        'max_time', max_time, ...
                        'last_execution', timer_data.elapsed_time, ...
                        'memory_delta', timer_data.memory_delta, ...
                        'cpu_usage', timer_data.cpu_start, ...
                        'timestamp', timer_data.timestamp ...
                    );
                end
            end

            % Performance summary
            report.summary = obj.generate_summary(report);

            % Bottleneck analysis
            report.bottlenecks = obj.identify_bottlenecks(report);

            % Recommendations
            report.recommendations = obj.generate_recommendations(report);
        end

        function display_performance_report(obj)
            % Display formatted performance report
            report = obj.get_performance_report();

            fprintf('\n📊 PERFORMANCE REPORT\n');
            fprintf('=====================================\n');
            fprintf('Session ID: %s\n', report.session_info.session_id);
            fprintf('Session Duration: %.2f seconds\n', report.session_info.session_duration);
            fprintf('Total Operations: %d\n', report.session_info.total_operations);
            fprintf('\n');

            % Display operation details
            operation_names = fieldnames(report.operations);
            if ~isempty(operation_names)
                fprintf('OPERATION DETAILS:\n');
                fprintf('%-30s %8s %8s %8s %8s %8s\n', ...
                    'Operation', 'Count', 'Total(s)', 'Avg(s)', 'Min(s)', 'Max(s)');
                fprintf('%-30s %8s %8s %8s %8s %8s\n', ...
                    '---------', '-----', '-------', '-----', '-----', '-----');

                for i = 1:length(operation_names)
                    op_name = operation_names{i};
                    op_data = report.operations.(op_name);

                    fprintf('%-30s %8d %8.3f %8.3f %8.3f %8.3f\n', ...
                        op_name, ...
                        op_data.count, ...
                        op_data.total_time, ...
                        op_data.average_time, ...
                        op_data.min_time, ...
                        op_data.max_time);
                end
                fprintf('\n');
            end

            % Display bottlenecks
            if ~isempty(report.bottlenecks)
                fprintf('BOTTLENECKS IDENTIFIED:\n');
                for i = 1:length(report.bottlenecks)
                    fprintf('  • %s\n', report.bottlenecks{i});
                end
                fprintf('\n');
            end

            % Display recommendations
            if ~isempty(report.recommendations)
                fprintf('RECOMMENDATIONS:\n');
                for i = 1:length(report.recommendations)
                    fprintf('  • %s\n', report.recommendations{i});
                end
                fprintf('\n');
            end
        end

        function save_performance_report(obj, filename)
            % Save performance report to file
            if nargin < 2
                filename = sprintf('performance_report_%s.mat', obj.session_id);
            end

            report = obj.get_performance_report();
            save(filename, 'report');
            fprintf('💾 Performance report saved to: %s\n', filename);
        end

        function export_performance_csv(obj, filename)
            % Export performance data to CSV format
            if nargin < 2
                filename = sprintf('performance_data_%s.csv', obj.session_id);
            end

            report = obj.get_performance_report();
            operation_names = fieldnames(report.operations);

            % Create table for CSV export
            data_table = table();
            for i = 1:length(operation_names)
                op_name = operation_names{i};
                op_data = report.operations.(op_name);

                row = table({op_name}, op_data.count, op_data.total_time, ...
                    op_data.average_time, op_data.min_time, op_data.max_time, ...
                    op_data.memory_delta, op_data.cpu_usage, ...
                    'VariableNames', {'Operation', 'Count', 'TotalTime', ...
                    'AverageTime', 'MinTime', 'MaxTime', 'MemoryDelta', 'CPUUsage'});

                data_table = [data_table; row];
            end

            writetable(data_table, filename);
            fprintf('📊 Performance data exported to: %s\n', filename);
        end

        function enable_tracking(obj)
            % Enable performance tracking
            obj.is_enabled = true;
            fprintf('🔍 Performance tracking enabled\n');
        end

        function disable_tracking(obj)
            % Disable performance tracking
            obj.is_enabled = false;
            fprintf('🔍 Performance tracking disabled\n');
        end

        function clear_history(obj)
            % Clear all performance history
            obj.timers = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.memory_snapshots = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.cpu_usage = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.start_times = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.operation_counts = containers.Map('KeyType', 'char', 'ValueType', 'double');
            obj.total_times = containers.Map('KeyType', 'char', 'ValueType', 'double');
            obj.min_times = containers.Map('KeyType', 'char', 'ValueType', 'double');
            obj.max_times = containers.Map('KeyType', 'char', 'ValueType', 'double');

            fprintf('🗑️  Performance history cleared\n');
        end
    end

    methods (Access = private)
        function memory_usage = get_memory_usage(obj)
            % Get current memory usage in bytes
            try
                [~, systemview] = memory;
                memory_usage = systemview.PhysicalMemory.Total - systemview.PhysicalMemory.Available;
            catch
                memory_usage = 0; % Fallback if memory function fails
            end
        end

        function cpu_usage = get_cpu_usage(obj)
            % Get current CPU usage (simplified implementation)
            try
                % This is a simplified CPU measurement
                % In a real implementation, you might use system calls
                cpu_usage = 0; % Placeholder
            catch
                cpu_usage = 0;
            end
        end

        function summary = generate_summary(obj, report)
            % Generate performance summary
            operation_names = fieldnames(report.operations);

            if isempty(operation_names)
                summary = struct('total_time', 0, 'slowest_operation', '', 'fastest_operation', '');
                return;
            end

            total_time = 0;
            slowest_op = '';
            fastest_op = '';
            slowest_time = 0;
            fastest_time = inf;

            for i = 1:length(operation_names)
                op_name = operation_names{i};
                op_data = report.operations.(op_name);

                total_time = total_time + op_data.total_time;

                if op_data.average_time > slowest_time
                    slowest_time = op_data.average_time;
                    slowest_op = op_name;
                end

                if op_data.average_time < fastest_time
                    fastest_time = op_data.average_time;
                    fastest_op = op_name;
                end
            end

            summary = struct(...
                'total_time', total_time, ...
                'slowest_operation', slowest_op, ...
                'fastest_operation', fastest_op, ...
                'slowest_time', slowest_time, ...
                'fastest_time', fastest_time ...
            );
        end

        function bottlenecks = identify_bottlenecks(obj, report)
            % Identify performance bottlenecks
            operation_names = fieldnames(report.operations);

            if isempty(operation_names)
                bottlenecks = {};
                return;
            end

            % Pre-allocate for performance (estimate max possible bottlenecks)
            bottlenecks_temp = cell(length(operation_names) * 2, 1);
            count = 0;

            % Find operations taking more than 1 second on average
            for i = 1:length(operation_names)
                op_name = operation_names{i};
                op_data = report.operations.(op_name);

                if op_data.average_time > 1.0
                    count = count + 1;
                    bottlenecks_temp{count} = sprintf('%s: %.3f seconds average', op_name, op_data.average_time);
                end
            end

            % Find operations with high memory usage
            for i = 1:length(operation_names)
                op_name = operation_names{i};
                op_data = report.operations.(op_name);

                if op_data.memory_delta > 100 * 1024 * 1024 % 100 MB
                    count = count + 1;
                    bottlenecks_temp{count} = sprintf('%s: High memory usage (%.2f MB)', ...
                        op_name, op_data.memory_delta / 1024 / 1024);
                end
            end

            % Trim to actual size
            bottlenecks = bottlenecks_temp(1:count);
        end

        function recommendations = generate_recommendations(obj, report)
            % Generate performance improvement recommendations
            operation_names = fieldnames(report.operations);

            if isempty(operation_names)
                recommendations = {'No operations tracked yet'};
                return;
            end

            % Pre-allocate for performance
            recommendations_temp = cell(length(operation_names) * 3, 1);
            count = 0;

            % Check for slow operations
            for i = 1:length(operation_names)
                op_name = operation_names{i};
                op_data = report.operations.(op_name);

                if op_data.average_time > 5.0
                    count = count + 1;
                    recommendations_temp{count} = sprintf('Consider optimizing %s (%.3f seconds average)', ...
                        op_name, op_data.average_time);
                end

                if op_data.count > 10 && op_data.average_time > 1.0
                    count = count + 1;
                    recommendations_temp{count} = sprintf('Cache results for frequently called %s', op_name);
                end
            end

            % Check for memory issues
            total_memory = 0;
            for i = 1:length(operation_names)
                op_name = operation_names{i};
                op_data = report.operations.(op_name);
                total_memory = total_memory + op_data.memory_delta;
            end

            if total_memory > 500 * 1024 * 1024 % 500 MB
                count = count + 1;
                recommendations_temp{count} = 'Consider implementing memory cleanup for large operations';
            end

            if count == 0
                recommendations = {'Performance looks good! No major issues identified.'};
            else
                % Trim to actual size
                recommendations = recommendations_temp(1:count);
            end
        end
    end
end
