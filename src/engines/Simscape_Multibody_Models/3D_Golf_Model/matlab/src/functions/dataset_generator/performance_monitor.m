function performance_monitor(varargin)
    % PERFORMANCE_MONITOR - Monitor and analyze performance metrics
    %
    % Usage:
    %   performance_monitor()                    % Display current performance status
    %   performance_monitor('start')             % Start monitoring session
    %   performance_monitor('stop')              % Stop monitoring and show report
    %   performance_monitor('analyze', data)     % Analyze performance data
    %   performance_monitor('bottlenecks')       % Identify performance bottlenecks

    persistent monitor_data;

    if nargin == 0
        displayPerformanceStatus();
        return;
    end

    action = varargin{1};

    switch lower(action)
        case 'start'
            startMonitoring();
        case 'stop'
            stopMonitoring();
        case 'analyze'
            if nargin > 1
                analyzePerformance(varargin{2});
            else
                error('Please provide performance data to analyze');
            end
        case 'bottlenecks'
            identifyBottlenecks();
        case 'reset'
            monitor_data = [];
            fprintf('Performance monitor reset\n');
        otherwise
            error('Unknown action: %s', action);
    end
end

function startMonitoring()
    % Start a new performance monitoring session
    global performance_data;

    performance_data = struct();
    performance_data.start_time = tic;
    performance_data.start_memory = getMemoryUsage();
    performance_data.phases = {};
    performance_data.current_phase = '';
    performance_data.phase_start = [];
    performance_data.phase_times = {};
    performance_data.memory_usage = [];
    performance_data.trial_times = [];
    performance_data.batch_times = [];
    performance_data.simulation_times = [];
    performance_data.processing_times = [];
    performance_data.checkpoint_times = [];

    fprintf('Performance monitoring started\n');
end

function stopMonitoring()
    % Stop monitoring and generate performance report
    global performance_data;

    if isempty(performance_data)
        fprintf('No active monitoring session found\n');
        return;
    end

    performance_data.end_time = toc(performance_data.start_time);
    performance_data.end_memory = getMemoryUsage();

    generatePerformanceReport(performance_data);
end

function recordPhase(phase_name)
    % Record the start of a new performance phase
    global performance_data;

    if isempty(performance_data)
        return; % Monitoring not active
    end

    % End previous phase if exists
    if ~isempty(performance_data.current_phase)
        endPhase();
    end

    % Start new phase
    performance_data.current_phase = phase_name;
    performance_data.phase_start = tic;
    performance_data.phases{end+1} = phase_name;

    % Record memory at phase start
    performance_data.memory_usage(end+1) = getMemoryUsage();
end

function endPhase()
    % End the current performance phase
    global performance_data;

    if isempty(performance_data) || isempty(performance_data.current_phase)
        return;
    end

    phase_time = toc(performance_data.phase_start);
    performance_data.phase_times{end+1} = struct(...
        'phase', performance_data.current_phase, ...
        'duration', phase_time, ...
        'memory_start', performance_data.memory_usage(end), ...
        'memory_end', getMemoryUsage());

    performance_data.current_phase = '';
    performance_data.phase_start = [];
end

function recordTrialTime(trial_num, simulation_time, processing_time)
    % Record timing for individual trials
    global performance_data;

    if isempty(performance_data)
        return;
    end

    performance_data.trial_times(end+1) = struct(...
        'trial', trial_num, ...
        'simulation_time', simulation_time, ...
        'processing_time', processing_time, ...
        'total_time', simulation_time + processing_time);
end

function recordBatchTime(batch_num, batch_size, total_time, successful_trials)
    % Record timing for batches
    global performance_data;

    if isempty(performance_data)
        return;
    end

    performance_data.batch_times(end+1) = struct(...
        'batch', batch_num, ...
        'size', batch_size, ...
        'total_time', total_time, ...
        'successful_trials', successful_trials, ...
        'trials_per_second', successful_trials / total_time);
end

function recordSimulationTime(trial_num, time)
    % Record individual simulation time
    global performance_data;

    if isempty(performance_data)
        return;
    end

    performance_data.simulation_times(end+1) = struct(...
        'trial', trial_num, ...
        'time', time);
end

function recordProcessingTime(trial_num, time)
    % Record individual processing time
    global performance_data;

    if isempty(performance_data)
        return;
    end

    performance_data.processing_times(end+1) = struct(...
        'trial', trial_num, ...
        'time', time);
end

function recordCheckpointTime(time)
    % Record checkpoint save time
    global performance_data;

    if isempty(performance_data)
        return;
    end

    performance_data.checkpoint_times(end+1) = time;
end

function memory_usage = getMemoryUsage()
    % Get current memory usage in MB
    try
        [~, systemview] = memory;
        memory_usage = systemview.PhysicalMemory.Available / 1024^2;
    catch
        memory_usage = 0;
    end
end

function displayPerformanceStatus()
    % Display current performance status
    global performance_data;

    if isempty(performance_data)
        fprintf('No active performance monitoring session\n');
        return;
    end

    elapsed_time = toc(performance_data.start_time);
    current_memory = getMemoryUsage();

    fprintf('\n=== Performance Status ===\n');
    fprintf('Elapsed time: %.2f seconds\n', elapsed_time);
    fprintf('Current memory: %.1f MB\n', current_memory);

    if ~isempty(performance_data.current_phase)
        phase_elapsed = toc(performance_data.phase_start);
        fprintf('Current phase: %s (%.2f seconds)\n', ...
            performance_data.current_phase, phase_elapsed);
    end

    if ~isempty(performance_data.phase_times)
        fprintf('\nCompleted phases:\n');
        for i = 1:length(performance_data.phase_times)
            phase = performance_data.phase_times{i};
            fprintf('  %s: %.2f seconds\n', phase.phase, phase.duration);
        end
    end
end

function generatePerformanceReport(data)
    % Generate comprehensive performance report
    fprintf('\n=== PERFORMANCE REPORT ===\n');
    fprintf('Total execution time: %.2f seconds (%.2f minutes)\n', ...
        data.end_time, data.end_time / 60);

    % Memory analysis
    memory_change = data.end_memory - data.start_memory;
    fprintf('Memory change: %.1f MB\n', memory_change);

    % Phase analysis
    if ~isempty(data.phase_times)
        fprintf('\n--- Phase Analysis ---\n');
        total_phase_time = 0;
        for i = 1:length(data.phase_times)
            phase = data.phase_times{i};
            total_phase_time = total_phase_time + phase.duration;
            percentage = 100 * phase.duration / data.end_time;
            fprintf('  %s: %.2f seconds (%.1f%%)\n', ...
                phase.phase, phase.duration, percentage);
        end

        % Identify slowest phases
        [~, slowest_idx] = max([data.phase_times{:}.duration]);
        fprintf('\nSlowest phase: %s (%.2f seconds)\n', ...
            data.phase_times{slowest_idx}.phase, ...
            data.phase_times{slowest_idx}.duration);
    end

    % Trial analysis
    if ~isempty(data.trial_times)
        fprintf('\n--- Trial Analysis ---\n');
        trial_times = [data.trial_times.total_time];
        simulation_times = [data.trial_times.simulation_time];
        processing_times = [data.trial_times.processing_time];

        fprintf('Average trial time: %.3f seconds\n', mean(trial_times));
        fprintf('Average simulation time: %.3f seconds\n', mean(simulation_times));
        fprintf('Average processing time: %.3f seconds\n', mean(processing_times));
        fprintf('Trial time std dev: %.3f seconds\n', std(trial_times));

        % Identify slowest trials
        [~, slowest_trial_idx] = max(trial_times);
        slowest_trial = data.trial_times(slowest_trial_idx);
        fprintf('Slowest trial: %d (%.3f seconds)\n', ...
            slowest_trial.trial, slowest_trial.total_time);
    end

    % Batch analysis
    if ~isempty(data.batch_times)
        fprintf('\n--- Batch Analysis ---\n');
        batch_times = [data.batch_times.total_time];
        trials_per_second = [data.batch_times.trials_per_second];

        fprintf('Average batch time: %.2f seconds\n', mean(batch_times));
        fprintf('Average throughput: %.2f trials/second\n', mean(trials_per_second));
        fprintf('Batch time std dev: %.2f seconds\n', std(batch_times));

        % Identify slowest batches
        [~, slowest_batch_idx] = max(batch_times);
        slowest_batch = data.batch_times(slowest_batch_idx);
        fprintf('Slowest batch: %d (%.2f seconds, %.2f trials/sec)\n', ...
            slowest_batch.batch, slowest_batch.total_time, slowest_batch.trials_per_second);
    end

    % Checkpoint analysis
    if ~isempty(data.checkpoint_times)
        fprintf('\n--- Checkpoint Analysis ---\n');
        fprintf('Average checkpoint time: %.3f seconds\n', mean(data.checkpoint_times));
        fprintf('Total checkpoint overhead: %.2f seconds\n', sum(data.checkpoint_times));
        fprintf('Checkpoint overhead: %.1f%% of total time\n', ...
            100 * sum(data.checkpoint_times) / data.end_time);
    end

    % Performance recommendations
    fprintf('\n--- Performance Recommendations ---\n');
    generateRecommendations(data);
end

function generateRecommendations(data)
    % Generate performance improvement recommendations
    recommendations = {};

    % Check for slow phases
    if ~isempty(data.phase_times)
        phase_durations = [data.phase_times{:}.duration];
        [~, slowest_idx] = max(phase_durations);
        slowest_phase = data.phase_times{slowest_idx};

        if slowest_phase.duration > data.end_time * 0.5
            recommendations{end+1} = sprintf('Phase "%s" is taking %.1f%% of total time - consider optimization', ...
                slowest_phase.phase, 100 * slowest_phase.duration / data.end_time);
        end
    end

    % Check for slow trials
    if ~isempty(data.trial_times)
        trial_times = [data.trial_times.total_time];
        slow_trials = trial_times > mean(trial_times) + 2 * std(trial_times);

        if sum(slow_trials) > 0
            recommendations{end+1} = sprintf('%d trials are significantly slower than average - investigate outliers', ...
                sum(slow_trials));
        end
    end

    % Check for memory issues
    if ~isempty(data.memory_usage)
        memory_trend = diff(data.memory_usage);
        if any(memory_trend < -100) % More than 100MB decrease
            recommendations{end+1} = 'Significant memory usage detected - consider reducing batch size';
        end
    end

    % Check checkpoint overhead
    if ~isempty(data.checkpoint_times)
        checkpoint_overhead = 100 * sum(data.checkpoint_times) / data.end_time;
        if checkpoint_overhead > 5
            recommendations{end+1} = sprintf('Checkpoint overhead is %.1f%% - consider less frequent saves', ...
                checkpoint_overhead);
        end
    end

    % Check batch efficiency
    if ~isempty(data.batch_times)
        batch_times = [data.batch_times.total_time];
        batch_sizes = [data.batch_times.size];
        efficiency = batch_sizes ./ batch_times;

        if std(efficiency) > mean(efficiency) * 0.5
            recommendations{end+1} = 'Batch efficiency varies significantly - consider optimizing batch size';
        end
    end

    % Display recommendations
    if isempty(recommendations)
        fprintf('  No major performance issues detected\n');
    else
        for i = 1:length(recommendations)
            fprintf('  • %s\n', recommendations{i});
        end
    end
end

function analyzePerformance(data)
    % Analyze performance data and provide insights
    fprintf('\n=== Performance Analysis ===\n');

    % Time distribution analysis
    if ~isempty(data.phase_times)
        phase_names = {data.phase_times{:}.phase};
        phase_durations = [data.phase_times{:}.duration];

        fprintf('Time distribution by phase:\n');
        for i = 1:length(phase_names)
            percentage = 100 * phase_durations(i) / data.end_time;
            fprintf('  %s: %.1f%%\n', phase_names{i}, percentage);
        end
    end

    % Throughput analysis
    if ~isempty(data.batch_times)
        throughput = [data.batch_times.trials_per_second];
        fprintf('\nThroughput analysis:\n');
        fprintf('  Average: %.2f trials/second\n', mean(throughput));
        fprintf('  Peak: %.2f trials/second\n', max(throughput));
        fprintf('  Minimum: %.2f trials/second\n', min(throughput));
        fprintf('  Consistency: %.1f%% (lower is better)\n', ...
            100 * std(throughput) / mean(throughput));
    end

    % Memory analysis
    if ~isempty(data.memory_usage)
        memory_change = data.memory_usage(end) - data.memory_usage(1);
        fprintf('\nMemory analysis:\n');
        fprintf('  Start: %.1f MB\n', data.memory_usage(1));
        fprintf('  End: %.1f MB\n', data.memory_usage(end));
        fprintf('  Change: %.1f MB\n', memory_change);

        if memory_change < -500
            fprintf('  ⚠️  Significant memory usage detected\n');
        end
    end
end

function identifyBottlenecks()
    % Identify potential performance bottlenecks
    global performance_data;

    if isempty(performance_data)
        fprintf('No performance data available\n');
        return;
    end

    fprintf('\n=== Bottleneck Analysis ===\n');

    bottlenecks = {};

    % Check for slow phases
    if ~isempty(performance_data.phase_times)
        phase_durations = [performance_data.phase_times{:}.duration];
        [~, slowest_idx] = max(phase_durations);
        slowest_phase = performance_data.phase_times{slowest_idx};

        if slowest_phase.duration > performance_data.end_time * 0.3
            bottlenecks{end+1} = sprintf('Phase "%s" is the primary bottleneck (%.1f%% of time)', ...
                slowest_phase.phase, 100 * slowest_phase.duration / performance_data.end_time);
        end
    end

    % Check for memory bottlenecks
    if ~isempty(performance_data.memory_usage)
        memory_trend = diff(performance_data.memory_usage);
        if any(memory_trend < -200)
            bottlenecks{end+1} = 'Memory exhaustion detected - may be causing slowdowns';
        end
    end

    % Check for I/O bottlenecks
    if ~isempty(performance_data.checkpoint_times)
        avg_checkpoint_time = mean(performance_data.checkpoint_times);
        if avg_checkpoint_time > 1.0
            bottlenecks{end+1} = sprintf('Slow checkpoint saves (%.2f seconds average) - I/O bottleneck', ...
                avg_checkpoint_time);
        end
    end

    % Check for parallel efficiency
    if ~isempty(performance_data.batch_times)
        batch_times = [performance_data.batch_times.total_time];
        if std(batch_times) > mean(batch_times) * 0.3
            bottlenecks{end+1} = 'Inconsistent batch performance - parallel efficiency issues';
        end
    end

    % Display bottlenecks
    if isempty(bottlenecks)
        fprintf('No major bottlenecks identified\n');
    else
        for i = 1:length(bottlenecks)
            fprintf('• %s\n', bottlenecks{i});
        end
    end
end
