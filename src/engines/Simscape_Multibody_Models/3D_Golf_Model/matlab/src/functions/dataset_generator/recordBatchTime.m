function recordBatchTime(batch_num, batch_size, total_time, successful_trials)
    % RECORDBATCHTIME - Record timing for batches
    %
    % Usage:
    %   recordBatchTime(batch_num, batch_size, total_time, successful_trials)
    %
    % This function is part of the performance monitoring system and records
    % timing information for each batch of trials processed.

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
