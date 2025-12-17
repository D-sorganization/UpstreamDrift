function logBatchResult(batch_num, batch_size, successful, failed, duration)
    % LOGBATCHRESULT - Log batch processing results
    %
    % Usage:
    %   logBatchResult(batch_num, batch_size, successful, failed, duration)
    %
    % This function logs the results of batch processing for monitoring
    % and debugging purposes.

    success_rate = 100 * successful / batch_size;

    if failed == 0
        logMessage('info', 'Batch %d: %d/%d successful (%.1f%%) in %.1f seconds', ...
            batch_num, successful, batch_size, success_rate, duration);
    else
        logMessage('info', 'Batch %d completed: %d successful, %d failed (%.1f%% success rate) in %.2f seconds', ...
            batch_num, successful, failed, success_rate, duration);
    end
end
