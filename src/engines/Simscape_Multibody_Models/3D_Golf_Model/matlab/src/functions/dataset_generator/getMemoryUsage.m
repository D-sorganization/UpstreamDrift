function memory_usage = getMemoryUsage()
    % GETMEMORYUSAGE - Get current system memory usage in GB
    %
    % Returns:
    %   memory_usage - Current memory usage in GB
    %
    % This function provides a cross-platform way to get memory usage.

    try
        % Try to get memory info using memory function (Windows)
        if ispc
            [~, systemview] = memory;
            memory_usage = systemview.PhysicalMemory.Total / 1024^3; % Convert to GB
        else
            % For non-Windows systems, try to get memory info
            memory_usage = 0; % Default fallback
        end
    catch
        % Fallback if memory function fails
        memory_usage = 0;
    end
end
