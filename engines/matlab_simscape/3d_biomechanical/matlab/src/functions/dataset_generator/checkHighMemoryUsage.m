function isHighMemory = checkHighMemoryUsage(threshold_percent)
    if nargin < 1
        threshold_percent = 85; % Default threshold
    end

    try
        memoryInfo = getMemoryInfo();
        isHighMemory = memoryInfo.usage_percent > threshold_percent;

        if isHighMemory
            fprintf('Warning: High memory usage detected: %.1f%%\n', memoryInfo.usage_percent);
        end

    catch
        isHighMemory = false;
    end
end
