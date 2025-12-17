function memory_monitor(varargin)
    % MEMORY_MONITOR - Monitor memory usage and provide warnings
    %
    % Usage:
    %   memory_monitor()                    % Display current memory status
    %   memory_monitor('check')             % Check if memory is low
    %   memory_monitor('monitor', interval) % Continuous monitoring
    %   memory_monitor('cleanup')           % Force memory cleanup

    if nargin == 0
        displayMemoryStatus();
        return;
    end

    action = varargin{1};

    switch lower(action)
        case 'check'
            is_low = checkMemoryStatus();
            if is_low
                warning('Memory usage is high - consider cleanup');
            end
        case 'monitor'
            if nargin > 1
                interval = varargin{2};
            else
                interval = 30; % Default 30 seconds
            end
            continuousMonitoring(interval);
        case 'cleanup'
            performMemoryCleanup();
        otherwise
            error('Unknown action: %s', action);
    end
end

function displayMemoryStatus()
    % Display detailed memory status
    try
        [user, system] = memory;

        fprintf('\n=== Memory Status ===\n');
        fprintf('Physical Memory:\n');
        fprintf('  Total: %.1f GB\n', system.PhysicalMemory.Total / 1024^3);
        fprintf('  Available: %.1f GB\n', system.PhysicalMemory.Available / 1024^3);
        fprintf('  Used: %.1f GB (%.1f%%)\n', ...
            (system.PhysicalMemory.Total - system.PhysicalMemory.Available) / 1024^3, ...
            100 * (system.PhysicalMemory.Total - system.PhysicalMemory.Available) / system.PhysicalMemory.Total);

        fprintf('\nVirtual Memory:\n');
        fprintf('  Total: %.1f GB\n', system.VirtualMemory.Total / 1024^3);
        fprintf('  Available: %.1f GB\n', system.VirtualMemory.Available / 1024^3);
        fprintf('  Used: %.1f GB (%.1f%%)\n', ...
            (system.VirtualMemory.Total - system.VirtualMemory.Available) / 1024^3, ...
            100 * (system.VirtualMemory.Total - system.VirtualMemory.Available) / system.VirtualMemory.Total);

        fprintf('\nMATLAB Memory:\n');
        fprintf('  Peak: %.1f MB\n', user.MemUsedMATLAB / 1024^2);
        fprintf('  Current: %.1f MB\n', user.MemUsedMATLAB / 1024^2);

        % Check for potential issues
        physical_usage_pct = 100 * (system.PhysicalMemory.Total - system.PhysicalMemory.Available) / system.PhysicalMemory.Total;
        virtual_usage_pct = 100 * (system.VirtualMemory.Total - system.VirtualMemory.Available) / system.VirtualMemory.Total;

        if physical_usage_pct > 90
            fprintf('\n⚠️  WARNING: Physical memory usage is very high (%.1f%%)\n', physical_usage_pct);
        elseif physical_usage_pct > 80
            fprintf('\n⚠️  WARNING: Physical memory usage is high (%.1f%%)\n', physical_usage_pct);
        end

        if virtual_usage_pct > 90
            fprintf('\n⚠️  WARNING: Virtual memory usage is very high (%.1f%%)\n', virtual_usage_pct);
        end

    catch ME
        fprintf('Error getting memory status: %s\n', ME.message);
    end
end

function is_low = checkMemoryStatus()
    % Check if memory is running low
    try
        [~, system] = memory;

        physical_usage_pct = 100 * (system.PhysicalMemory.Total - system.PhysicalMemory.Available) / system.PhysicalMemory.Total;
        virtual_usage_pct = 100 * (system.VirtualMemory.Total - system.VirtualMemory.Available) / system.VirtualMemory.Total;

        % Consider memory low if physical > 85% or virtual > 90%
        is_low = physical_usage_pct > 85 || virtual_usage_pct > 90;

    catch
        is_low = false; % Assume OK if we can't check
    end
end

function continuousMonitoring(interval)
    % Continuously monitor memory usage
    fprintf('Starting continuous memory monitoring (interval: %d seconds)\n', interval);
    fprintf('Press Ctrl+C to stop\n');

    try
        while true
            displayMemoryStatus();

            if checkMemoryStatus()
                fprintf('\n🚨 CRITICAL: Memory usage is critically high!\n');
                fprintf('Consider stopping current operations and performing cleanup.\n');
            end

            pause(interval);
        end
    catch ME
        if strcmp(ME.identifier, 'MATLAB:interrupt')
            fprintf('\nMemory monitoring stopped by user.\n');
        else
            rethrow(ME);
        end
    end
end

function performMemoryCleanup()
    % Perform aggressive memory cleanup
    fprintf('Performing memory cleanup...\n');

    try
        % Clear workspace variables
        evalin('base', 'clear ans');

        % Force garbage collection
        java.lang.System.gc();

        % Clear MATLAB's internal caches
        clear('functions');

        % Clear persistent variables in current functions
        clear('persistent');

        % Clear any cached data
        if exist('OCTAVE_VERSION', 'builtin')
            % Octave-specific cleanup
            clear('global');
        else
            % MATLAB-specific cleanup
            evalin('base', 'clear global');
        end

        fprintf('✓ Memory cleanup completed\n');

        % Show memory status after cleanup
        displayMemoryStatus();

    catch ME
        fprintf('Warning: Memory cleanup failed: %s\n', ME.message);
    end
end
