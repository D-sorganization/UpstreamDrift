function endPhase()
    % ENDPHASE - End the current performance phase
    %
    % Usage:
    %   endPhase()
    %
    % This function is part of the performance monitoring system and should
    % be called at the end of each major phase of execution.

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
