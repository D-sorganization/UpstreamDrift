function recordPhase(phase_name)
    % RECORDPHASE - Record the start of a new performance phase
    %
    % Usage:
    %   recordPhase('Phase Name')
    %
    % This function is part of the performance monitoring system and should
    % be called at the start of each major phase of execution.

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
