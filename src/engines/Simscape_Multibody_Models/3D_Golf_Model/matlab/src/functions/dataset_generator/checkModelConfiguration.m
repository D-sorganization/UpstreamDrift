function checkModelConfiguration(model_name)
    % ... (existing code) ...

    % Ensure Simscape logging is enabled
    try
        log_type = get_param(model_name, 'SimscapeLogType');
        if ~strcmp(log_type, 'All')
            set_param(model_name, 'SimscapeLogType', 'All');
            fprintf('Updated Simscape Log Type to: All\n');
        else
            fprintf('Simscape Log Type: %s (good)\n', log_type);
        end
    catch
        fprintf('Warning: Could not set/check Simscape Log Type. Ensure model has Simscape components.\n');
    end

    % ... (rest of function) ...
end
