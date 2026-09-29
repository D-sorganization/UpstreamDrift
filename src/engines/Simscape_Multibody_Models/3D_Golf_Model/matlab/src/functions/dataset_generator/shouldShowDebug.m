function should_show_debug = shouldShowDebug(handles)
    if ~isfield(handles, 'verbosity_popup')
        should_show_debug = true; % Default to showing if no verbosity control
        return;
    end

    verbosity_options = {'Normal', 'Silent', 'Verbose', 'Debug'};
    verbosity_idx = get(handles.verbosity_popup, 'Value');
    if verbosity_idx <= length(verbosity_options)
        verbosity_level = verbosity_options{verbosity_idx};
    else
        verbosity_level = 'Normal';
    end

    % Only show debug output for Debug verbosity level
    should_show_debug = strcmp(verbosity_level, 'Debug');
end
