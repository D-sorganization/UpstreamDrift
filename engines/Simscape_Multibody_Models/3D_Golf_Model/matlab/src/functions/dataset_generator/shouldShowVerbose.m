function should_show_verbose = shouldShowVerbose(handles)
    if ~isfield(handles, 'verbosity_popup')
        should_show_verbose = true; % Default to showing if no verbosity control
        return;
    end

    verbosity_options = {'Normal', 'Silent', 'Verbose', 'Debug'};
    verbosity_idx = get(handles.verbosity_popup, 'Value');
    if verbosity_idx <= length(verbosity_options)
        verbosity_level = verbosity_options{verbosity_idx};
    else
        verbosity_level = 'Normal';
    end

    % Show verbose output for Verbose and Debug levels
    should_show_verbose = strcmp(verbosity_level, 'Verbose') || strcmp(verbosity_level, 'Debug');
end
