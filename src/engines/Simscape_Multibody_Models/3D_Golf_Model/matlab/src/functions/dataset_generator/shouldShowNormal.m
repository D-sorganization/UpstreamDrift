function should_show_normal = shouldShowNormal(handles)
    if ~isfield(handles, 'verbosity_popup')
        should_show_normal = true; % Default to showing if no verbosity control
        return;
    end

    verbosity_options = {'Normal', 'Silent', 'Verbose', 'Debug'};
    verbosity_idx = get(handles.verbosity_popup, 'Value');
    if verbosity_idx <= length(verbosity_options)
        verbosity_level = verbosity_options{verbosity_idx};
    else
        verbosity_level = 'Normal';
    end

    % Show normal output for Normal, Verbose, and Debug levels (not Silent)
    should_show_normal = ~strcmp(verbosity_level, 'Silent');
end
