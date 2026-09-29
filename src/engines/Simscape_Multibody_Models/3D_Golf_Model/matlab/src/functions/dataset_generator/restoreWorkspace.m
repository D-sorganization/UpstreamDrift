function restoreWorkspace(initial_vars)
    % Restore workspace to initial state by clearing new variables
    current_vars = who;
    new_vars = setdiff(current_vars, initial_vars);

    if ~isempty(new_vars)
        clear(new_vars{:});
    end
end
