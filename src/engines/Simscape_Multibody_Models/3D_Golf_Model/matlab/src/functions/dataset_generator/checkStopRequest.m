function shouldStop = checkStopRequest(handles)
    shouldStop = false;
    try
        % Get current handles
        current_handles = guidata(handles.fig);
        if isfield(current_handles, 'should_stop') && current_handles.should_stop
            shouldStop = true;
        end

        % Force UI update to prevent freezing
        drawnow;

    catch
        % If we can't access handles, assume we should stop
        shouldStop = true;
    end
end
