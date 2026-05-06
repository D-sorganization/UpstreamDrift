function predicted = extract_predicted(simOut, targetFrame)
%EXTRACT_PREDICTED  Pull each manifest target column from a candidate simOut.
%
%   predicted = frame_search.extract_predicted(simOut, targetFrame) returns
%   a struct whose field names match the field names of targetFrame
%   (excluding any 'time' field) and whose values are the corresponding
%   scalar values pulled from simOut at the final time index.
%
%   A missing target column raises an actionable error so production
%   overnight runs cannot silently claim good tracking.

    arguments
        simOut
        targetFrame struct
    end

    predicted = struct();
    targetNames = fieldnames(targetFrame);
    for idx = 1:numel(targetNames)
        name = targetNames{idx};
        if strcmp(name, 'time')
            continue;
        end
        info = frame_search.parse_target_column(string(name));
        try
            value = frame_search.lookup_signal_value(simOut, info, []);
        catch ME
            error('frame_search:extract_predicted:missingColumn', ...
                'Could not extract predicted target %s: %s', name, ME.message);
        end
        predicted.(name) = value;
    end
end
