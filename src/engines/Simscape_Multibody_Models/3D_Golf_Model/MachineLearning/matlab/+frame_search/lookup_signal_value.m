function value = lookup_signal_value(simOut, columnInfo, sampleIndex)
%LOOKUP_SIGNAL_VALUE  Pull a scalar from a Simulink simOut for one column.
%
%   value = frame_search.lookup_signal_value(simOut, columnInfo, sampleIndex)
%   returns the scalar value of the signal described by columnInfo (output
%   of frame_search.parse_target_column) at the given sample index. When
%   sampleIndex is empty or NaN the last sample is used.
%
%   Resolution order:
%     1. CombinedSignalBus.<group>.<signal>(:, idx)
%     2. CombinedSignalBus.<signal>(:, idx)
%     3. simOut.logsout.getElement(<group>) child <signal>
%     4. simOut.logsout.getElement(<signal>)
%     5. Direct property/field on simOut named <signal> or <group>_<signal>
%
%   Errors with frame_search:lookup_signal_value:notFound if the column
%   cannot be resolved. The runner converts that into an actionable error
%   listing the missing target column.

    arguments
        simOut
        columnInfo struct
        sampleIndex = []
    end

    raw = [];

    % --- 1. CombinedSignalBus dotted path -----------------------------------
    bus = local_get(simOut, 'CombinedSignalBus');
    if ~isempty(bus)
        raw = local_dig(bus, columnInfo.busPath);
        if isempty(raw) && ~isempty(columnInfo.signal)
            raw = local_dig_recursive(bus, columnInfo.signal);
        end
    end

    % --- 2. logsout (group element with child signal, then signal alone) ----
    if isempty(raw)
        raw = local_logsout(simOut, columnInfo);
    end

    % --- 3. Direct field/property lookup ------------------------------------
    if isempty(raw)
        for nameCell = columnInfo.lookupNames
            name = nameCell{1};
            try
                if isprop(simOut, name) || isfield(simOut, name)
                    candidate = simOut.(name);
                    raw = local_extract(candidate);
                    if ~isempty(raw)
                        break;
                    end
                end
            catch
            end
        end
    end

    if isempty(raw)
        error('frame_search:lookup_signal_value:notFound', ...
            'Target column %s could not be resolved in simOut.', ...
            columnInfo.raw);
    end

    raw = squeeze(raw);
    if isvector(raw)
        raw = raw(:);
        if isempty(sampleIndex) || ~isfinite(sampleIndex)
            value = raw(end);
        else
            value = raw(min(max(1, sampleIndex), numel(raw)));
        end
        return;
    end

    % Multi-column matrix: rows = time, cols = components.
    if size(raw, 1) < size(raw, 2)
        raw = raw.';
    end
    colIdx = 1;
    if isfield(columnInfo, 'index') && ~isnan(columnInfo.index)
        colIdx = max(1, min(size(raw, 2), round(columnInfo.index)));
    end
    if isempty(sampleIndex) || ~isfinite(sampleIndex)
        rowIdx = size(raw, 1);
    else
        rowIdx = min(max(1, sampleIndex), size(raw, 1));
    end
    value = raw(rowIdx, colIdx);
end

% --- helpers ---------------------------------------------------------------
function v = local_get(obj, name)
    v = [];
    try
        if isprop(obj, name) || isfield(obj, name)
            v = obj.(name);
        end
    catch
    end
end

function v = local_dig(bus, path)
    v = [];
    if ~isstruct(bus) || isempty(path)
        return;
    end
    cur = bus;
    for k = 1:numel(path)
        name = char(path(k));
        if isstruct(cur) && isfield(cur, name)
            cur = cur.(name);
        else
            return;
        end
    end
    v = local_extract(cur);
end

function v = local_dig_recursive(bus, name)
    v = [];
    if ~isstruct(bus)
        return;
    end
    if isfield(bus, name)
        v = local_extract(bus.(name));
        return;
    end
    f = fieldnames(bus);
    for i = 1:numel(f)
        sub = bus.(f{i});
        if isstruct(sub)
            v = local_dig_recursive(sub, name);
            if ~isempty(v), return; end
        end
    end
end

function v = local_logsout(simOut, columnInfo)
    v = [];
    try
        if ~(isprop(simOut, 'logsout') && ~isempty(simOut.logsout))
            return;
        end
        ls = simOut.logsout;
        candidates = columnInfo.lookupNames;
        if ~isempty(columnInfo.group)
            candidates = [{columnInfo.group}, candidates];
        end
        for k = 1:numel(candidates)
            name = candidates{k};
            try
                el = ls.getElement(name);
            catch
                el = [];
            end
            if isempty(el), continue; end
            if isprop(el, 'Values')
                v = local_extract(el.Values);
                if ~isempty(v), return; end
            end
        end
    catch
    end
end

function v = local_extract(x)
    v = [];
    try
        if isnumeric(x)
            v = double(x);
        elseif isa(x, 'timeseries')
            v = double(x.Data);
        elseif isstruct(x)
            if isfield(x, 'Data')
                v = double(x.Data);
            elseif isfield(x, 'signals') && isfield(x.signals, 'values')
                v = double(x.signals.values);
            end
        end
    catch
        v = [];
    end
end
