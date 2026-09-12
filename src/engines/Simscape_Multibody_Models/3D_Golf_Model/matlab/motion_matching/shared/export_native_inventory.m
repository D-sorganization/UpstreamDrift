function export_native_inventory(model, output)
%EXPORT_NATIVE_INVENTORY Record native expressions, values and block connections.
% Call after loading and configuring the exact candidate model. This inventory
% does not compile variants, convert units, or certify dynamic equivalence.
% Commented blocks are retained so a consumer can audit ancestor exclusions.
    arguments
        model (1,1) string
        output (1,1) string
    end
    assert(strcmp(version('-release'), '2025b'), ...
        'nativeInventory:Release', 'R2025b is required');
    assert(bdIsLoaded(model), 'nativeInventory:NotLoaded', ...
        'Load and configure the candidate model before exporting');
    assert(~isfile(output), 'nativeInventory:ExistingOutput', ...
        'Preserve existing inventories');
    paths = find_system(model, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'IncludeCommented', 'on', 'Type', 'Block');
    blocks = cell(numel(paths), 1);
    for k = 1:numel(paths)
        path = paths{k};
        block = struct('path', path, 'parent', get_param(path, 'Parent'), ...
            'block_type', get_param(path, 'BlockType'), ...
            'commented', get_param(path, 'Commented'), ...
            'source_block', optionalParameter(path, 'SourceBlock'), ...
            'library_reference', optionalParameter(path, 'ReferenceBlock'), ...
            'parameters', {parameterInventory(path)}, ...
            'connectivity', connectionInventory(path));
        blocks{k} = block;
    end
    result = struct('schema_version', 2, 'matlab_release', version('-release'), ...
        'matlab_version', version, 'model', model, ...
        'model_file', get_param(model, 'FileName'), ...
        'qualification', 'native uncompiled inventory only; not a physics certificate', ...
        'blocks', {blocks});
    fid = fopen(output, 'w');
    assert(fid ~= -1, 'nativeInventory:Output', 'Cannot open output');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s', jsonencode(result, PrettyPrint=true));
end

function value = optionalParameter(path, name)
    fields = get_param(path, 'ObjectParameters');
    value = '';
    if isfield(fields, name)
        value = get_param(path, name);
    end
end

function parameters = parameterInventory(path)
    dialog = get_param(path, 'DialogParameters');
    names = {};
    if isstruct(dialog)
        names = fieldnames(dialog);
    end
    parameters = cell(numel(names), 1);
    for k = 1:numel(names)
        raw = get_param(path, names{k});
        entry = struct('name', names{k}, 'expression', {raw}, ...
            'resolved_numeric', false, 'numeric_value', [], 'resolution_error', '');
        try
            if ischar(raw) || isstring(raw)
                value = slResolve(raw, path);
            else
                value = raw;
            end
            if (isnumeric(value) || islogical(value)) && isreal(value) && all(isfinite(value), 'all')
                entry.resolved_numeric = true;
                entry.numeric_value = value;
            else
                entry.resolution_error = 'nonfinite, complex, or nonnumeric value';
            end
        catch error
            entry.resolution_error = error.identifier;
        end
        parameters{k} = entry;
    end
end

function connections = connectionInventory(path)
    connections = get_param(path, 'PortConnectivity');
    for k = 1:numel(connections)
        connections(k).SrcEndpoint = {};
        connections(k).DstEndpoint = {};
        if startsWith(connections(k).Type, {'LConn', 'RConn'})
            connections(k).SrcEndpoint = portEndpoints(connections(k).SrcPort);
            connections(k).DstEndpoint = portEndpoints(connections(k).DstPort);
        end
        connections(k).SrcBlock = blockPaths(connections(k).SrcBlock);
        connections(k).DstBlock = blockPaths(connections(k).DstBlock);
    end
end

function endpoints = portEndpoints(handles)
    endpoints = cell(numel(handles), 1);
    for k = 1:numel(handles)
        assert(handles(k) > 0, 'nativeInventory:PortHandle', ...
            'A physical connection must reference a live port');
        parent = get_param(handles(k), 'Parent');
        ports = get_param(parent, 'PortHandles');
        kinds = fieldnames(ports);
        endpoint = '';
        for n = 1:numel(kinds)
            index = find(ports.(kinds{n}) == handles(k));
            if ~isempty(index)
                assert(isscalar(index), 'nativeInventory:AmbiguousPort', ...
                    'Port must have one stable identity');
                endpoint = sprintf('%s:%s%d', parent, kinds{n}, index);
                break;
            end
        end
        assert(~isempty(endpoint), 'nativeInventory:MissingPort', ...
            'Physical endpoint not found in parent PortHandles');
        endpoints{k} = endpoint;
    end
end

function paths = blockPaths(handles)
    paths = cell(numel(handles), 1);
    for k = 1:numel(handles)
        if handles(k) == -1
            paths{k} = '<unconnected>';
        else
            paths{k} = getfullname(handles(k));
        end
    end
end
