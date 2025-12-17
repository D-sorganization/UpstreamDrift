function inspect_simscape_hierarchy(simOut)
    % INSPECT_SIMSCAPE_HIERARCHY - Recursively displays simlog hierarchy
    % Usage: inspect_simscape_hierarchy(simOut)
    % Displays the complete structure of simOut.simlog with data availability info

    if nargin < 1 || isempty(simOut)
        error('Please provide a simOut object with simlog data');
    end

    if ~isfield(simOut, 'simlog')
        error('simOut object does not contain simlog field');
    end

    fprintf('\n=== SIMSCAPE HIERARCHY INSPECTION ===\n');
    fprintf('Timestamp: %s\n', datestr(now));
    fprintf('========================================\n\n');

    % Start recursive traversal
    displayNode(simOut.simlog, '', 0, 5); % Max depth 5

    fprintf('\n=== END HIERARCHY ===\n');
end

function displayNode(node, prefix, depth, maxDepth)
    % Recursive function to display node hierarchy

    if depth > maxDepth
        fprintf('%s[MAX DEPTH REACHED]\n', prefix);
        return;
    end

    % Get node info
    nodeInfo = getNodeInfo(node);

    % Display current node
    fprintf('%s%s\n', prefix, nodeInfo);

    % Get children
    children = getNodeChildren(node);

    % Display children
    if ~isempty(children)
        childNames = fieldnames(children);
        for i = 1:length(childNames)
            childName = childNames{i};
            childNode = children.(childName);

            % Create new prefix for child
            if i == length(childNames)
                newPrefix = [prefix '└── '];
                nextPrefix = [prefix '    '];
            else
                newPrefix = [prefix '├── '];
                nextPrefix = [prefix '│   '];
            end

            % Display child name and info
            childInfo = getNodeInfo(childNode);
            fprintf('%s%s: %s\n', newPrefix, childName, childInfo);

            % Recurse into child
            displayNode(childNode, nextPrefix, depth + 1, maxDepth);
        end
    end
end

function nodeInfo = getNodeInfo(node)
    % Get comprehensive info about a node

    nodeInfo = '';

    % Check node type
    if isa(node, 'simscape.logging.Node')
        nodeInfo = '[Node]';

        % Check for ID
        if isprop(node, 'id') && ~isempty(node.id)
            nodeInfo = [nodeInfo ' ID:' node.id];
        end

        % Check if exportable
        if isprop(node, 'exportable')
            nodeInfo = [nodeInfo ' exportable:' num2str(node.exportable)];
        end

        % Check for series data
        if isprop(node, 'series') && ~isempty(node.series)
            try
                if isprop(node.series, 'time') && ~isempty(node.series.time)
                    timeData = node.series.time;
                    nodeInfo = [nodeInfo ' [TIME: ' num2str(length(timeData)) ' pts]'];
                end

                if isprop(node.series, 'values') && ~isempty(node.series.values)
                    try
                        valData = node.series.values('');
                        if isnumeric(valData)
                            nodeInfo = [nodeInfo ' [VALUES: ' num2str(size(valData)) ']'];
                        end
                    catch
                        nodeInfo = [nodeInfo ' [VALUES: cannot access]'];
                    end
                end

                % Check for series children
                if isprop(node.series, 'children')
                    try
                        seriesChildren = node.series.children();
                        if ~isempty(seriesChildren)
                            nodeInfo = [nodeInfo ' [SERIES_CHILDREN: ' num2str(length(seriesChildren)) ']'];
                        end
                    catch
                        % Series children method failed
                    end
                end
            catch me
                nodeInfo = [nodeInfo ' [SERIES_ERROR: ' me.message ']'];
            end
        end

        % Check for hasData method
        if ismethod(node, 'hasData')
            try
                hasDataResult = node.hasData();
                nodeInfo = [nodeInfo ' hasData:' num2str(hasDataResult)];
            catch
                nodeInfo = [nodeInfo ' hasData:error'];
            end
        end

    elseif isstruct(node)
        nodeInfo = '[Struct]';
        fields = fieldnames(node);
        nodeInfo = [nodeInfo ' fields:' num2str(length(fields))];

        % Show some field names
        if length(fields) <= 5
            nodeInfo = [nodeInfo ' (' strjoin(fields, ',') ')'];
        else
            nodeInfo = [nodeInfo ' (' strjoin(fields(1:3), ',') '...)'];
        end

    elseif iscell(node)
        nodeInfo = ['[Cell ' num2str(size(node)) ']'];

    elseif isnumeric(node)
        if isempty(node)
            nodeInfo = '[Empty numeric]';
        else
            nodeInfo = ['[Numeric ' num2str(size(node)) ']'];
            if numel(node) <= 10
                nodeInfo = [nodeInfo ' = [' num2str(node(:)') ']'];
            end
        end

    elseif ischar(node) || isstring(node)
        nodeInfo = ['[String] "' char(node) '"'];

    else
        nodeInfo = ['[' class(node) ']'];
    end
end

function children = getNodeChildren(node)
    % Get children of a node using multiple methods

    children = struct();

    if isa(node, 'simscape.logging.Node')
        % Method 1: Try children() method
        try
            childrenMethod = node.children();
            if ~isempty(childrenMethod)
                for i = 1:length(childrenMethod)
                    child = childrenMethod{i};
                    if isprop(child, 'id') && ~isempty(child.id)
                        fieldName = matlab.lang.makeValidName(child.id);
                        children.(fieldName) = child;
                    else
                        children.(['child_' num2str(i)]) = child;
                    end
                end
            end
        catch
            % children() method failed
        end

        % Method 2: Try to access properties as children
        try
            props = properties(node);
            for i = 1:length(props)
                propName = props{i};
                if ~strcmp(propName, 'series') && ~strcmp(propName, 'id') && ~strcmp(propName, 'exportable')
                    try
                        propValue = node.(propName);
                        if isa(propValue, 'simscape.logging.Node')
                            children.(propName) = propValue;
                        end
                    catch
                        % Property access failed
                    end
                end
            end
        catch
            % Property enumeration failed
        end

    elseif isstruct(node)
        % For structs, use fieldnames
        children = node;

    end
end
