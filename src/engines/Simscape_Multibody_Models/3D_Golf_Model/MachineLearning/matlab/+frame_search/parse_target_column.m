function info = parse_target_column(columnName)
%PARSE_TARGET_COLUMN  Decode a manifest target column into a bus path + index.
%
%   info = frame_search.parse_target_column('ClubLogs_CHGlobalPosition_1')
%   returns a struct with fields:
%       .raw      : original column name (char)
%       .busPath  : string array describing CombinedSignalBus dotted path,
%                   e.g. ["ClubLogs", "CHGlobalPosition"].
%       .signal   : the leaf signal name (char), e.g. 'CHGlobalPosition'.
%       .group    : the top-level bus group (char), e.g. 'ClubLogs'.
%       .index    : 1-based column index when name ends with _<digit>,
%                   else NaN (caller treats scalar signal as column 1).
%       .lookupNames : char cell array of fallback names to try when the
%                   bus path does not resolve, e.g. {'CHGlobalPosition'}.
%
%   This helper is deliberately pure (no Simulink calls) so it can be
%   unit-tested without MATLAB Simulink installed.
%
%   Recognized patterns:
%       <Group>_<Signal>           -> column index 1 (scalar)
%       <Group>_<Signal>_<digit>   -> column index <digit>
%       <Group>_<Signal><digit>    -> column index <digit> (legacy)
%
%   Preconditions:
%     - columnName is a non-empty char or string scalar.

    arguments
        columnName (1,1) string {mustBeNonzeroLengthText}
    end

    raw = char(columnName);
    info = struct( ...
        'raw', raw, ...
        'busPath', string.empty(1,0), ...
        'signal', '', ...
        'group', '', ...
        'index', NaN, ...
        'lookupNames', {{}});

    % Trailing _<digit> => explicit column index.
    tokens = regexp(raw, '^(.+)_([0-9]+)$', 'tokens', 'once');
    if ~isempty(tokens)
        body = tokens{1};
        info.index = str2double(tokens{2});
    else
        % Legacy <name><digit> with no underscore separator.
        legacy = regexp(raw, '^(.+?)([0-9]+)$', 'tokens', 'once');
        if ~isempty(legacy) && ~isempty(legacy{1})
            body = legacy{1};
            info.index = str2double(legacy{2});
        else
            body = raw;
        end
    end

    % Split <Group>_<Signal> on the FIRST underscore so signal names
    % containing underscores survive (e.g. AngularPosition_Z).
    underscore = strfind(body, '_');
    if isempty(underscore)
        info.group = '';
        info.signal = body;
        info.busPath = string(body);
    else
        info.group = body(1:underscore(1) - 1);
        info.signal = body(underscore(1) + 1:end);
        info.busPath = [string(info.group), string(info.signal)];
    end

    info.lookupNames = {info.signal, body, raw};
end
