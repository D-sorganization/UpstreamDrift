function scope = gs3dx_ik_target_scope(jp, requested)
%GS3DX_IK_TARGET_SCOPE Determine IK target scope from joint parameter table.
%   SCOPE = GS3DX_IK_TARGET_SCOPE(JP) resolves the target scope automatically
%   ('auto') based on the presence of lower-body joints in the joint
%   parameter table JP.
%
%   SCOPE = GS3DX_IK_TARGET_SCOPE(JP, REQUESTED) resolves the target scope
%   according to REQUESTED ('auto', 'whole_body', or 'upper_body').
%
%   The returned struct SCOPE has fields:
%       name           - 'whole_body' or 'upper_body' (string)
%       target_indices - 1:14 for 'whole_body', [1 8:14] for 'upper_body' (double row vector)
%
%   Lower-body roles are detected from unique native leaf BlockPath strings
%   containing:
%       'Left Hip Joint', 'Right Hip Joint',
%       'Left Knee', 'Right Knee',
%       'Left Ankle', 'Right Ankle'.

    narginchk(1, 2);

    % Validate JP table
    if ~istable(jp) || ~ismember('BlockPath', jp.Properties.VariableNames)
        error('gs3dx:ik:scope', 'JP must be a table containing a BlockPath variable.');
    end

    % Validate requested argument
    if nargin < 2 || isempty(requested)
        requested = "auto";
    else
        if ~(isstring(requested) || ischar(requested))
            error('gs3dx:ik:scope', 'Requested scope must be a string or character vector.');
        end
        requested = lower(string(requested));
        if ~isscalar(requested) || ~ismember(requested, ["auto", "whole_body", "upper_body"])
            error('gs3dx:ik:scope', 'Requested scope must be scalar and equal to ''auto'', ''whole_body'', or ''upper_body''.');
        end
    end

    % Extract and normalize BlockPaths to unique string array
    raw_paths = string(jp.BlockPath);
    unique_paths = unique(raw_paths(:));

    % Define lower-body roles
    roles = { ...
        'Left Hip Joint', ...
        'Right Hip Joint', ...
        'Left Knee', ...
        'Right Knee', ...
        'Left Ankle', ...
        'Right Ankle'};

    role_present = false(1, numel(roles));

    for i = 1:numel(roles)
        matches = contains(unique_paths, roles{i});
        n_match = sum(matches);
        if n_match > 1
            error('gs3dx:ik:scope', 'Ambiguous topology: duplicate unique block paths found for role "%s".', roles{i});
        elseif n_match == 1
            role_present(i) = true;
        end
    end

    n_present = sum(role_present);

    % Decide scope
    switch requested
        case "auto"
            if n_present == 6
                scope_name = "whole_body";
            elseif n_present == 0
                scope_name = "upper_body";
            else
                error('gs3dx:ik:scope', 'Partial lower-body topology detected (%d/6 roles present). Auto scope cannot resolve.', n_present);
            end

        case "whole_body"
            if n_present == 6
                scope_name = "whole_body";
            else
                error('gs3dx:ik:scope', 'Requested scope ''whole_body'' requires all 6 lower-body roles, but %d/6 were found.', n_present);
            end

        case "upper_body"
            if n_present == 0 || n_present == 6
                scope_name = "upper_body";
            else
                error('gs3dx:ik:scope', 'Requested scope ''upper_body'' requires a consistent lower-body topology (0 or 6 roles), but partial topology (%d/6) was detected.', n_present);
            end
    end

    scope.name = scope_name;
    if scope_name == "whole_body"
        scope.target_indices = 1:14;
    else
        scope.target_indices = [1, 8:14];
    end
end
