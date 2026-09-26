function repointed = gs3dx_repoint_references(mdl, mapping)
%GS3DX_REPOINT_REFERENCES  Re-point ReferencedSubsystem blocks by name map.
%
%   REPOINTED = GS3DX_REPOINT_REFERENCES(MDL, MAPPING) sets every SubSystem
%   block in the loaded model MDL whose ReferencedSubsystem is a key of
%   MAPPING (containers.Map, old name -> new name) to the mapped name, in all
%   variants and under masks.  Returns a cellstr "block: old -> new" per
%   change.  The caller saves the model.
%
%   Postcondition: no block in MDL still references a key of MAPPING.

    arguments
        mdl (1,:) char
        mapping containers.Map
    end
    blocks = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'MatchFilter', @Simulink.match.allVariants, 'LookInsideSubsystemReference', 'off', ...
        'BlockType', 'SubSystem');
    repointed = {};
    for k = 1:numel(blocks)
        ref = get_param(blocks{k}, 'ReferencedSubsystem');
        if isKey(mapping, ref)
            set_param(blocks{k}, 'ReferencedSubsystem', mapping(ref));
            repointed{end+1} = sprintf('%s: %s -> %s', blocks{k}, ref, mapping(ref)); %#ok<AGROW>
        end
    end
    for k = 1:numel(blocks)
        ref = get_param(blocks{k}, 'ReferencedSubsystem');
        assert(~isKey(mapping, ref), 'gs3dx:repoint', ...
            'Postcondition: %s still references %s', blocks{k}, ref);
    end
end
