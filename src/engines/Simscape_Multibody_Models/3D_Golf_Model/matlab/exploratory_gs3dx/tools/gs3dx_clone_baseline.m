function report = gs3dx_clone_baseline(info, opts)
%GS3DX_CLONE_BASELINE  Create GS3DX_Baseline as a renamed, self-contained copy.
%
%   REPORT = GS3DX_CLONE_BASELINE(INFO) copies the hand-built
%   GolfSwing3D_Kinetic.slx and its three Kinetically_Driven_* referenced
%   subsystems from INFO.original_model_dir into INFO.models_dir under GS3DX_
%   names (see GS3DX_NAMES). Then it re-points every ReferencedSubsystem in the
%   clone to the GS3DX copies and saves the result.
%
%   REPORT = GS3DX_CLONE_BASELINE(INFO, overwrite=true) replaces existing
%   clones. The default is false, so an existing (possibly edited) clone is
%   never clobbered by accident.
%
%   The originals are only ever read with COPYFILE; they are never loaded, so
%   no Simulink state can dirty or re-save them.
%
%   Postconditions:
%     - The GS3DX_Baseline top model and all GS3DX_KD_* files exist.
%     - No block in GS3DX_Baseline references an original subsystem name.
%     - report.repointed lists every re-pointed block.

    arguments
        info (1,1) struct
        opts.overwrite (1,1) logical = false
    end
    names = gs3dx_names();
    top = char(names.variants.baseline);

    pairs = {};   % {source, destination}
    for r = cellstr(names.roles)
        pairs(end+1, :) = { ...
            fullfile(info.original_model_dir, [names.original_subsys(r{1}) '.slx']), ...
            fullfile(info.models_dir, [names.clone_subsys(r{1}) '.slx'])}; %#ok<AGROW>
    end
    pairs(end+1, :) = { ...
        fullfile(info.original_model_dir, [char(names.original_model) '.slx']), ...
        fullfile(info.models_dir, [top '.slx'])};

    for k = 1:size(pairs, 1)
        [src, dst] = pairs{k, :};
        assert(isfile(src), 'gs3dx:clone', 'Original not found: %s', src);
        if isfile(dst) && ~opts.overwrite
            error('gs3dx:clone', '%s exists; pass overwrite=true to replace it.', dst);
        end
        [~, stem] = fileparts(dst);
        if bdIsLoaded(stem)
            close_system(stem, 0);
        end
        copyfile(src, dst, 'f');
        fileattrib(dst, '+w');
    end
    rehash;

    % Referenced subsystems still point at the original names in the fresh
    % copy.  Load without resolving them and re-point.
    load_system(fullfile(info.models_dir, [top '.slx']));
    cleanup = onCleanup(@() close_system(top, 0));
    blocks = find_system(top, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'MatchFilter', @Simulink.match.allVariants, 'BlockType', 'SubSystem');
    report = struct('model', top, 'repointed', {{}});
    reverse = containers.Map(values(names.original_subsys), values(names.clone_subsys));
    for k = 1:numel(blocks)
        ref = get_param(blocks{k}, 'ReferencedSubsystem');
        if isKey(reverse, ref)
            set_param(blocks{k}, 'ReferencedSubsystem', reverse(ref));
            report.repointed{end+1} = sprintf('%s: %s -> %s', blocks{k}, ref, reverse(ref));
        end
    end
    gs3dx_save_model(top, info);

    % Postcondition: no remaining reference to an original subsystem.
    for k = 1:numel(blocks)
        ref = get_param(blocks{k}, 'ReferencedSubsystem');
        assert(~isKey(reverse, ref), 'gs3dx:clone', ...
            'Postcondition: %s still references original %s', blocks{k}, ref);
    end
end
