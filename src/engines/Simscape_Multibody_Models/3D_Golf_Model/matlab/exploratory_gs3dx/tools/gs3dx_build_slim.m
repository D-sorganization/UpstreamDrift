function report = gs3dx_build_slim(info, opts)
%GS3DX_BUILD_SLIM  Build GS3DX_Slim: GS3DX_Baseline with direct joint torque drive.
%
%   REPORT = GS3DX_BUILD_SLIM(INFO) copies GS3DX_Baseline to GS3DX_Slim and
%   each GS3DX_KD_* subsystem to GS3DX_KDS_*, rewires every torque axis in
%   the KDS copies with GS3DX_DIRECT_TORQUE_DRIVE, re-points GS3DX_Slim at the
%   KDS subsystems and saves everything through GS3DX_SAVE_MODEL (#10954).
%
%   GS3DX_Baseline and GS3DX_KD_* are only read with COPYFILE, so the
%   verified clone is never modified.  Existing slim files are never
%   replaced unless overwrite=true.
%
%   REPORT fields: .axes_rewired (map role -> count), .repointed (cellstr).

    arguments
        info (1,1) struct
        opts.overwrite (1,1) logical = false
    end
    names = gs3dx_names();
    pairs = {char(names.variants.baseline), char(names.variants.slim)};
    for r = cellstr(names.roles)
        pairs(end+1, :) = {names.clone_subsys(r{1}), names.slim_subsys(r{1})}; %#ok<AGROW>
    end
    target = @(stem) fullfile(info.models_dir, [stem '.slx']);
    for k = 1:size(pairs, 1)
        if isfile(target(pairs{k, 2})) && ~opts.overwrite
            error('gs3dx:slim', '%s exists; pass overwrite=true to rebuild it.', target(pairs{k, 2}));
        end
    end
    for k = 1:size(pairs, 1)
        if bdIsLoaded(pairs{k, 2})
            close_system(pairs{k, 2}, 0);
        end
        copyfile(target(pairs{k, 1}), target(pairs{k, 2}), 'f');
        fileattrib(target(pairs{k, 2}), '+w');
    end
    rehash;

    report = struct('axes_rewired', containers.Map(), 'repointed', {{}});
    for r = cellstr(names.roles)
        sub = names.slim_subsys(r{1});
        load_system(sub);
        report.axes_rewired(r{1}) = gs3dx_direct_torque_drive(sub);
        gs3dx_save_model(sub, info);
        close_system(sub, 0);
    end

    slim = char(names.variants.slim);
    load_system(slim);
    cleanup = onCleanup(@() close_system(slim, 0));
    report.repointed = gs3dx_repoint_references(slim, ...
        containers.Map(values(names.clone_subsys), values(names.slim_subsys)));
    gs3dx_save_model(slim, info);
end
