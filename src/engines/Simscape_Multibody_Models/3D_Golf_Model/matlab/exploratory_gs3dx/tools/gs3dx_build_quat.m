function report = gs3dx_build_quat(info, opts)
%GS3DX_BUILD_QUAT  Build GS3DX_Quat: GS3DX_Slim with quaternion (Spherical) shoulders.
%
%   REPORT = GS3DX_BUILD_QUAT(INFO) copies GS3DX_Slim to GS3DX_Quat and
%   GS3DX_KDS_Gimbal to GS3DX_KDS_Spherical, swaps the copy's Gimbal Joint
%   for a Spherical Joint with GS3DX_GIMBAL_TO_SPHERICAL, re-points
%   GS3DX_Quat's Gimbal references (the two shoulders) at it and saves
%   through GS3DX_SAVE_MODEL (#10955).
%
%   GS3DX_Slim and GS3DX_KDS_Gimbal are only read with COPYFILE.  Existing
%   quaternion files are never replaced unless overwrite=true.
%
%   REPORT field: .repointed (cellstr, one entry per re-pointed block).

    arguments
        info (1,1) struct
        opts.overwrite (1,1) logical = false
    end
    names = gs3dx_names();
    gimbal = names.slim_subsys('Gimbal');
    pairs = {char(names.variants.slim), char(names.variants.quat); ...
             gimbal, names.spherical_subsys};
    files = cellfun(@(stem) fullfile(info.models_dir, [stem '.slx']), pairs, 'UniformOutput', false);
    gs3dx_copy_models(files, opts.overwrite, 'gs3dx:quat');

    load_system(names.spherical_subsys);
    gs3dx_gimbal_to_spherical(names.spherical_subsys);
    gs3dx_save_model(names.spherical_subsys, info);
    close_system(names.spherical_subsys, 0);

    quat = char(names.variants.quat);
    load_system(quat);
    cleanup = onCleanup(@() close_system(quat, 0));
    report.repointed = gs3dx_repoint_references(quat, ...
        containers.Map({gimbal}, {names.spherical_subsys}));
    gs3dx_save_model(quat, info);
end
