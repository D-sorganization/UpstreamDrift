function file = gs3dx_capture_original_baseline(info, opts)
%GS3DX_CAPTURE_ORIGINAL_BASELINE  Record the hand-built model's reference run.
%
%   FILE = GS3DX_CAPTURE_ORIGINAL_BASELINE(INFO) simulates the original
%   GolfSwing3D_Kinetic for the 0.3 s window and saves the flattened
%   CombinedSignalBus (double precision, 1 kHz) to
%   INFO.baselines_dir/original_GolfSwing3D_Kinetic_0p3s.mat.
%
%   Safety: the original folder goes on the path only for this call (an
%   onCleanup removes it), and the model is always closed with
%   close_system(...,0), so it can never be saved. The manifest hash is
%   checked before and after the run.

    arguments
        info (1,1) struct
        opts.stop_time (1,1) double {mustBePositive} = 0.3
    end
    names = gs3dx_names();
    mdl = char(names.original_model);
    before = gs3dx_original_manifest(info);

    addpath(info.original_model_dir);
    restore = onCleanup(@() local_restore(mdl, info.original_model_dir));
    assert(strcmpi(fileparts(which(mdl)), info.original_model_dir), 'gs3dx:baseline', ...
        'Precondition: %s resolves to %s, not the original folder', mdl, which(mdl));

    run = gs3dx_simulate(mdl, stop_time = opts.stop_time);
    assert(run.status == "success", 'gs3dx:baseline', 'Original run failed: %s', run.message);

    baseline = run;                 % double precision: float32 rounding alone
                                    % exceeded atol on constant signals (#10952)
    baseline.captured_on = string(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
    baseline.source_sha256 = before(1).sha256;
    file = fullfile(info.baselines_dir, sprintf('original_%s_%sS.mat', mdl, ...
        strrep(num2str(opts.stop_time), '.', 'p')));
    save(file, 'baseline', '-v7');

    clear restore
    after = gs3dx_original_manifest(info);
    assert(isequal({before.sha256}, {after.sha256}), 'gs3dx:baseline', ...
        'Postcondition violated: an original model file changed during capture');
end

function local_restore(mdl, folder)
    if bdIsLoaded(mdl)
        close_system(mdl, 0);
    end
    names = gs3dx_names();
    for r = values(names.original_subsys)
        if bdIsLoaded(r{1})
            close_system(r{1}, 0);
        end
    end
    rmpath(folder);
end
