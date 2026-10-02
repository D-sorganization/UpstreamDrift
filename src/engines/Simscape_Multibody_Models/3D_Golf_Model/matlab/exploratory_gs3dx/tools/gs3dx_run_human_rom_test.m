function gs3dx_run_human_rom_test()
%GS3DX_RUN_HUMAN_ROM_TEST  Run human_references_stay_in_the_human_range only.

    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(root);
    info = gs3dx_setup();
    addpath(fullfile(info.root, 'tests'));
    results = runtests('test_gs3dx_joint_rom/human_references_stay_in_the_human_range');
    nfail = sum([results.Failed]);
    fprintf('TEST failures %d\n', nfail);
    fprintf('GS3DX_BATCH_DONE\n');
    if nfail > 0
        quit(1, 'force');
    end
    quit(0, 'force');
end
