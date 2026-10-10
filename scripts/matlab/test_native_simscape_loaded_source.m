function receipt = test_native_simscape_loaded_source(producer)
%TEST_NATIVE_SIMSCAPE_LOADED_SOURCE Bind loaded model to its requested SLX path.
%   A name-shadowing warning is insufficient evidence of the loaded source.
%   The source path must be checked before compilation/simulation.
    arguments
        producer (1,1) string
    end
    assert(strcmp(version('-release'), '2025b'), 'NativeOwned:R2025bRequired');
    mdl = "native_restart_fixture_11921";
    path = fullfile(producer, mdl + ".slx");
    assert(~bdIsLoaded(mdl), 'NativeOwned:ModelAlreadyLoaded');
    load_system(path);
    cleanup = onCleanup(@() close_system(mdl, 0)); %#ok<NASGU>
    rows = readmatrix(fullfile(producer, 'saved-input.csv'));
    input = timeseries(rows(:,2), rows(:,1));
    valid = capture_native_simscape_execution(mdl, input, path);
    wrong = fullfile(producer, "different", mdl + ".slx");
    observed = '';
    try
        capture_native_simscape_execution(mdl, input, wrong);
    catch exception
        observed = exception.identifier;
    end
    assert(strcmp(observed, 'NativeOwned:LoadedModelPathMismatch'), ...
        'NativeOwned:LoadedModelPathRejectionMissing', ...
        'A different requested SLX path must be refused before compilation');
    receipt = struct('issue', '#11942', 'runtime_id', version, ...
        'loaded_filename', get_param(mdl, 'FileName'), ...
        'valid_configuration_sha256', valid.effective_configuration_sha256, ...
        'wrong_path_identifier', observed);
    disp(jsonencode(receipt, 'PrettyPrint', true));
end
