%GS3DX_EXPORT_BATCH Environment-configured entry for run_matlab_locked.ps1.
% Required: GS3DX_CAPTURE_ID, GS3DX_OUTPUT_DIR.
% Optional: CAPTURE_REGISTRY_REPO, MATLAB_PYTHON_EXE, CAPTURE_DATA_DIR.
% No raw capture data or machine-specific paths are written to the repository.
try
    set(groot, 'defaultFigureVisible', 'off');
    python_exe = getenv('MATLAB_PYTHON_EXE');
    if ~isempty(python_exe)
        pyenv('Version', python_exe, 'ExecutionMode', 'OutOfProcess');
    end
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(root);
    gs3dx_setup();
    capture_id = string(getenv('GS3DX_CAPTURE_ID'));
    output_dir = string(getenv('GS3DX_OUTPUT_DIR'));
    registry_repo = string(getenv('CAPTURE_REGISTRY_REPO'));
    assert(strlength(capture_id) > 0 && strlength(output_dir) > 0, ...
        'gs3dx:export_batch', 'Set GS3DX_CAPTURE_ID and GS3DX_OUTPUT_DIR explicitly');
    report = gs3dx_match_export(capture_id, output_dir=output_dir, registry_repo=registry_repo);
    fprintf('EXPORT_RECEIPT %s\n', report.provenance_file);
    fprintf('STATUS success\nGS3DX_BATCH_DONE\n');
    python_env = pyenv;
    if python_env.Status == "Loaded" && python_env.ExecutionMode == "OutOfProcess"
        terminate(python_env);
    end
    quit(0, 'force');
catch err
    fprintf('%s\n', getReport(err, 'extended', 'hyperlinks', 'off'));
    fprintf('STATUS failed\n');
    quit(1, 'force');
end
