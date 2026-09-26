function info = gs3dx_setup()
%GS3DX_SETUP  Put the exploratory GS3DX workspace on the MATLAB path safely.
%
%   INFO = GS3DX_SETUP() adds exploratory_gs3dx/, its tools/ and models/
%   folders to the path for this session only, redirects the Simulink cache
%   and code-generation folders to tempdir, and asserts that no GS3DX model
%   shadows (or is shadowed by) the hand-built originals.
%
%   Safety contract (epic #10950):
%     - Never calls SAVEPATH; the path change lasts for this session only.
%     - Never adds the original model folder (matlab/src/model) to the path;
%       only matlab/src/functions (model-called MATLAB functions) is added.
%     - Returns absolute folder locations so callers never hard-code paths.
%
%   Postconditions:
%     - info.root, info.models_dir, info.tools_dir, info.original_model_dir,
%       info.baselines_dir and info.cache_dir are existing folders.
%     - gs3dx_assert_no_shadowing() has passed.
%
%   See also: GS3DX_ASSERT_NO_SHADOWING, GS3DX_CLONE_BASELINE.

    root = fileparts(mfilename('fullpath'));
    info = struct();
    info.root               = root;
    info.tools_dir          = fullfile(root, 'tools');
    info.models_dir         = fullfile(root, 'models');
    info.baselines_dir      = fullfile(root, 'baselines');
    info.docs_dir           = fullfile(root, 'docs');
    info.original_model_dir = fullfile(fileparts(root), 'src', 'model');
    info.cache_dir          = fullfile(tempdir, 'gs3dx_slcache');
    % MATLAB functions the model's Stateflow/MATLAB Function blocks call by
    % name (HexPolyInputFunction).  Without it the Hip Torque Output chart
    % fails to size its outputs at compile time.
    info.dependency_dirs    = {fullfile(fileparts(root), 'src', 'functions')};

    addpath(root, info.tools_dir, info.models_dir, info.dependency_dirs{:});

    for d = {info.cache_dir, info.baselines_dir, info.docs_dir}
        if ~isfolder(d{1})
            mkdir(d{1});
        end
    end
    Simulink.fileGenControl('set', 'CacheFolder', info.cache_dir, ...
        'CodeGenFolder', info.cache_dir, 'createDir', true);

    gs3dx_assert_no_shadowing(info);

    required = {'root', 'tools_dir', 'models_dir', 'baselines_dir', ...
                'original_model_dir', 'cache_dir'};
    for k = 1:numel(required)
        assert(isfolder(info.(required{k})), 'gs3dx_setup:missingFolder', ...
            'Postcondition: %s does not exist: %s', required{k}, info.(required{k}));
    end
end
