function receipt = export_simscape_urdf_exchange(repo, out_dir)
%EXPORT_SIMSCAPE_URDF_EXCHANGE  R2025b URDF exchange receipt (#11569 task 3).
%
%   RECEIPT = EXPORT_SIMSCAPE_URDF_EXCHANGE(REPO, OUT_DIR) records, headless:
%     - whether smexport exists in this release (it does not in R2025b, so
%       the canonical model's tree is read from its blocks instead);
%     - the joint and mass inventory of the canonical GolfSwing3D_Kinetic;
%     - an smimport of the spec URDF (src/engines/physics_engines/pinocchio/
%       models/generated/golfer.urdf, built by the #9965 exporter) and the
%       same inventory of the imported model, so both sides are read by one
%       routine (simscape_model_inventory).
%   Writes OUT_DIR/simscape_urdf_exchange_receipt.json. The Python side
%   (src/engines/simscape/urdf_exchange.py) diffs it against the URDF.
%
%   Preconditions: MATLAB R2025b (asserted); REPO is an UpstreamDrift
%   checkout; OUT_DIR is writable.
%   Postconditions: no model file, MATLAB path or preference is saved.

    arguments
        repo (1,1) string
        out_dir (1,1) string
    end
    rel = version('-release');
    assert(strcmp(rel, '2025b'), 'R2025bRequired: detected %s', rel);
    if ~isfolder(out_dir); mkdir(out_dir); end
    matlab_root = fullfile(repo, 'src', 'engines', 'Simscape_Multibody_Models', ...
        '3D_Golf_Model', 'matlab');
    addpath(genpath(fullfile(matlab_root, 'src')));
    addpath(fileparts(mfilename('fullpath')));
    cache = fullfile(tempdir, 'ud_urdf_exchange_cache');
    Simulink.fileGenControl('set', 'CacheFolder', cache, 'CodeGenFolder', cache, ...
        'createDir', true);

    model = 'GolfSwing3D_Kinetic';
    model_file = fullfile(matlab_root, 'src', 'model', [model '.slx']);
    load_system(model_file);
    canonical = struct('model', model, ...
        'model_file', 'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/GolfSwing3D_Kinetic.slx', ...
        'model_sha256', local_sha256(model_file), ...
        'inventory', simscape_model_inventory(model));
    close_system(model, 0);

    urdf_rel = 'src/engines/physics_engines/pinocchio/models/generated/golfer.urdf';
    urdf = fullfile(repo, urdf_rel);
    spec = struct('urdf_file', urdf_rel, 'urdf_sha256', local_sha256(urdf), ...
        'smimport_ok', false, 'smimport_error', '', 'inventory', struct());
    imported = 'ud_spec_urdf_import';
    try
        h = smimport(urdf, 'ModelName', imported);
        spec.smimport_ok = true;
        spec.inventory = simscape_model_inventory(get_param(h, 'Name'));
        close_system(h, 0);
    catch err
        spec.smimport_error = err.message;
        if bdIsLoaded(imported); close_system(imported, 0); end
    end

    receipt = struct( ...
        'schema', 'simscape-urdf-exchange/v1', 'issue', 11569, ...
        'matlab_release', rel, 'matlab_version', version, ...
        'smexport_available', exist('smexport', 'file') > 0, ...
        'canonical', canonical, 'spec_urdf', spec);
    path = fullfile(out_dir, 'simscape_urdf_exchange_receipt.json');
    fid = fopen(path, 'w');
    assert(fid > 0, 'OpenFailed: %s', path);
    fwrite(fid, jsonencode(receipt, 'PrettyPrint', true), 'char');
    fclose(fid);
    fprintf('wrote %s\n', path);
    fprintf('canonical: %d joints, %d coordinates; spec import ok=%d\n', ...
        numel(canonical.inventory.joints), canonical.inventory.n_coordinates, ...
        spec.smimport_ok);
end

function h = local_sha256(file)
    fid = fopen(file, 'r');
    assert(fid > 0, 'OpenFailed: %s', file);
    bytes = fread(fid, Inf, '*uint8');
    fclose(fid);
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(bytes);
    h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2).', 1, []));
end
