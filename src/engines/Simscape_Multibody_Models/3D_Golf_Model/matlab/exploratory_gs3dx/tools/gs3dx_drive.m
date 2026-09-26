function vars = gs3dx_drive(info, drive, mdl)
%GS3DX_DRIVE  Model-workspace overrides for a named regression drive.
%
%   VARS = GS3DX_DRIVE(INFO, DRIVE, MDL) returns the struct of model-workspace
%   variables that GS3DX_SIMULATE(MDL, variables=VARS) applies for DRIVE:
%
%     "persisted"  no overrides: the inputs saved in the model.  This run is
%                  ill-conditioned (clubhead > 4 km/s by 0.3 s; a 1e-9 RelTol
%                  change moves the clubhead 2.4 m), so it cannot separate a
%                  structural change from rounding noise.  See
%                  docs/SENSITIVITY_FINDINGS.md.
%     "impact"     3DModelInputs_Impact.mat from the original model folder:
%                  physical speeds, and a 1e-9 RelTol change moves the
%                  clubhead < 1e-10 m.  This is the regression drive.
%
%   Only variables that exist in MDL's model workspace are returned (the
%   input file also carries variables the model does not use).  MDL is
%   loaded if needed and never saved.  The input file is only read.

    arguments
        info (1,1) struct
        drive (1,1) string {mustBeMember(drive, ["persisted", "impact"])}
        mdl (1,:) char
    end
    vars = struct();
    if drive == "persisted"
        return;
    end
    file = fullfile(info.original_model_dir, 'inputs', '3DModelInputs_Impact.mat');
    assert(isfile(file), 'gs3dx:drive', 'Input file not found: %s', file);
    if ~bdIsLoaded(mdl)
        load_system(mdl);
    end
    ws = get_param(mdl, 'ModelWorkspace');
    in_model = {ws.whos.name};
    data = load(file);
    for f = reshape(intersect(fieldnames(data), in_model), 1, [])
        vars.(f{1}) = data.(f{1});
    end
    assert(~isempty(fieldnames(vars)), 'gs3dx:drive', ...
        'Postcondition: %s shares no variables with %s', file, mdl);
end
