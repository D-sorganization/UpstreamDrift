function app = gs3dx_read_appearance(file)
%GS3DX_READ_APPEARANCE  Read and validate the shared-spec appearance file (R9c, #12690).
%
%   APP = GS3DX_READ_APPEARANCE() reads appearance/gs3dx_appearance.v1.json,
%   written by scripts/export_simscape_appearance.py from the shared
%   model_appearance spec, the one source of every GS3DX colour and trunk
%   size.  APP = GS3DX_READ_APPEARANCE(FILE) reads another file.
%
%   Fails closed: a missing file, an unknown schema_version, a missing
%   field, a colour outside [0, 1] or a non-positive radius raises
%   'gs3dx:appearance'.  There are no default colours.
%
%   See also: GS3DX_BUILD_HUMAN, GS3DX_APPLY_APPEARANCE.

    arguments
        file (1,1) string = ""
    end
    version = "simscape-appearance/v1";
    if file == ""
        file = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'appearance', 'gs3dx_appearance.v1.json');
    end
    if ~isfile(file)
        error('gs3dx:appearance', ...
            'Appearance file %s is missing; run scripts/export_simscape_appearance.py', file);
    end
    app = jsondecode(fileread(file));
    if ~isfield(app, 'schema_version') || ~strcmp(app.schema_version, version)
        error('gs3dx:appearance', 'Unknown appearance schema_version in %s (want %s)', file, version);
    end
    for f = ["colours" "visuals" "recolour" "neck" "shoe_rgb"]
        if ~isfield(app, f)
            error('gs3dx:appearance', '%s has no field %s', file, f);
        end
    end
    rgbs = [{app.shoe_rgb}, {app.neck.rgb}, {app.visuals.rgb}, {app.recolour.rgb}];
    for k = 1:numel(rgbs)
        c = rgbs{k};
        if numel(c) ~= 3 || any(~isfinite(c)) || any(c < 0 | c > 1)
            error('gs3dx:appearance', 'Colour %s is not 3 values in [0, 1]', mat2str(c(:).'));
        end
    end
    for v = app.visuals(:).'
        if numel(v.radii_m) ~= 3 || any(~(v.radii_m > 0))
            error('gs3dx:appearance', '%s: radii must be 3 positive values', v.name);
        end
    end
end
