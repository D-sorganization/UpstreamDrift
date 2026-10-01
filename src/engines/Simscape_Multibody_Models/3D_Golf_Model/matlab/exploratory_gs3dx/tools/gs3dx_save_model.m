function target = gs3dx_save_model(mdl, info)
%GS3DX_SAVE_MODEL  The only sanctioned way for GS3DX tools to write a model.
%
%   TARGET = GS3DX_SAVE_MODEL(MDL, INFO) saves the loaded block diagram MDL
%   to INFO.models_dir/<MDL>.slx and returns the absolute file path.
%
%   Preconditions (enforced, epic #10950 safety rules):
%     - MDL starts with the GS3DX_ prefix.
%     - The file currently backing MDL (if any) lives in INFO.models_dir, so
%       a hand-built original can never be overwritten through this helper.

    arguments
        mdl  (1,:) char
        info (1,1) struct
    end
    names = gs3dx_names();
    if ~startsWith(mdl, names.prefix)
        error('gs3dx:unsafeSave', 'Refusing to save %s: missing %s prefix.', mdl, names.prefix);
    end
    target = fullfile(info.models_dir, [mdl '.slx']);
    current = get_param(mdl, 'FileName');
    if ~isempty(current) && ~strcmpi(fileparts(current), info.models_dir)
        error('gs3dx:unsafeSave', ...
            'Refusing to save %s: it is backed by %s, outside %s.', mdl, current, info.models_dir);
    end
    if isempty(current) || ~strcmpi(current, target)
        save_system(mdl, target);
    else
        save_system(mdl);
    end
end
