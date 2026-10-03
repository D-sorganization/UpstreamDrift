function gs3dx_assert_no_shadowing(info)
%GS3DX_ASSERT_NO_SHADOWING  Fail fast if any GS3DX file could shadow an original.
%
%   GS3DX_ASSERT_NO_SHADOWING(INFO) checks, for the folders in INFO (from
%   GS3DX_SETUP):
%     1. Every model file in models/ starts with the GS3DX_ prefix.
%     2. No model file in models/ uses an original model or subsystem name.
%     3. Every GS3DX model name resolves to exactly one file on the path.
%     4. No file anywhere under the exploratory root reuses an original name
%        (a stray copy there would silently win which()).
%
%   Raises gs3dx:shadowing with a precise message on the first violation.

    names = gs3dx_names();
    originals = [names.original_model, string(values(names.original_subsys))];

    files = [dir(fullfile(info.models_dir, '*.slx')); dir(fullfile(info.models_dir, '*.mdl'))];
    for k = 1:numel(files)
        [~, stem] = fileparts(files(k).name);
        if ~startsWith(stem, names.prefix)
            error('gs3dx:shadowing', ...
                'Model file %s in %s lacks the %s prefix.', files(k).name, ...
                info.models_dir, names.prefix);
        end
        if any(strcmp(stem, originals))
            error('gs3dx:shadowing', ...
                'Model file %s reuses an original model name.', files(k).name);
        end
        hits = which(stem, '-all');
        hits = hits(endsWith(hits, {'.slx', '.mdl'}));
        if numel(hits) ~= 1
            error('gs3dx:shadowing', ...
                'Model name %s resolves to %d files on the path:\n%s', ...
                stem, numel(hits), strjoin(hits, newline));
        end
    end

    stray = [dir(fullfile(info.root, '**', '*.slx')); dir(fullfile(info.root, '**', '*.mdl'))];
    for k = 1:numel(stray)
        [~, stem] = fileparts(stray(k).name);
        if any(strcmp(stem, originals))
            error('gs3dx:shadowing', ...
                'Stray copy of original model %s found at %s.', stem, stray(k).folder);
        end
    end
end
