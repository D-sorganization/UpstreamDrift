function gs3dx_copy_models(pairs, overwrite, err_id)
%GS3DX_COPY_MODELS  Copy .slx files, refusing to replace existing targets.
%
%   GS3DX_COPY_MODELS(PAIRS, OVERWRITE, ERR_ID) copies each PAIRS{k,1} file
%   to PAIRS{k,2} (full .slx paths) with COPYFILE, so the source is only
%   read, never loaded.  Every source must exist, and unless OVERWRITE is
%   true no target may exist; both are checked before anything is copied.
%   A target already loaded in Simulink is closed without saving first.
%   Errors use identifier ERR_ID.

    arguments
        pairs (:,2) cell
        overwrite (1,1) logical
        err_id (1,:) char
    end
    for k = 1:size(pairs, 1)
        assert(isfile(pairs{k, 1}), err_id, 'Source not found: %s', pairs{k, 1});
        if isfile(pairs{k, 2}) && ~overwrite
            error(err_id, '%s exists; pass overwrite=true to replace it.', pairs{k, 2});
        end
    end
    for k = 1:size(pairs, 1)
        [~, stem] = fileparts(pairs{k, 2});
        if bdIsLoaded(stem)
            close_system(stem, 0);
        end
        copyfile(pairs{k, 1}, pairs{k, 2}, 'f');
        fileattrib(pairs{k, 2}, '+w');
    end
    rehash;
end
