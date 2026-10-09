function hex = native_simscape_file_sha256(path)
%NATIVE_SIMSCAPE_FILE_SHA256 Hash complete bytes of a task-owned native file.
    arguments
        path (1,1) string
    end
    assert(isfile(path), 'MissingNativeFile: %s', path);
    fid = fopen(path, 'r');
    assert(fid > 0, 'NativeFileOpenFailed: %s', path);
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    bytes = fread(fid, inf, '*uint8');
    md = java.security.MessageDigest.getInstance('SHA-256');
    digest = typecast(md.digest(bytes), 'uint8');
    hex = lower(reshape(dec2hex(digest, 2).', 1, []));
end
