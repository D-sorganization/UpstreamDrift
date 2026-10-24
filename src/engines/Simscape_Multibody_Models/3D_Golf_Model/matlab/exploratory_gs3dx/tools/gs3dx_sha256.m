function hex = gs3dx_sha256(path)
%GS3DX_SHA256  Lowercase hex SHA-256 digest of a file's bytes (#11709).
%
%   HEX = GS3DX_SHA256(PATH) reads PATH as raw bytes and returns its
%   SHA-256 digest (java.security.MessageDigest; no extra toolbox).  Shared
%   by EXPORT_GS3DX_GRF_FIXTURE and EXPORT_GS3DX_RIGID_STANCE_FIXTURE so
%   every GS3DX receipt hashes its fixture the same way (DRY).

    arguments
        path (1,1) string
    end
    fid = fopen(path, 'r');
    assert(fid > 0, 'gs3dx:sha256', 'CannotOpen: %s', path);
    bytes = fread(fid, inf, '*uint8');
    fclose(fid);
    md = java.security.MessageDigest.getInstance('SHA-256');
    digest = typecast(md.digest(bytes), 'uint8');
    hex = lower(reshape(dec2hex(digest, 2).', 1, []));
end
