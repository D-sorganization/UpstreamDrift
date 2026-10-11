function hex = native_simscape_bytes_sha256(bytes)
%NATIVE_SIMSCAPE_BYTES_SHA256 Hash exact byte order without text recoding.
    md = java.security.MessageDigest.getInstance('SHA-256');
    digest = typecast(md.digest(uint8(bytes)), 'uint8');
    hex = lower(reshape(dec2hex(digest, 2).', 1, []));
end
