function digest = sha256_of_bytes(bytes)
%SHA256_OF_BYTES  SHA-256 hex digest of an in-memory uint8 vector.
%
%   DIGEST = SHA256_OF_BYTES(BYTES) returns a 64-character lowercase hex
%   string of the SHA-256 hash of BYTES. Used by synthetic-source
%   provenance (no file on disk to hash).
%
%   Falls back to a deterministic non-cryptographic fingerprint if the
%   Java MessageDigest is unavailable.
    arguments
        bytes (:,1) uint8
    end

    try
        md = java.security.MessageDigest.getInstance("SHA-256");
        md.update(bytes);
        raw = typecast(md.digest(), "uint8");
        digest = string(lower(reshape(dec2hex(raw, 2)', 1, [])));
    catch
        % Deterministic fallback (not collision-resistant, padded to 64).
        h = uint64(1469598103934665603); % FNV-1a 64 offset
        prime = uint64(1099511628211);
        for k = 1:numel(bytes)
            h = bitxor(h, uint64(bytes(k)));
            h = h * prime;
        end
        hex16 = lower(dec2hex(h, 16));
        digest = string([repmat('0', 1, 48), hex16]);
    end

    assert(strlength(digest) == 64, ...
        "sha256_of_bytes:badLength", ...
        "Postcondition: digest must be 64 hex chars");
end
