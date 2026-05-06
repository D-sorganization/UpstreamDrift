function digest = sha256_of_file(file_path)
%SHA256_OF_FILE  Compute the SHA-256 hex digest of a file.
%
%   DIGEST = SHA256_OF_FILE(FILE_PATH) returns a 64-character lowercase hex
%   string of the SHA-256 hash of the file at FILE_PATH. Used by the loader
%   provenance check (CLUB_IK_SPEC.md, validation rule 6).
%
%   Preconditions:
%     - FILE_PATH must be a string/char of an existing file.
%
%   Postconditions:
%     - DIGEST is a 1x64 char array of lowercase hex digits.
    arguments
        file_path (1,1) string {mustBeFile}
    end

    md = java.security.MessageDigest.getInstance("SHA-256");
    fid = fopen(file_path, "r");
    cleaner = onCleanup(@() fclose(fid));
    if fid < 0
        error("sha256_of_file:cannotOpen", ...
              "Cannot open file: %s", file_path);
    end

    chunk_size = 65536;
    while ~feof(fid)
        bytes = fread(fid, chunk_size, "*uint8");
        if ~isempty(bytes)
            md.update(bytes);
        end
    end

    raw = typecast(md.digest(), "uint8");
    digest = lower(reshape(dec2hex(raw, 2)', 1, []));

    assert(ischar(digest) && numel(digest) == 64, ...
        "Postcondition: SHA-256 digest must be 64 hex chars");
end
