function manifest = gs3dx_original_manifest(info)
%GS3DX_ORIGINAL_MANIFEST  SHA-256 of every hand-built original model file.
%
%   MANIFEST = GS3DX_ORIGINAL_MANIFEST(INFO) returns a struct array with
%   fields .file and .sha256 for GolfSwing3D_Kinetic.slx and the three
%   Kinetically_Driven_* subsystems. Tests compare it before and after the
%   GS3DX tooling runs, to prove that the originals were never modified.

    names = gs3dx_names();
    stems = [names.original_model, string(values(names.original_subsys))];
    manifest = struct('file', {}, 'sha256', {});
    for k = 1:numel(stems)
        f = fullfile(info.original_model_dir, char(stems(k)) + ".slx");
        manifest(end+1) = struct('file', char(f), 'sha256', local_sha256(f)); %#ok<AGROW>
    end
end

function hex = local_sha256(file)
    fid = fopen(file, 'r');
    assert(fid > 0, 'gs3dx:manifest', 'Cannot open %s', file);
    bytes = fread(fid, Inf, '*uint8');
    fclose(fid);
    md = java.security.MessageDigest.getInstance('SHA-256');
    digest = typecast(md.digest(bytes), 'uint8');
    hex = lower(reshape(dec2hex(digest, 2).', 1, []));
end
