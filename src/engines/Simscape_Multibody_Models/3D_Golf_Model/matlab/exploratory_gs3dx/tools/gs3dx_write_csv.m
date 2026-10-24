function gs3dx_write_csv(path, header, data)
%GS3DX_WRITE_CSV  Write a header row and numeric rows with LF line endings (#11709).
%
%   GS3DX_WRITE_CSV(PATH, HEADER, DATA) writes HEADER (a string row vector)
%   as the first CSV line, then one line per row of DATA (numeric, same
%   width as HEADER) at %.17g precision.  Opened in binary mode so line
%   endings are LF regardless of platform, matching every other GS3DX
%   fixture export.  Shared by EXPORT_GS3DX_GRF_FIXTURE and
%   EXPORT_GS3DX_RIGID_STANCE_FIXTURE (DRY).

    arguments
        path (1,1) string
        header (1,:) string
        data (:,:) double
    end
    assert(size(data, 2) == numel(header), 'gs3dx:write_csv', ...
        'Precondition: DATA has %d columns, HEADER has %d', size(data, 2), numel(header));
    fid = fopen(path, 'w');   % binary mode: LF line endings
    assert(fid > 0, 'gs3dx:write_csv', 'CannotOpen: %s', path);
    fprintf(fid, '%s\n', strjoin(header, ','));
    fmt = [char(strjoin(repmat("%.17g", 1, size(data, 2)), ',')), '\n'];
    assert(size(fmt, 1) == 1, 'gs3dx:write_csv', 'BadFormat: row format must be one char row');
    fprintf(fid, fmt, data.');
    fclose(fid);
end
