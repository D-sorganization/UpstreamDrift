function blocks = gs3dx_track_blocks(jp, spec)
%GS3DX_TRACK_BLOCKS  Joint block of each upper-body chart (#10979, #11173).
%
%   BLOCKS = GS3DX_TRACK_BLOCKS(JP, SPEC) is a cell array, one joint block
%   path per element of SPEC (GS3DX_UPPER_BODY_JOINTS), found from the
%   joint ids of JP, a KinematicsSolver's jointPositionVariables table.
%   Used by GS3DX_TRACK_LEARN and GS3DX_CONTACT_CHECK.

    blocks = cell(1, numel(spec));
    for k = 1:numel(spec)
        b = unique(string(jp.BlockPath(startsWith(string(jp.ID), [spec(k).id '.']))));
        assert(isscalar(b), 'gs3dx:trackblocks', 'No single joint block for %s', spec(k).id);
        blocks{k} = char(b);
    end
end
