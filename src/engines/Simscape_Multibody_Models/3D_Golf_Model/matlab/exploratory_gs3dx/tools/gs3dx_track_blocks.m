function blocks = gs3dx_track_blocks(jp, spec)
%GS3DX_TRACK_BLOCKS  Joint block of each upper-body chart (#10979, #11173).
%
%   BLOCKS = GS3DX_TRACK_BLOCKS(JP, SPEC) is a cell array, one joint block
%   path per element of SPEC (GS3DX_UPPER_BODY_JOINTS), found by its path
%   below the model among the joints of JP, a KinematicsSolver's
%   jointPositionVariables table.  Joint ids are not used: they differ
%   between variants (GS3DX_Human's neck renumbers every joint after it).
%   Used by GS3DX_TRACK_LEARN and GS3DX_CONTACT_CHECK.

    blocks = cell(1, numel(spec));
    paths = string(jp.BlockPath);
    for k = 1:numel(spec)
        b = unique(paths(endsWith(paths, "/" + spec(k).block)));
        assert(isscalar(b), 'gs3dx:trackblocks', 'No single joint block %s for %s', spec(k).block, spec(k).prefix);
        blocks{k} = char(b);
    end
end
