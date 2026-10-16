function applied = gs3dx_apply_appearance(mdl, app)
%GS3DX_APPLY_APPEARANCE  Restyle a built GS3DX_Human from the shared appearance spec (R9c, #12690).
%
%   APPLIED = GS3DX_APPLY_APPEARANCE(MDL) sets the radii and colour of the
%   ellipsoids GS3DX_BUILD_HUMAN drew, and the colour of the head, hands and
%   legs, from GS3DX_READ_APPEARANCE, so a model built before the shared spec
%   (or from an older file) matches it without a rebuild.  It does not save.
%   APPLIED lists the block paths it set.
%
%   Graphics only: it touches EllipsoidRadii and GraphicDiffuseColor, never a
%   mass, inertia, frame or joint.  A block the file names that the model
%   lacks raises 'gs3dx:appearance'.
%
%   See also: GS3DX_READ_APPEARANCE, GS3DX_BUILD_HUMAN.

    arguments
        mdl (1,1) string
        app (1,1) struct = gs3dx_read_appearance()
    end
    applied = strings(1, 0);
    solids = string(find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'ReferenceBlock', 'sm_lib/Body Elements/Ellipsoidal Solid'));
    for v = app.visuals(:).'
        blk = solids(endsWith(solids, "/" + string(v.name)));
        if numel(blk) ~= 1
            error('gs3dx:appearance', '%s: expected one ellipsoid %s, found %d', mdl, v.name, numel(blk));
        end
        set_param(blk, 'EllipsoidRadii', mat2str(v.radii_m(:).', 6), 'EllipsoidRadiiUnits', 'm', ...
            'GraphicDiffuseColor', mat2str(v.rgb(:).'));
        applied(end + 1) = blk; %#ok<AGROW>
    end
    % Neck width follows the model's NeckLength, so only its colour is set.
    colours = {"Neck Shape", app.neck.rgb};
    for s = ["L" "R"]
        colours(end + 1, :) = {s + " Rearfoot", app.shoe_rgb}; %#ok<AGROW>
        colours(end + 1, :) = {s + " Forefoot", app.shoe_rgb}; %#ok<AGROW>
    end
    for k = 1:size(colours, 1)
        blk = solids(endsWith(solids, "/" + colours{k, 1}));
        if numel(blk) ~= 1
            error('gs3dx:appearance', '%s: expected one ellipsoid %s, found %d', mdl, colours{k, 1}, numel(blk));
        end
        set_param(blk, 'GraphicDiffuseColor', mat2str(colours{k, 2}(:).'));
        applied(end + 1) = blk; %#ok<AGROW>
    end
    for r = app.recolour(:).'
        blk = mdl + "/" + string(r.block);
        set_param(blk, 'GraphicDiffuseColor', mat2str(r.rgb(:).'));
        applied(end + 1) = blk; %#ok<AGROW>
    end
end
