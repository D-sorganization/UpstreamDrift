function q = gs3dx_rom_reference(ws, rom)
%GS3DX_ROM_REFERENCE  The joint-angle references of a GS3DX model workspace in ROM row order (#11158).
%
%   Q = GS3DX_ROM_REFERENCE(WS, ROM) reads, for each row of ROM
%   (GS3DX_JOINT_ROM), row ROM.track_row of the model-workspace variable
%   ROM.track of WS (a Simulink.ModelWorkspace) and returns them as Q
%   (height(ROM) x N, deg): '<J>TrackAngle' and 'LegReferenceAngle' are in
%   deg, 'NeckReference' in rad.  A row whose reference is absent from WS,
%   or that has none, is NaN.

    arguments
        ws
        rom table
    end
    n = numel(ws.getVariable('LegReferenceTime'));
    q = nan(height(rom), n);
    for k = find(rom.track_row(:).' > 0)
        name = char(rom.track(k));
        if ~hasVariable(ws, name)
            continue
        end
        v = ws.getVariable(name);
        assert(size(v, 2) == n && rom.track_row(k) <= size(v, 1), 'gs3dx:rom_reference', ...
            '%s is %s, expected row %d of %d frames', name, mat2str(size(v)), rom.track_row(k), n);
        q(k, :) = v(rom.track_row(k), :);
        if strcmp(name, 'NeckReference')
            q(k, :) = rad2deg(q(k, :));
        end
    end
end
