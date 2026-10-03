function q = gs3dx_rom_from_ik(ik, rom)
%GS3DX_ROM_FROM_IK  The joint angles of a whole-body IK in ROM row order (#11158).
%
%   Q = GS3DX_ROM_FROM_IK(IK, ROM) picks, for each row of ROM
%   (GS3DX_JOINT_ROM), the IK joint (GS3DX_WHOLE_BODY_IK, IK.joint in deg)
%   with the row's joint key, and returns them as Q (height(ROM) x frames,
%   deg) for GS3DX_ROM_CHECK.  Rows whose joint the IK model lacks (the
%   neck and midfoot on GS3DX_Fit) are NaN.

    arguments
        ik (1,1) struct
        rom table
    end
    [keys, ids] = gs3dx_joint_keys(char(ik.model));
    [has, at] = ismember(rom.key, keys);
    [found, row] = ismember(ids(at(has)), string(ik.joint_ids));
    assert(all(found), 'gs3dx:rom_from_ik', 'IK of %s lacks joints of its own model', ik.model);
    q = nan(height(rom), size(ik.joint, 2));
    q(has, :) = ik.joint(row, :);
end
