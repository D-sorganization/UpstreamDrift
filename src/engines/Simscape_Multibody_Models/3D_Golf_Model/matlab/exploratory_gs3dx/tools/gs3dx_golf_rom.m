function rom = gs3dx_golf_rom(rom)
%GS3DX_GOLF_ROM  The golf-swing band inside the human range of motion (#11156).
%
%   ROM = GS3DX_GOLF_ROM() is GS3DX_JOINT_ROM with the rows a golf swing
%   holds tighter than the anatomical range narrowed to the swing's band;
%   GS3DX_GOLF_ROM(ROM) narrows a given ROM table the same way.  Pass it to
%   GS3DX_WHOLE_BODY_IK ('rom') and GS3DX_ROM_CHECK like the anatomical
%   table.
%
%   Bands:
%     LE  lead elbow flexion at most 20 deg: a skilled golfer keeps the lead
%         arm close to straight from address through impact (#11156 target;
%         literature source to confirm before it is tightened further)
    arguments
        rom table = gs3dx_joint_rom()
    end
    band = {
    %   joint  motion                max_deg  source
        'LE',  "lead elbow flexion",  20,     "#11156 golf band: lead arm near straight address to impact"
        };
    for k = 1:size(band, 1)
        at = find(rom.joint == band{k, 1} & rom.motion == band{k, 2});
        assert(isscalar(at), 'gs3dx:golfrom', 'No single ROM row %s (%s)', band{k, 1}, band{k, 2});
        assert(band{k, 3} > 0 && band{k, 3} < rom.max_deg(at), 'gs3dx:golfrom', ...
            'The %s band must lie inside the anatomical range', band{k, 1});
        rom.max_deg(at) = band{k, 3};
        rom.span_deg(at) = rom.max_deg(at) - rom.min_deg(at);
        rom.source(at) = band{k, 4};
    end
end
