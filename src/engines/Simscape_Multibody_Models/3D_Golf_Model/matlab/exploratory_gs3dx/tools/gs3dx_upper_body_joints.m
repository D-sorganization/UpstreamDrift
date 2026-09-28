function spec = gs3dx_upper_body_joints()
%GS3DX_UPPER_BODY_JOINTS  The twelve upper-body joints tracked from the capture (#10979).
%
%   SPEC = GS3DX_UPPER_BODY_JOINTS() is a struct array, one per upper-body
%   '<prefix> Input Function' chart of GS3DX_Fit and its variants:
%     .prefix  chart prefix ('Spine', 'LS', ...)
%     .id      KinematicsSolver joint id ('j7'; the joint position
%              variables start with it)
%     .axes    '' (revolute Rz), 'XY' (universal Rx, Ry) or 'XYZ'
%              (spherical S, read as intrinsic X-Y-Z angles)

    c = {'Spine', 'j2', 'XY'; 'Torso', 'j3', ''; ...
         'LE', 'j4', ''; 'LF', 'j5', ''; 'LScap', 'j6', 'XY'; 'LS', 'j7', 'XYZ'; 'LW', 'j8', 'XY'; ...
         'RE', 'j15', ''; 'RF', 'j16', ''; 'RScap', 'j17', 'XY'; 'RS', 'j18', 'XYZ'; 'RW', 'j19', 'XY'};
    spec = cell2struct(c, {'prefix', 'id', 'axes'}, 2);
end
