function spec = gs3dx_upper_body_joints()
%GS3DX_UPPER_BODY_JOINTS  The twelve upper-body joints tracked from the capture (#10979).
%
%   SPEC = GS3DX_UPPER_BODY_JOINTS() is a struct array, one per upper-body
%   '<prefix> Input Function' chart of GS3DX_Fit and its variants:
%     .prefix  chart prefix ('Spine', 'LS', ...)
%     .id      GS3DX_Fit's KinematicsSolver joint id ('j7'; the joint
%              position variables start with it).  Simscape numbers joints
%              in block-path order, so a variant with more joints (the
%              neck of GS3DX_Human) renumbers them: use .block there
%     .block   the joint block's path below the model, the same in every
%              variant
%     .axes    '' (revolute Rz), 'XY' (universal Rx, Ry) or 'XYZ'
%              (spherical S, read as intrinsic X-Y-Z angles)

    uj = 'Universal Joint/Kinetically Driven Universal Joint';
    rj = 'Revolute Joint/Kinetically Driven Revolute';
    gj = 'Gimbal Joint/Kinetically Driven';
    c = {'Spine', 'j2', 'XY', ['Hips and Torso Inputs/Spine Tilt Kinetically Driven/' uj]
         'Torso', 'j3', '', ['Hips and Torso Inputs/Torso Kinetically Driven/' rj]
         'LE', 'j4', '', ['Left Elbow Joint/' rj]
         'LF', 'j5', '', ['Left Forearm/' rj]
         'LScap', 'j6', 'XY', ['Left Scapula Joint/' uj]
         'LS', 'j7', 'XYZ', ['Left Shoulder Joint/' gj]
         'LW', 'j8', 'XY', ['Left Wrist and Hand/' uj]
         'RE', 'j15', '', ['Right Elbow Joint/' rj]
         'RF', 'j16', '', ['Right Forearm/' rj]
         'RScap', 'j17', 'XY', ['Right Scapula Joint/' uj]
         'RS', 'j18', 'XYZ', ['Right Shoulder Joint/' gj]
         'RW', 'j19', 'XY', ['Right Wrist and Hand/' uj]};
    spec = cell2struct(c, {'prefix', 'id', 'axes', 'block'}, 2);
end
