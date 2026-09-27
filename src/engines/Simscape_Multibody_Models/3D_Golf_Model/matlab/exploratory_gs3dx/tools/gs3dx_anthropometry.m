function a = gs3dx_anthropometry(body_mass)
%GS3DX_ANTHROPOMETRY  Typical segment masses for the GS3DX golfer (#11011).
%
%   A = GS3DX_ANTHROPOMETRY(BODY_MASS) returns the segment masses (kg) of a
%   typical adult male of BODY_MASS kg (default 80), from the de Leva (1996)
%   mass fractions (J. Biomech. 29(9):1223-1230, Table 4, males), mapped
%   onto the solids of the GS3DX model:
%
%     .fraction   de Leva fractions of body mass, per segment (one side for
%                 limbs): head (head + neck), trunk, upper_trunk,
%                 middle_trunk, lower_trunk, upper_arm, forearm, hand,
%                 thigh, shank, foot.  They sum to 1 over the whole body.
%     .vars       model-workspace variables (kg) for the upper-body solids:
%                   GolferBodyMass        BODY_MASS
%                   GolferHeadMass        head + neck less the neck
%                   GolferNeckMass        neck_share of head + neck
%                   GolferLowerTrunkMass  lower trunk + half the middle trunk
%                                         ('LowerTorso', pelvis to mid-trunk)
%                   GolferUpperTrunkMass  the rest of the trunk less both
%                                         shoulder hubs ('UpperTorsoBase' 0.2
%                                         and 'UpperTorsoTop' 0.8 of it)
%                   GolferShoulderMass    shoulder_share of the upper trunk,
%                                         per side ('HubtoLS', 'HubtoRS')
%                   GolferUpperArmMass    per side
%                   GolferForearmMass     per side (two half solids each)
%                   GolferHandMass        per side
%     .legs       ThighMass, ShankMass, FootMass (the GS3DX_LEG_TABLE names)
%     .com        de Leva centre-of-mass position as a fraction of the
%                 segment length from its proximal end (head from the
%                 vertex, trunk from C7 toward the mid-hip point)
%
%   The model's trunk has no pelvis/abdomen/thorax split and its shoulder
%   hubs have no de Leva counterpart, so neck_share (0.15) and
%   shoulder_share (0.10) are modelling assumptions: they move mass inside
%   the head and trunk segments without changing either segment total.
%
%   Postcondition: the whole body (both sides of every limb) sums to
%   BODY_MASS within round-off.

    arguments
        body_mass (1,1) double {mustBePositive} = 80
    end
    f = struct('head', 0.0694, 'trunk', 0.4346, 'upper_trunk', 0.1596, ...
        'middle_trunk', 0.1633, 'lower_trunk', 0.1117, 'upper_arm', 0.0271, ...
        'forearm', 0.0162, 'hand', 0.0061, 'thigh', 0.1416, 'shank', 0.0433, ...
        'foot', 0.0137);
    neck_share = 0.15;
    shoulder_share = 0.10;
    M = body_mass;

    v = struct();
    v.GolferBodyMass = M;
    v.GolferNeckMass = neck_share * f.head * M;
    v.GolferHeadMass = f.head * M - v.GolferNeckMass;
    v.GolferLowerTrunkMass = (f.lower_trunk + f.middle_trunk / 2) * M;
    upper = (f.upper_trunk + f.middle_trunk / 2) * M;
    v.GolferShoulderMass = shoulder_share * upper;
    v.GolferUpperTrunkMass = upper - 2 * v.GolferShoulderMass;
    v.GolferUpperArmMass = f.upper_arm * M;
    v.GolferForearmMass = f.forearm * M;
    v.GolferHandMass = f.hand * M;

    legs = struct('ThighMass', f.thigh * M, 'ShankMass', f.shank * M, 'FootMass', f.foot * M);

    trunk_parts = f.upper_trunk + f.middle_trunk + f.lower_trunk;
    assert(abs(trunk_parts - f.trunk) < 1e-12, 'gs3dx:anthro', ...
        'de Leva trunk subsegments sum to %.4f, not %.4f', trunk_parts, f.trunk);
    total = v.GolferHeadMass + v.GolferNeckMass + v.GolferLowerTrunkMass + v.GolferUpperTrunkMass ...
        + 2 * (v.GolferShoulderMass + v.GolferUpperArmMass + v.GolferForearmMass + v.GolferHandMass ...
        + legs.ThighMass + legs.ShankMass + legs.FootMass);
    assert(abs(total - M) <= 1e-9 * M, 'gs3dx:anthro', ...
        'Postcondition: segments sum to %.3f kg, not %.3f kg', total, M);

    com = struct('head', 0.5976, 'trunk', 0.4486, 'upper_arm', 0.5772, 'forearm', 0.4574, ...
        'hand', 0.7900, 'thigh', 0.4095, 'shank', 0.4459, 'foot', 0.4415);
    a = struct('fraction', f, 'vars', v, 'legs', legs, 'com', com, ...
        'neck_share', neck_share, 'shoulder_share', shoulder_share);
end
