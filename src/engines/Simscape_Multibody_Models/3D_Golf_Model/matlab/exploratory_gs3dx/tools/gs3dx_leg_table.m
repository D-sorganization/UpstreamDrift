function leg = gs3dx_leg_table()
%GS3DX_LEG_TABLE  Segment/joint table and anthropometry for the GS3DX legs.
%
%   LEG = GS3DX_LEG_TABLE() returns the data GS3DX_BUILD_LOWER_BODY builds
%   from (#10957):
%     .params   struct of model-workspace variables (SI units).  Segment
%               masses and lengths are de Leva (1996) male fractions of
%               LegBodyMass and LegBodyHeight.  These are assumptions, not
%               fitted to a golfer; edit them in the model workspace.
%     .joints   struct array, proximal to distal: name, kds (referenced
%               subsystem), axes (torque axes), start (start angles, deg,
%               in leg-frame terms, see below), segment (distal body) and
%               priority (start-position target priority).
%     .theta    knee-bend half-angle (deg) of the start posture.
%
%   Leg frame: x forward (facing), y to the golfer's left, z up.  With the
%   start angles below the thigh leans forward by theta, the shank leans
%   back by theta and the foot is level, so the ankle sits under the hip at
%   params.LegReachFraction of the straight-leg length.  A welded leg is a
%   closed loop with the pelvis joint, and Simscape ignores targets when
%   every joint in a loop has one, so the hip has none; Low knee and ankle
%   targets pick the bent-knee branch.

    H = 1.80;  M = 80;
    p = struct();
    p.LegBodyHeight   = H;
    p.LegBodyMass     = M;
    p.ThighLength     = 0.2425 * H;
    p.ThighMass       = 0.1416 * M;
    p.ThighRadius     = 0.07;
    p.ShankLength     = 0.2465 * H;
    p.ShankMass       = 0.0433 * M;
    p.ShankRadius     = 0.05;
    p.FootLength      = 0.152 * H;
    p.FootWidth       = 0.10;
    p.FootMass        = 0.0137 * M;
    p.AnkleHeight     = 0.039 * H;
    p.FootHeelOffset  = 0.25;       % fraction of foot length behind the ankle
    p.HipJointSpacing = 0.10 * H;   % between hip joint centres
    p.HipJointDrop    = 0.10;       % hip centres below the pelvis frame, m
    p.LegReachFraction = 0.97;      % hip-to-ankle distance / straight leg
    p.LegTorqueCommand = zeros(12, 1);   % [L hip XYZ, knee, ankle XY, R ...], N*m

    theta = acosd(p.LegReachFraction);
    leg.params = p;
    leg.theta = theta;
    leg.joints = struct( ...
        'name',    {'Hip', 'Knee', 'Ankle'}, ...
        'kds',     {'GS3DX_KDS_Spherical', 'GS3DX_KDS_Revolute', 'GS3DX_KDS_Universal'}, ...
        'axes',    {'XYZ', '', 'XY'}, ...
        'start',   {[0 -theta 0], -2 * theta, [0 -theta]}, ...
        'segment', {'Thigh', 'Shank', 'Foot'}, ...
        'priority', {'None', 'Low', 'Low'});
end
