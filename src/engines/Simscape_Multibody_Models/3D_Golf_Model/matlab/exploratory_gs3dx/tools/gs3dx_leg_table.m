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
%     .stance   ground-contact stance (#10985/#10986), in the leg frame
%               relative to the pelvis frame origin at t = 0 (m, deg):
%               ankle_L/ankle_R (x facing, y target side) and foot_yaw_L/_R
%               (toe direction from facing, positive toward the target)
%               measured from the tour-average driver capture at address
%               (GS3DX_CAPTURE_STANCE, data/C3D_TA_Driver.c3d); drop, the
%               ankle joint centre below the pelvis frame origin, is the
%               capture's waist-centre-to-ankle-marker height (mean of both
%               sides) and assumes the model pelvis origin sits at the
%               waist-marker centre; inset moves the lateral-malleolus
%               marker to the joint centre (assumed half ankle width);
%               foot_width is the contact-point spacing across the foot.
%     .contact  foot-ground contact and leg servo parameters: sphere
%               radius, stiffness, damping, friction (assumed; see
%               docs/DATA_AUDIT.md) and joint servo gains per leg axis.
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
    leg.stance = struct( ...
        'ankle_L', [0.0208; 0.3185], 'ankle_R', [0.0208; -0.3319], ...
        'foot_yaw_L', -2.15, 'foot_yaw_R', 8.15, ...
        'drop', 0.931, 'inset', 0.035, ...
        'foot_width', 0.11);        % capture ToeIn-ToeOut 0.107 (L) / 0.117 (R)
    % Servo gains per leg axis [hip X Y Z, knee, ankle X Y] (N*m/deg and
    % N*m*s/deg; the joints report degrees).
    leg.contact = struct( ...
        'sphere_radius', 0.01, 'stiffness', 1e5, 'damping', 1e3, ...
        'transition_width', 1e-4, 'mu_static', 0.9, 'mu_dynamic', 0.7, ...
        'critical_velocity', 1e-3, ...
        'kp', [100 100 100 100 50 50], 'kd', [2 2 2 2 1 1]);
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
