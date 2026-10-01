function rom = gs3dx_joint_rom()
%GS3DX_JOINT_ROM  Human range of motion of every GS3DX_Human revolute and universal joint axis (#11158).
%
%   ROM = GS3DX_JOINT_ROM() is the single source of truth for the joint
%   limits of the GS3DX golfer: one table row per joint primitive, with
%   the anatomical meaning of the primitive, its neutral and sign, and the
%   normal adult range with its source.  The IK penalty, the built joint
%   limits and every ROM check read this table.
%
%   The anatomical angle of a primitive value Q (deg, as the
%   KinematicsSolver and the '<J>TrackAngle' charts give it) is
%       A = SIGN * wrap180(Q - NEUTRAL)
%   and is normal when MIN_DEG <= A <= MAX_DEG.  NEUTRAL and SIGN were
%   measured on GS3DX_Human at the capture address pose (docs/ROM.md):
%     - elbows and knees: the primitive is the angle between the segment
%       long axes (zero = straight);
%     - knees: flexion is -Rz on GS3DX_Human (its leg servo reference runs
%       -5 to -57 deg over the swing; the sign probe agrees);
%     - spine, neck: zero where the child segment's long axis lies along
%       the parent's; torso: where the chest's lateral axis is closest to
%       the pelvis's;
%     - forearms: zero where the wrist's dorsal-palmar axis lies along the
%       elbow axis (thumb up with the elbow bent);
%     - scapulae, ankles, midfoot: the model's zero (the original model's
%       rest posture; standing neutral for the feet);
%     - SIGN from the direction a distal point moves for a positive
%       primitive, against the anatomical positive direction named in
%       MOTION.
%   For the wrists no neutral is established: the drawn hands sit across
%   the shaft, not along a real grip, so their long axis is no anatomical
%   reference (#11157).  Their rows bound the total arc only: SPAN_DEG,
%   the swing's range max(A) - min(A), must not exceed the normal arc.
%
%   The spherical joints (shoulders, hips) and the free pelvis have no rows
%   yet (docs/ROM.md, open work).
%
%   ROM columns:
%     joint       short name ('LE', 'Spine', ...)
%     key         model-independent joint key (GS3DX_JOINT_KEYS)
%     track       model-workspace reference: '<prefix>TrackAngle', or
%                 'LegReferenceAngle' / 'NeckReference' ('' for none)
%     track_row   row of that reference (0 for none)
%     motion      the anatomical motion that is positive
%     sign        +1 or -1
%     neutral_deg primitive value at the anatomical neutral (NaN: none)
%     min_deg, max_deg  normal range of the anatomical angle (NaN: none)
%     span_deg    normal total arc (deg)
%     source      the reference of the range

    S.aaos = "AAOS normal ROM (Greene & Heckman 1994)";
    S.soucie = "Soucie et al. 2011, Haemophilia 17:500-507";
    S.xf = "Cheetham et al. 2001; Joyce et al. 2010 (golf X-factor up to ~60 deg)";
    S.ludewig = "Ludewig et al. 2009, J Bone Joint Surg Am 91:378-389";
    S.mtp = "AAOS first MTP (Greene & Heckman 1994)";
    trunk = "Hips and Torso Inputs/";
    c = {
    %   joint     key                                                                                         track                row motion                              sign  neutral  min  max  span source
        'Neck',   trunk + "Neck Joint|Rx.q",                                                                 "NeckReference",      1, "neck flexion",                         1,  24.01, -45,  45,  90, S.aaos
        'Neck',   trunk + "Neck Joint|Ry.q",                                                                 "NeckReference",      2, "neck lateral bend to trail",           1,   7.67, -45,  45,  90, S.aaos
        'Spine',  trunk + "Spine Tilt Kinetically Driven/Universal Joint/Kinetically Driven Universal Joint|Rx.q", "SpineTrackAngle", 1, "trunk flexion",                  1,      0, -25,  80, 105, S.aaos
        'Spine',  trunk + "Spine Tilt Kinetically Driven/Universal Joint/Kinetically Driven Universal Joint|Ry.q", "SpineTrackAngle", 2, "trunk lateral bend to trail",    1,      0, -35,  35,  70, S.aaos
        'Torso',  trunk + "Torso Kinetically Driven/Revolute Joint/Kinetically Driven Revolute|Rz.q",       "TorsoTrackAngle",    1, "trunk rotation toward target",         1, -26.81, -60,  60, 120, S.xf
        'LE',     "Left Elbow Joint/Revolute Joint/Kinetically Driven Revolute|Rz.q",                       "LETrackAngle",       1, "lead elbow flexion",                   1,      0,  -5, 150, 155, S.aaos
        'RE',     "Right Elbow Joint/Revolute Joint/Kinetically Driven Revolute|Rz.q",                      "RETrackAngle",       1, "trail elbow flexion",                 -1,      0,  -5, 150, 155, S.aaos
        'LF',     "Left Forearm/Revolute Joint/Kinetically Driven Revolute|Rz.q",                           "LFTrackAngle",       1, "lead forearm pronation",               1,     90, -80,  80, 160, S.aaos
        'RF',     "Right Forearm/Revolute Joint/Kinetically Driven Revolute|Rz.q",                          "RFTrackAngle",       1, "trail forearm pronation",             -1,     90, -80,  80, 160, S.aaos
        'LScap',  "Left Scapula Joint/Universal Joint/Kinetically Driven Universal Joint|Rx.q",             "LScapTrackAngle",    1, "lead shoulder girdle protraction",     1,      0, -25,  25,  50, S.ludewig
        'LScap',  "Left Scapula Joint/Universal Joint/Kinetically Driven Universal Joint|Ry.q",             "LScapTrackAngle",    2, "lead shoulder girdle elevation",       1,      0, -10,  40,  50, S.ludewig
        'RScap',  "Right Scapula Joint/Universal Joint/Kinetically Driven Universal Joint|Rx.q",            "RScapTrackAngle",    1, "trail shoulder girdle protraction",   -1,      0, -25,  25,  50, S.ludewig
        'RScap',  "Right Scapula Joint/Universal Joint/Kinetically Driven Universal Joint|Ry.q",            "RScapTrackAngle",    2, "trail shoulder girdle elevation",     -1,      0, -10,  40,  50, S.ludewig
        'LW',     "Left Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint|Rx.q",            "LWTrackAngle",       1, "lead wrist radial deviation",          1,    NaN, NaN, NaN,  50, S.aaos
        'LW',     "Left Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint|Ry.q",            "LWTrackAngle",       2, "lead wrist flexion",                   1,    NaN, NaN, NaN, 150, S.aaos
        'RW',     "Right Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint|Rx.q",           "RWTrackAngle",       1, "trail wrist radial deviation",         1,    NaN, NaN, NaN,  50, S.aaos
        'RW',     "Right Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint|Ry.q",           "RWTrackAngle",       2, "trail wrist flexion",                 -1,    NaN, NaN, NaN, 150, S.aaos
        'LK',     "Lower Body/Left Knee Joint/Kinetically Driven Revolute|Rz.q",                            "LegReferenceAngle",  4, "lead knee flexion",                   -1,      0,  -5, 135, 140, S.aaos
        'RK',     "Lower Body/Right Knee Joint/Kinetically Driven Revolute|Rz.q",                           "LegReferenceAngle", 10, "trail knee flexion",                  -1,      0,  -5, 135, 140, S.aaos
        'LA',     "Lower Body/Left Ankle Joint/Kinetically Driven Universal Joint|Rx.q",                    "LegReferenceAngle",  5, "lead ankle inversion",                -1,      0, -15,  35,  50, S.aaos
        'LA',     "Lower Body/Left Ankle Joint/Kinetically Driven Universal Joint|Ry.q",                    "LegReferenceAngle",  6, "lead ankle dorsiflexion",             -1,      0, -50,  20,  70, S.aaos
        'RA',     "Lower Body/Right Ankle Joint/Kinetically Driven Universal Joint|Rx.q",                   "LegReferenceAngle", 11, "trail ankle inversion",                1,      0, -15,  35,  50, S.aaos
        'RA',     "Lower Body/Right Ankle Joint/Kinetically Driven Universal Joint|Ry.q",                   "LegReferenceAngle", 12, "trail ankle dorsiflexion",            -1,      0, -50,  20,  70, S.aaos
        'LMid',   "Lower Body/L Midfoot Joint|Rz.q",                                                        "",                   0, "lead toe extension",                  -1,      0, -45,  70, 115, S.mtp
        'RMid',   "Lower Body/R Midfoot Joint|Rz.q",                                                        "",                   0, "trail toe extension",                 -1,      0, -45,  70, 115, S.mtp
        };
    rom = cell2table(c, 'VariableNames', {'joint', 'key', 'track', 'track_row', 'motion', 'sign', ...
        'neutral_deg', 'min_deg', 'max_deg', 'span_deg', 'source'});
    rom.joint = string(rom.joint);

    % Postconditions
    assert(numel(unique(rom.key)) == height(rom), 'gs3dx:rom', 'Repeated joint key');
    assert(all(abs(rom.sign) == 1), 'gs3dx:rom', 'SIGN must be +1 or -1');
    bounded = ~isnan(rom.min_deg);
    assert(isequal(bounded, ~isnan(rom.max_deg), ~isnan(rom.neutral_deg)), 'gs3dx:rom', ...
        'A row has all of NEUTRAL, MIN and MAX or none');
    assert(all(rom.min_deg(bounded) < 0 & rom.max_deg(bounded) > 0), 'gs3dx:rom', 'The neutral must lie inside the range');
    assert(all(abs(rom.max_deg(bounded) - rom.min_deg(bounded) - rom.span_deg(bounded)) < 1e-9), 'gs3dx:rom', ...
        'SPAN must be MAX - MIN where both are given');
    assert(all(rom.span_deg > 0 & rom.span_deg <= 180), 'gs3dx:rom', 'SPAN must be in (0, 180]');
    assert(all(strlength(rom.source) > 0), 'gs3dx:rom', 'Every range needs a source');
    assert(all((rom.track == "") == (rom.track_row == 0)), 'gs3dx:rom', 'TRACK and TRACK_ROW go together');
end
