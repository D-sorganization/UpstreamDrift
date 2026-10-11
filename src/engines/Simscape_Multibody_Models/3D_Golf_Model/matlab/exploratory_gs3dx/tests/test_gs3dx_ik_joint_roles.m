function tests = test_gs3dx_ik_joint_roles
%TEST_GS3DX_IK_JOINT_ROLES  Unit tests for pure gs3dx_ik_joint_roles helper (#10979, #11161).
%   Proves invariant joint roles across standard, renumbered, and extended joint IDs,
%   and rejects ambiguous or missing anatomy.
    tests = functiontests(localfunctions);
end

function setupOnce(t)
    t.TestData.original_path = path;
    tools_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'tools');
    addpath(tools_dir);
end

function teardownOnce(t)
    path(t.TestData.original_path);
end

%% Helper: Synthetic Model Joint Table Builder
function [paths, ids] = local_build_synthetic_fit(id_prefix, pelvis_first)
    if nargin < 1
        id_prefix = "j";
    end
    if nargin < 2
        pelvis_first = true;
    end

    % Joint definitions: Name, sub-path, primitives
    defs = {
        'pelvis',   'Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint', {'Px.p', 'Py.p', 'Pz.p', 'S.q', 'S.ax_x', 'S.ax_y', 'S.ax_z'}
        'spine',    'Hips and Torso Inputs/Spine Tilt Kinetically Driven/Universal Joint/Kinetically Driven Universal Joint', {'Rx.q', 'Ry.q'}
        'torso',    'Hips and Torso Inputs/Torso Kinetically Driven/Revolute Joint/Kinetically Driven Revolute', {'Rz.q'}
        'l_hip',    'Lower Body/Left Hip Joint/Spherical Joint', {'S.q', 'S.ax_x', 'S.ax_y', 'S.ax_z'}
        'l_knee',   'Lower Body/Left Knee Joint/Kinetically Driven Revolute', {'Rz.q'}
        'l_ankle',  'Lower Body/Left Ankle Joint/Universal Joint', {'Rx.q', 'Ry.q'}
        'r_hip',    'Lower Body/Right Hip Joint/Spherical Joint', {'S.q', 'S.ax_x', 'S.ax_y', 'S.ax_z'}
        'r_knee',   'Lower Body/Right Knee Joint/Kinetically Driven Revolute', {'Rz.q'}
        'r_ankle',  'Lower Body/Right Ankle Joint/Universal Joint', {'Rx.q', 'Ry.q'}
        'l_scap',   'Left Scapula Joint/Universal Joint/Kinetically Driven Universal Joint', {'Rx.q', 'Ry.q'}
        'l_sh',     'Left Shoulder Joint/Spherical Joint', {'S.q', 'S.ax_x', 'S.ax_y', 'S.ax_z'}
        'l_el',     'Left Elbow Joint/Revolute Joint', {'Rz.q'}
        'l_fa',     'Left Forearm/Revolute Joint/Kinetically Driven Revolute', {'Rz.q'}
        'l_wr',     'Left Wrist and Hand/Universal Joint', {'Rx.q', 'Ry.q'}
        'r_scap',   'Right Scapula Joint/Universal Joint/Kinetically Driven Universal Joint', {'Rx.q', 'Ry.q'}
        'r_sh',     'Right Shoulder Joint/Spherical Joint', {'S.q', 'S.ax_x', 'S.ax_y', 'S.ax_z'}
        'r_el',     'Right Elbow Joint/Revolute Joint', {'Rz.q'}
        'r_fa',     'Right Forearm/Revolute Joint/Kinetically Driven Revolute', {'Rz.q'}
        'r_wr',     'Right Wrist and Hand/Universal Joint', {'Rx.q', 'Ry.q'}
    };

    if ~pelvis_first
        % Move pelvis to the end of definitions
        defs = [defs(2:end, :); defs(1, :)];
    end

    paths = strings(0, 1);
    ids = strings(0, 1);
    for i = 1:size(defs, 1)
        blk = "Model/" + string(defs{i, 2});
        prims = defs{i, 3};
        jid = id_prefix + string(i);
        for p = 1:numel(prims)
            paths(end + 1, 1) = blk; %#ok<AGROW>
            ids(end + 1, 1) = jid + "." + string(prims{p}); %#ok<AGROW>
        end
    end
end

%% 1. Invariant Roles on Standard Fit Model
function testStandardFitRoles(t)
    [paths, ids] = local_build_synthetic_fit("j", true);
    roles = gs3dx_ik_joint_roles(paths, ids);

    % Total independent coordinates must be 33 for Fit
    verifyEqual(t, roles.n_independent, 33);

    % Closed RHS arm joints
    verifyTrue(t, any(contains(paths(roles.closed_mask), "Right Elbow Joint")));
    verifyTrue(t, any(contains(paths(roles.closed_mask), "Right Shoulder Joint")));
    verifyTrue(t, any(contains(paths(roles.closed_mask), "Right Wrist and Hand")));
    % Total RHS arm closed variables: 1 (elbow Rz) + 4 (shoulder S) + 2 (wrist Rx,Ry) = 7
    verifyEqual(t, numel(roles.closed_ids), 7);
    verifyTrue(t, all(startsWith(roles.closed_ids, ["j16.", "j17.", "j19."])));

    % Trunk coordinates: Spine (2) + Torso (1) + LScap (2) + RScap (2) = 7
    verifyEqual(t, nnz(roles.is_trunk_coord), 7);

    % Pelvis translation indices in p: must be [1, 2, 3] when pelvis leads layout
    verifyEqual(t, roles.pelvis_trans_indices, [1, 2, 3]);
end

%% 2. Invariant Roles with Arbitrarily Renumbered IDs and Shuffled Blocks
function testRenumberedAndShuffledJoints(t)
    % Pelvis moved to the end, prefix changed to "joint_var_"
    [paths, ids] = local_build_synthetic_fit("joint_var_", false);
    roles = gs3dx_ik_joint_roles(paths, ids);

    % Total independent coordinates still 33
    verifyEqual(t, roles.n_independent, 33);

    % Pelvis translation indices must now point to the pelvis parameter slots
    verifyEqual(t, numel(roles.pelvis_trans_indices), 3);
    verifyFalse(t, isequal(roles.pelvis_trans_indices, [1, 2, 3]));
    % Indices must be valid within 1..33
    verifyTrue(t, all(roles.pelvis_trans_indices >= 1 & roles.pelvis_trans_indices <= 33));

    % Closed RHS arm variables still 7
    verifyEqual(t, numel(roles.closed_ids), 7);

    % Trunk coordinates still 7
    verifyEqual(t, nnz(roles.is_trunk_coord), 7);
end

%% 3. Dynamic Expansion for Human Model (Added Neck and MTP Joints)
function testExtendedHumanModelLayout(t)
    [paths, ids] = local_build_synthetic_fit("j", true);

    % Add Neck joint (Rx, Ry: 2 DOFs)
    paths = [paths; "Model/Hips and Torso Inputs/Neck Joint"; "Model/Hips and Torso Inputs/Neck Joint"];
    ids = [ids; "j100.Rx.q"; "j100.Ry.q"];

    % Add Left & Right MTP joints (Rz: 1 DOF each)
    paths = [paths; "Model/Lower Body/L Midfoot Joint"; "Model/Lower Body/R Midfoot Joint"];
    ids = [ids; "j101.Rz.q"; "j102.Rz.q"];

    roles = gs3dx_ik_joint_roles(paths, ids);

    % Dimension expands dynamically: 33 + 2 (Neck) + 2 (MTP) = 37 independent coordinates
    verifyEqual(t, roles.n_independent, 37);

    % Invariant roles preserved
    verifyEqual(t, numel(roles.closed_ids), 7);
    verifyEqual(t, nnz(roles.is_trunk_coord), 7);
    verifyEqual(t, roles.pelvis_trans_indices, [1, 2, 3]);
end

%% 4. Rejection of Missing RHS Arm Joints
function testMissingRhsArmRejected(t)
    [paths, ids] = local_build_synthetic_fit();
    % Drop Right Elbow Joint
    drop = contains(paths, "Right Elbow Joint");
    paths(drop) = [];
    ids(drop) = [];

    verifyError(t, @() gs3dx_ik_joint_roles(paths, ids), 'gs3dx:ik:missing_anatomy');
end

%% 5. Rejection of Missing Trunk Joints
function testMissingTrunkRejected(t)
    [paths, ids] = local_build_synthetic_fit();
    % Drop Spine Tilt
    drop = contains(paths, "Spine Tilt");
    paths(drop) = [];
    ids(drop) = [];

    verifyError(t, @() gs3dx_ik_joint_roles(paths, ids), 'gs3dx:ik:missing_anatomy');
end

%% 6. Rejection of Missing Pelvis Translation Primitives
function testMissingPelvisTranslationRejected(t)
    [paths, ids] = local_build_synthetic_fit();
    % Drop Pz primitive of pelvis Hip Joint
    drop = contains(paths, "Hip Joint") & ~contains(paths, "Left Hip") & ~contains(paths, "Right Hip") & contains(ids, ".Pz.");
    paths(drop) = [];
    ids(drop) = [];

    verifyError(t, @() gs3dx_ik_joint_roles(paths, ids), 'gs3dx:ik:missing_anatomy');
end

%% 7. Rejection of Ambiguous Duplicate Joints
function testAmbiguousJointsRejected(t)
    [paths, ids] = local_build_synthetic_fit();
    % Duplicate Right Elbow Joint under another subsystem
    paths = [paths; "Model/Conflicting Body/Right Elbow Joint/Revolute Joint"];
    ids = [ids; "j999.Rz.q"];

    verifyError(t, @() gs3dx_ik_joint_roles(paths, ids), 'gs3dx:ik:ambiguous_anatomy');
end

%% 8. Rejection of Malformed Spherical Group
function testMalformedSphericalGroupRejected(t)
    [paths, ids] = local_build_synthetic_fit();
    % Drop S.ax_z from Left Hip Joint
    drop = contains(paths, "Left Hip Joint") & contains(ids, ".ax_z");
    paths(drop) = [];
    ids(drop) = [];

    verifyError(t, @() gs3dx_ik_joint_roles(paths, ids), 'gs3dx:ik:missing_anatomy');
end

%% 9. Rejection of Unsupported 1-DOF Primitive Suffix
function testUnsupported1DofPrimitiveSuffixRejected(t)
    [paths, ids] = local_build_synthetic_fit();
    % Corrupt Left Knee Rz.q to Rz.foo
    idx = find(contains(paths, "Left Knee") & endsWith(ids, ".Rz.q"), 1);
    ids(idx) = replace(ids(idx), ".Rz.q", ".Rz.foo");

    verifyError(t, @() gs3dx_ik_joint_roles(paths, ids), 'gs3dx:ik:missing_anatomy');
end

%% 10. Rejection of Duplicate 1-DOF Primitive Variable
function testDuplicate1DofPrimitiveRejected(t)
    [paths, ids] = local_build_synthetic_fit();
    % Duplicate Left Knee variable with same group key but distinct ID
    idx = find(contains(paths, "Left Knee"), 1);
    paths = [paths; paths(idx)];
    parts = split(ids(idx), '.');
    ids = [ids; parts(1) + ".Rz.p"];

    verifyError(t, @() gs3dx_ik_joint_roles(paths, ids), 'gs3dx:ik:ambiguous_anatomy');
end

%% 11. Rejection of Pelvis Translation with .q Alternative
function testPelvisTranslationWithQRejected(t)
    [paths, ids] = local_build_synthetic_fit();
    % Change pelvis Px.p to Px.q
    idx = find(contains(paths, "Hip Joint") & ~contains(paths, ["Left Hip", "Right Hip"]) & contains(ids, ".Px.p"), 1);
    ids(idx) = replace(ids(idx), ".Px.p", ".Px.q");

    verifyError(t, @() gs3dx_ik_joint_roles(paths, ids), 'gs3dx:ik:missing_anatomy');
end

%% 12. Rejection of Ambiguous Multiple Trunk Leaf Blocks
function testAmbiguousTrunkLeafBlocksRejected(t)
    [paths, ids] = local_build_synthetic_fit();
    % Add conflicting second leaf block for Spine Tilt
    paths = [paths; "Model/Hips and Torso Inputs/Spine Tilt Second Block/Universal Joint"];
    ids = [ids; "j998.Rx.q"];

    verifyError(t, @() gs3dx_ik_joint_roles(paths, ids), 'gs3dx:ik:ambiguous_anatomy');
end

%% 13. Rejection of Ambiguous Multiple Pelvis Hip Joint Blocks
function testAmbiguousPelvisBlocksRejected(t)
    [paths, ids] = local_build_synthetic_fit();
    % Add conflicting second leaf block matching Hip Joint
    paths = [paths; "Model/Hips and Torso Inputs/Alternative Hip/Hip Joint"];
    ids = [ids; "j997.Px.p"];

    verifyError(t, @() gs3dx_ik_joint_roles(paths, ids), 'gs3dx:ik:ambiguous_anatomy');
end

%% 14. Rejection of Empty or Missing Paths or IDs
function testEmptyOrMissingPathsOrIdsRejected(t)
    [paths, ids] = local_build_synthetic_fit();

    % Empty string in paths
    bad_paths = paths;
    bad_paths(1) = "";
    verifyError(t, @() gs3dx_ik_joint_roles(bad_paths, ids), 'gs3dx:ik:missing_anatomy');

    % Missing string in ids
    bad_ids = ids;
    bad_ids(1) = string(missing);
    verifyError(t, @() gs3dx_ik_joint_roles(paths, bad_ids), 'gs3dx:ik:missing_anatomy');

    % Length mismatch
    verifyError(t, @() gs3dx_ik_joint_roles(paths(1:end-1), ids), 'gs3dx:ik:ambiguous_anatomy');
end

%% 15. Default Result Exactly Equals grounded_legs=false
function testDefaultEqualsGroundedLegsFalse(t)
    [paths, ids] = local_build_synthetic_fit();
    roles_default = gs3dx_ik_joint_roles(paths, ids);
    roles_explicit_false = gs3dx_ik_joint_roles(paths, ids, grounded_legs=false);

    verifyEqual(t, roles_default, roles_explicit_false);
end

%% 16. grounded_legs=true Converts Lower Body Variables to Closed Guesses
function testGroundedLegsTrueLowerBodyClosed(t)
    [paths, ids] = local_build_synthetic_fit("j", true);
    roles = gs3dx_ik_joint_roles(paths, ids, grounded_legs=true);

    % All six lower body role variables closed guesses
    leg_roles = ["Left Hip Joint", "Right Hip Joint", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"];
    for pat = leg_roles
        m = contains(paths, pat);
        verifyTrue(t, all(roles.closed_mask(m)));
    end

    % Total closed variables: 7 (RHS arm) + 4 (L Hip) + 4 (R Hip) + 1 (L Knee) + 1 (R Knee) + 2 (L Ankle) + 2 (R Ankle) = 21
    % Independent count: 21 rather than 33
    verifyEqual(t, roles.n_independent, 21);

    % Trunk mask count: 7
    verifyEqual(t, nnz(roles.is_trunk_coord), 7);

    % Pelvis indices preserved: [1, 2, 3]
    verifyEqual(t, roles.pelvis_trans_indices, [1, 2, 3]);
end

%% 17. Renumbered Fixture with Pelvis Last Retains grounded_legs Properties
function testGroundedLegsRenumberedPelvisLast(t)
    [paths, ids] = local_build_synthetic_fit("joint_var_", false);
    roles = gs3dx_ik_joint_roles(paths, ids, grounded_legs=true);

    % Lower body role variables closed
    leg_roles = ["Left Hip Joint", "Right Hip Joint", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"];
    for pat = leg_roles
        m = contains(paths, pat);
        verifyTrue(t, all(roles.closed_mask(m)));
    end

    % Independent count 21
    verifyEqual(t, roles.n_independent, 21);

    % Trunk mask count 7
    verifyEqual(t, nnz(roles.is_trunk_coord), 7);

    % Pelvis indices preserved within valid range of independent parameters
    verifyEqual(t, numel(roles.pelvis_trans_indices), 3);
    verifyFalse(t, isequal(roles.pelvis_trans_indices, [1, 2, 3]));
    verifyTrue(t, all(roles.pelvis_trans_indices >= 1 & roles.pelvis_trans_indices <= 21));
end

%% 18. Missing or Ambiguous Grounded Leg Role Errors gs3dx:ik:missing_anatomy
function testMissingOrAmbiguousGroundedLegRoleErrors(t)
    % Missing grounded leg role (drop Left Knee)
    [paths, ids] = local_build_synthetic_fit();
    drop = contains(paths, "Left Knee");
    paths_missing = paths(~drop);
    ids_missing = ids(~drop);
    verifyError(t, @() gs3dx_ik_joint_roles(paths_missing, ids_missing, grounded_legs=true), ...
        'gs3dx:ik:missing_anatomy');

    % Ambiguous grounded leg role (duplicate Right Ankle with conflicting leaf block)
    [paths, ids] = local_build_synthetic_fit();
    paths_ambig = [paths; "Model/Conflicting Lower Body/Right Ankle Joint/Universal Joint"];
    ids_ambig = [ids; "j999.Rx.q"];
    verifyError(t, @() gs3dx_ik_joint_roles(paths_ambig, ids_ambig, grounded_legs=true), ...
        'gs3dx:ik:missing_anatomy');
end
