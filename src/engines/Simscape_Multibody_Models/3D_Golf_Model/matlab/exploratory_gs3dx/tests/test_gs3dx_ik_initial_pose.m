classdef test_gs3dx_ik_initial_pose < matlab.unittest.TestCase
%TEST_GS3DX_IK_INITIAL_POSE  Pure contract tests for gs3dx_ik_initial_pose (#10979).

    methods (TestClassSetup)
        function setupClass(~)
            here = fileparts(mfilename('fullpath'));
            addpath(here);
            root_tools = fullfile(fileparts(fileparts(here)), 'tools');
            if exist(root_tools, 'dir'), addpath(root_tools); end
        end
    end

    methods (Test)
        function testEmptyDefaultCompatibility(testCase)
            testCase.verifyEqual(gs3dx_ik_initial_pose([]), struct([]));
            testCase.verifyEqual(gs3dx_ik_initial_pose(struct([])), struct([]));
            testCase.verifyError(@() gs3dx_ik_initial_pose(""), 'gs3dx:ik');
            testCase.verifyError(@() gs3dx_ik_initial_pose(''), 'gs3dx:ik');
            testCase.verifyError(@() gs3dx_ik_initial_pose(strings(0, 0)), 'gs3dx:ik');
        end

        function testStrictScalarStructAndAliases(testCase)
            p = local_fixture_pose();
            testCase.verifyError(@() gs3dx_ik_initial_pose(42), 'gs3dx:ik');
            testCase.verifyError(@() gs3dx_ik_initial_pose([p, p]), 'gs3dx:ik');
            alias_k = struct('keys', p.joint_keys, 'joint', p.joint, 'units', p.units, 'status', 1);
            testCase.verifyError(@() gs3dx_ik_initial_pose(alias_k), 'gs3dx:ik');
            alias_q = struct('joint_keys', p.joint_keys, 'q', p.joint, 'units', p.units, 'status', 1);
            testCase.verifyError(@() gs3dx_ik_initial_pose(alias_q), 'gs3dx:ik');
            alias_v = struct('joint_keys', p.joint_keys, 'values', p.joint, 'units', p.units, 'status', 1);
            testCase.verifyError(@() gs3dx_ik_initial_pose(alias_v), 'gs3dx:ik');
        end

        function testRequiredFieldsMissing(testCase)
            p = local_fixture_pose();
            for f = {'joint_keys', 'joint', 'units', 'status'}
                testCase.verifyError(@() gs3dx_ik_initial_pose(rmfield(p, f{1})), 'gs3dx:ik');
            end
        end

        function testStatusContractViolations(testCase)
            p = local_fixture_pose();
            for bad_st = {[1, 1], NaN, Inf, 1 + 1i, 0, -1, 2, [1; 1]}
                p.status = bad_st{1};
                testCase.verifyError(@() gs3dx_ik_initial_pose(p), 'gs3dx:ik');
            end
        end

        function testJointVectorContractViolations(testCase)
            p = local_fixture_pose();
            for bad_j = {repmat(p.joint, 1, 2), repmat(p.joint.', 2, 1), reshape(p.joint, [6, 8]), ...
                         NaN(48, 1), Inf(48, 1), complex(p.joint, 1), p.joint(1:47)}
                p.joint = bad_j{1};
                testCase.verifyError(@() gs3dx_ik_initial_pose(p), 'gs3dx:ik');
            end
        end

        function testKeysAndUnitsContractViolations(testCase)
            p = local_fixture_pose();
            bad_empty = p; bad_empty.joint_keys(1) = "";
            testCase.verifyError(@() gs3dx_ik_initial_pose(bad_empty), 'gs3dx:ik');
            bad_miss = p; bad_miss.joint_keys(1) = string(missing);
            testCase.verifyError(@() gs3dx_ik_initial_pose(bad_miss), 'gs3dx:ik');
            bad_dup = p; bad_dup.joint_keys(2) = bad_dup.joint_keys(1);
            testCase.verifyError(@() gs3dx_ik_initial_pose(bad_dup), 'gs3dx:ik');
            bad_types = p; bad_types.joint_keys = 1:48;
            testCase.verifyError(@() gs3dx_ik_initial_pose(bad_types), 'gs3dx:ik');
            bad_unit_empty = p; bad_unit_empty.units(1) = "";
            testCase.verifyError(@() gs3dx_ik_initial_pose(bad_unit_empty), 'gs3dx:ik');
            bad_unit_type = p; bad_unit_type.units = 1:48;
            testCase.verifyError(@() gs3dx_ik_initial_pose(bad_unit_type), 'gs3dx:ik');
            bad_unit_mismatch = p; bad_unit_mismatch.units(1) = "deg";
            testCase.verifyError(@() gs3dx_ik_initial_pose(bad_unit_mismatch), 'gs3dx:ik');
        end

        function testSphericalAxisAndGimbalSingularity(testCase)
            p = local_fixture_pose();
            k_ax = "Right Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_x";
            k_q = "Right Shoulder Joint/Gimbal Joint/Kinetically Driven|S.q";
            p_ax = p; p_ax.joint(p.joint_keys == k_ax) = 2.0; p_ax.joint(p.joint_keys == k_q) = 15.0;
            testCase.verifyError(@() gs3dx_ik_initial_pose(p_ax), 'gs3dx:ik');

            p_gim = p;
            p_gim.joint(p.joint_keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.ax_y") = 1.0;
            p_gim.joint(p.joint_keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.ax_z") = 0.0;
            p_gim.joint(p.joint_keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.q") = 90.0;
            testCase.verifyError(@() gs3dx_ik_initial_pose(p_gim), 'gs3dx:ik');
        end

        function testKeyedPermutationAssociation(testCase)
            p = local_fixture_pose();
            p.joint = (1:48).' * 0.75;
            sph_z = endsWith(p.joint_keys, ".S.ax_z") | endsWith(p.joint_keys, "|S.ax_z");
            p.joint(sph_z) = 1.0;
            sph_x = endsWith(p.joint_keys, ".S.ax_x") | endsWith(p.joint_keys, "|S.ax_x");
            p.joint(sph_x) = 0.0;
            sph_y = endsWith(p.joint_keys, ".S.ax_y") | endsWith(p.joint_keys, "|S.ax_y");
            p.joint(sph_y) = 0.0;
            sph_q = endsWith(p.joint_keys, ".S.q") | endsWith(p.joint_keys, "|S.q");
            p.joint(sph_q) = 0.0;

            perm = [25:48, 1:24];
            p_perm = struct('joint_keys', p.joint_keys(perm), 'joint', p.joint(perm), ...
                'units', p.units(perm), 'status', 1);

            out = gs3dx_ik_initial_pose(p_perm);
            testCase.verifyEqual(out.status, 1);
            [found, loc] = ismember(p.joint_keys, out.joint_keys);
            testCase.verifyTrue(all(found));
            testCase.verifyEqual(out.joint(loc), p.joint);
            testCase.verifyEqual(out.units(loc), p.units);
        end
    end
end

function p = local_fixture_pose()
    keys = [
        "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|Px.p"
        "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|Py.p"
        "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|Pz.p"
        "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.ax_x"
        "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.ax_y"
        "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.ax_z"
        "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.q"
        "Hips and Torso Inputs/Neck Joint|Rx.q"
        "Hips and Torso Inputs/Neck Joint|Ry.q"
        "Hips and Torso Inputs/Spine Tilt Kinetically Driven/Universal Joint/Kinetically Driven Universal Joint|Rx.q"
        "Hips and Torso Inputs/Spine Tilt Kinetically Driven/Universal Joint/Kinetically Driven Universal Joint|Ry.q"
        "Hips and Torso Inputs/Torso Kinetically Driven/Revolute Joint/Kinetically Driven Revolute|Rz.q"
        "Left Elbow Joint/Revolute Joint/Kinetically Driven Revolute|Rz.q"
        "Left Forearm/Revolute Joint/Kinetically Driven Revolute|Rz.q"
        "Left Scapula Joint/Universal Joint/Kinetically Driven Universal Joint|Rx.q"
        "Left Scapula Joint/Universal Joint/Kinetically Driven Universal Joint|Ry.q"
        "Left Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_x"
        "Left Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_y"
        "Left Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_z"
        "Left Shoulder Joint/Gimbal Joint/Kinetically Driven|S.q"
        "Left Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint|Rx.q"
        "Left Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint|Ry.q"
        "Lower Body/L Midfoot Joint|Rz.q"
        "Lower Body/Left Ankle Joint/Kinetically Driven Universal Joint|Rx.q"
        "Lower Body/Left Ankle Joint/Kinetically Driven Universal Joint|Ry.q"
        "Lower Body/Left Hip Joint/Kinetically Driven|S.ax_x"
        "Lower Body/Left Hip Joint/Kinetically Driven|S.ax_y"
        "Lower Body/Left Hip Joint/Kinetically Driven|S.ax_z"
        "Lower Body/Left Hip Joint/Kinetically Driven|S.q"
        "Lower Body/Left Knee Joint/Kinetically Driven Revolute|Rz.q"
        "Lower Body/R Midfoot Joint|Rz.q"
        "Lower Body/Right Ankle Joint/Kinetically Driven Universal Joint|Rx.q"
        "Lower Body/Right Ankle Joint/Kinetically Driven Universal Joint|Ry.q"
        "Lower Body/Right Hip Joint/Kinetically Driven|S.ax_x"
        "Lower Body/Right Hip Joint/Kinetically Driven|S.ax_y"
        "Lower Body/Right Hip Joint/Kinetically Driven|S.ax_z"
        "Lower Body/Right Hip Joint/Kinetically Driven|S.q"
        "Lower Body/Right Knee Joint/Kinetically Driven Revolute|Rz.q"
        "Right Elbow Joint/Revolute Joint/Kinetically Driven Revolute|Rz.q"
        "Right Forearm/Revolute Joint/Kinetically Driven Revolute|Rz.q"
        "Right Scapula Joint/Universal Joint/Kinetically Driven Universal Joint|Rx.q"
        "Right Scapula Joint/Universal Joint/Kinetically Driven Universal Joint|Ry.q"
        "Right Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_x"
        "Right Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_y"
        "Right Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_z"
        "Right Shoulder Joint/Gimbal Joint/Kinetically Driven|S.q"
        "Right Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint|Rx.q"
        "Right Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint|Ry.q"
    ];
    n = numel(keys);
    units = repmat("deg", n, 1);
    units(startsWith(keys, "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|P")) = "m";
    units(endsWith(keys, ".ax_x") | endsWith(keys, ".ax_y") | endsWith(keys, ".ax_z")) = "1";
    q = zeros(n, 1);
    q(keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|Px.p") = 0.1;
    q(keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|Py.p") = -0.2;
    q(keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|Pz.p") = 0.85;
    q(endsWith(keys, ".ax_z")) = 1.0;
    p = struct('joint_keys', keys, 'units', units, 'joint', q, 'status', 1);
end
