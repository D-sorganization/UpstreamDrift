classdef test_gs3dx_ik_native_initial_pose < matlab.unittest.TestCase
%TEST_GS3DX_IK_NATIVE_INITIAL_POSE  Pure contract tests for gs3dx_ik_initial_pose native schema (#10979).

    methods (TestClassSetup)
        function setupPath(tc)
            here = fileparts(mfilename('fullpath'));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(here));
            root = fileparts(here);
            tools_dir = fullfile(root, 'tools');
            if isfolder(tools_dir)
                tc.applyFixture(matlab.unittest.fixtures.PathFixture(tools_dir));
            end
        end
    end

    methods (Test)
        % --- Native schema contract tests ---
        function testNativeSchemaValidScalar(testCase)
            [pose, schema] = local_native_fixture_pose();
            out = gs3dx_ik_initial_pose(pose, 'native_schema', schema);
            testCase.verifyEqual(out.status, 1);
            testCase.verifyEqual(out.joint_keys, pose.joint_keys);
            testCase.verifyEqual(out.joint, pose.joint);
            testCase.verifyEqual(out.units, pose.units);
            testCase.verifyEqual(numel(out.joint), 6); % not 48
        end

        function testNativeSchemaEmptyInitialPose(testCase)
            [~, schema] = local_native_fixture_pose();
            testCase.verifyEqual(gs3dx_ik_initial_pose([], 'native_schema', schema), struct([]));
            testCase.verifyEqual(gs3dx_ik_initial_pose(struct([]), 'native_schema', schema), struct([]));
        end

        function testNativeSchemaAcceptedKeyedPermutation(testCase)
            [pose, schema] = local_native_fixture_pose();
            perm = [5, 2, 6, 1, 4, 3];
            pose_perm = struct( ...
                'joint_keys', pose.joint_keys(perm), ...
                'joint', pose.joint(perm), ...
                'units', pose.units(perm), ...
                'status', 1);

            out = gs3dx_ik_initial_pose(pose_perm, 'native_schema', schema);
            testCase.verifyEqual(out.status, 1);
            % Preserves and normalizes the input keyed association
            [found, loc] = ismember(pose.joint_keys, out.joint_keys);
            testCase.verifyTrue(all(found));
            testCase.verifyEqual(out.joint(loc), pose.joint);
            testCase.verifyEqual(out.units(loc), pose.units);
        end

        function testNativeSchemaZeroAngleArbitraryAxisAccepted(testCase)
            [pose, schema] = local_native_fixture_pose();
            % Set S.q = 0 and non-unit axis
            pose.joint(pose.joint_keys == "JointGroup/Ball|S.q") = 0.0;
            pose.joint(pose.joint_keys == "JointGroup/Ball|S.ax_x") = 5.0;
            pose.joint(pose.joint_keys == "JointGroup/Ball|S.ax_y") = 0.0;
            pose.joint(pose.joint_keys == "JointGroup/Ball|S.ax_z") = 0.0;

            out = gs3dx_ik_initial_pose(pose, 'native_schema', schema);
            testCase.verifyEqual(out.status, 1);
            idx_q = (out.joint_keys == "JointGroup/Ball|S.q");
            testCase.verifyEqual(out.joint(idx_q), 0.0);
        end

        function testNativeSchemaNonzeroSphericalNonunitAxisRejected(testCase)
            [pose, schema] = local_native_fixture_pose();
            % S.q is nonzero (10.0), make axis nonunit
            pose.joint(pose.joint_keys == "JointGroup/Ball|S.ax_z") = 2.0;
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose, 'native_schema', schema), 'gs3dx:ik');

            % Zero vector axis with nonzero angle
            pose.joint(pose.joint_keys == "JointGroup/Ball|S.ax_x") = 0.0;
            pose.joint(pose.joint_keys == "JointGroup/Ball|S.ax_y") = 0.0;
            pose.joint(pose.joint_keys == "JointGroup/Ball|S.ax_z") = 0.0;
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose, 'native_schema', schema), 'gs3dx:ik');
        end

        function testNativeSchemaWrongUnitRejected(testCase)
            [pose, schema] = local_native_fixture_pose();
            % Mutate one unit in pose
            pose_bad = pose;
            pose_bad.units(pose_bad.joint_keys == "JointGroup/Slide|Px.p") = "deg";
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose_bad, 'native_schema', schema), 'gs3dx:ik');

            pose_bad2 = pose;
            pose_bad2.units(pose_bad2.joint_keys == "JointGroup/Ball|S.ax_x") = "deg";
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose_bad2, 'native_schema', schema), 'gs3dx:ik');
        end

        function testNativeSchemaUnknownMissingOrDuplicateKeyRejected(testCase)
            [pose, schema] = local_native_fixture_pose();

            % Missing key / count mismatch
            pose_missing = pose;
            pose_missing.joint_keys(end) = [];
            pose_missing.joint(end) = [];
            pose_missing.units(end) = [];
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose_missing, 'native_schema', schema), 'gs3dx:ik');

            % Unknown key (same count, replaced key)
            pose_unknown = pose;
            pose_unknown.joint_keys(end) = "JointGroup/Unknown|Rz.q";
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose_unknown, 'native_schema', schema), 'gs3dx:ik');

            % Duplicate key
            pose_dup = pose;
            pose_dup.joint_keys(2) = pose_dup.joint_keys(1);
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose_dup, 'native_schema', schema), 'gs3dx:ik');
        end

        function testNativeSchemaMalformedSchemaRejected(testCase)
            [pose, ~] = local_native_fixture_pose();

            % Schema missing fields
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose, 'native_schema', struct()), 'gs3dx:ik');
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose, 'native_schema', struct('joint_keys', pose.joint_keys)), 'gs3dx:ik');
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose, 'native_schema', struct('units', pose.units)), 'gs3dx:ik');

            % Schema non-struct or array struct
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose, 'native_schema', "schema"), 'gs3dx:ik');
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose, 'native_schema', [struct('joint_keys', pose.joint_keys, 'units', pose.units), ...
                                                                                  struct('joint_keys', pose.joint_keys, 'units', pose.units)]), 'gs3dx:ik');

            % Schema keys/units count mismatch or non-vector or duplicates
            bad_schema1 = struct('joint_keys', pose.joint_keys(1:3), 'units', pose.units);
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose, 'native_schema', bad_schema1), 'gs3dx:ik');

            bad_schema_dup = struct('joint_keys', [pose.joint_keys(1); pose.joint_keys(1); pose.joint_keys(3:end)], 'units', pose.units);
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose, 'native_schema', bad_schema_dup), 'gs3dx:ik');

            bad_schema_empty_str = struct('joint_keys', [""; pose.joint_keys(2:end)], 'units', pose.units);
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose, 'native_schema', bad_schema_empty_str), 'gs3dx:ik');

            bad_schema_missing = struct('joint_keys', [string(missing); pose.joint_keys(2:end)], 'units', pose.units);
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose, 'native_schema', bad_schema_missing), 'gs3dx:ik');
        end

        function testNativeSchemaMultiframeNonfiniteStatusRejected(testCase)
            [pose, schema] = local_native_fixture_pose();

            % Multiframe joint matrix
            pose_multi = pose;
            pose_multi.joint = repmat(pose.joint, 1, 2);
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose_multi, 'native_schema', schema), 'gs3dx:ik');

            % Nonfinite joint values
            pose_nan = pose;
            pose_nan.joint(1) = NaN;
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose_nan, 'native_schema', schema), 'gs3dx:ik');

            pose_inf = pose;
            pose_inf.joint(1) = Inf;
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose_inf, 'native_schema', schema), 'gs3dx:ik');

            % Complex joint values
            pose_cplx = pose;
            pose_cplx.joint(1) = 1 + 1i;
            testCase.verifyError(@() gs3dx_ik_initial_pose(pose_cplx, 'native_schema', schema), 'gs3dx:ik');

            % Status violations
            for bad_st = {[1, 1], NaN, Inf, 1 + 1i, 0, -1, 2, [1; 1]}
                pose_st = pose;
                pose_st.status = bad_st{1};
                testCase.verifyError(@() gs3dx_ik_initial_pose(pose_st, 'native_schema', schema), 'gs3dx:ik');
            end
        end
    end
end

function [pose, schema] = local_native_fixture_pose()
    % Small synthetic scalar fixture with spherical S.ax_x/y/z and S.q, a Px.p and Rx.q (length 6, not 48)
    keys = [
        "JointGroup/Slide|Px.p"
        "JointGroup/Hinge|Rx.q"
        "JointGroup/Ball|S.ax_x"
        "JointGroup/Ball|S.ax_y"
        "JointGroup/Ball|S.ax_z"
        "JointGroup/Ball|S.q"
    ];
    units = [
        "m"
        "deg"
        "1"
        "1"
        "1"
        "deg"
    ];
    joint = [
        0.05
        15.0
        0.0
        0.0
        1.0
        10.0
    ];
    schema = struct('joint_keys', keys, 'units', units);
    pose = struct('joint_keys', keys, 'joint', joint, 'units', units, 'status', 1);
end
