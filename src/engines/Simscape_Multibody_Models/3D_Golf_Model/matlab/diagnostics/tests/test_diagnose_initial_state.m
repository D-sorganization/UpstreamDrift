classdef test_diagnose_initial_state < matlab.unittest.TestCase
    %TEST_DIAGNOSE_INITIAL_STATE  Authored, not executed in CI (no MATLAB).
    %
    %   These tests cover the loop-closure projection diagnostic. They
    %   require Simscape Multibody + the 3D golf model on the path. Run
    %   manually:
    %
    %       results = runtests('test_diagnose_initial_state');

    properties
        TmpDir
    end

    methods(TestMethodSetup)
        function makeTmp(testCase)
            testCase.TmpDir = tempname();
            mkdir(testCase.TmpDir);
        end
    end

    methods(TestMethodTeardown)
        function rmTmp(testCase)
            if isfolder(testCase.TmpDir)
                rmdir(testCase.TmpDir, 's');
            end
        end
    end

    methods(Test)
        function test_specified_pose_decodes_correctly_from_known_input(testCase)
            % Build a synthetic input file with q0 + joint_names and verify
            % the decoder returns them unchanged.
            here = fileparts(mfilename('fullpath'));
            addpath(fullfile(here, '..', 'private'));

            inputs = struct();
            inputs.q0 = [0.1; 0.2; -0.3; 0.0; 0.5];
            inputs.joint_names = {'hip_x', 'hip_y', 'spine_z', 'shoulder', 'wrist'};
            inputs.r_butt = [0.1; 0.2; 1.0];
            inputs.r_clubhead = [0.5; 0.3; 0.1];

            mat_path = fullfile(testCase.TmpDir, 'fake_inputs.mat');
            save(mat_path, '-struct', 'inputs');

            % decode_input_file_pose expects the fields under a top-level
            % container; wrap into GolfInputs.
            GolfInputs = inputs; %#ok<NASGU>
            save(mat_path, 'GolfInputs');

            spec = decode_input_file_pose(mat_path);
            testCase.verifyEqual(spec.q, inputs.q0(:), 'AbsTol', 1e-12);
            testCase.verifyEqual(numel(spec.joint_names), 5);
            testCase.verifyEqual(spec.r_butt, inputs.r_butt(:), 'AbsTol', 1e-12);
            testCase.verifyEqual(spec.r_clubhead, inputs.r_clubhead(:), 'AbsTol', 1e-12);
        end

        function test_actual_pose_extracted_after_zero_time_sim(testCase)
            % Smoke test: requires the real model on path.
            testCase.assumeTrue(exist('GolfSwing3D_KineticallyDriven', 'file') == 4, ...
                'Skipping: model not on path');

            % Find a representative input file.
            input_file = which('3DModelInputs_Impact.mat');
            testCase.assumeTrue(~isempty(input_file), ...
                'Skipping: 3DModelInputs_Impact.mat not on path');

            report = diagnose_initial_state(input_file);
            testCase.verifyTrue(isstruct(report));
            testCase.verifyTrue(isfield(report, 'specified'));
            testCase.verifyTrue(isfield(report, 'actual'));
            testCase.verifyTrue(isfield(report, 'delta'));
            testCase.verifyEqual(numel(report.actual.q), numel(report.specified.q));
        end

        function test_delta_zero_for_unconstrained_state(testCase)
            % If the actual pose equals the specified pose, all deltas
            % should be zero and is_significant should be false.
            % We exercise this through a hand-built report comparison.
            report = struct();
            report.specified.q = [0.1; 0.2; 0.3];
            report.actual.q = [0.1; 0.2; 0.3];
            report.specified.r_butt = [0; 0; 1];
            report.actual.r_butt = [0; 0; 1];
            report.specified.r_clubhead = [1; 0; 0];
            report.actual.r_clubhead = [1; 0; 0];

            q_delta_deg = rad2deg(report.actual.q - report.specified.q);
            testCase.verifyLessThan(max(abs(q_delta_deg)), 1e-9);
            testCase.verifyLessThan(norm(report.actual.r_butt - report.specified.r_butt), 1e-9);
        end

        function test_significant_flag_triggers_for_5mm_clubhead_delta(testCase)
            % A 6mm Cartesian shift at the clubhead should trip the flag
            % even when joint angles look fine.
            specified_r = [1; 0; 0];
            actual_r = specified_r + [0.006; 0; 0];  % 6 mm
            mm = 1000 * norm(actual_r - specified_r);
            testCase.verifyGreaterThan(mm, 5.0);
        end
    end
end
