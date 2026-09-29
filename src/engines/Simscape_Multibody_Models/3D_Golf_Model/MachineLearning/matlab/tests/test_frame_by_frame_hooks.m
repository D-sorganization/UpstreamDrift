classdef test_frame_by_frame_hooks < matlab.unittest.TestCase
%TEST_FRAME_BY_FRAME_HOOKS  Unit tests for the +frame_search hook helpers.
%
%   These tests cover the pure helpers used by the frame-by-frame torque
%   search Simscape stepping hooks. Helpers that require Simulink are
%   exercised via a fake "simOut" struct mimicking Simulink.SimulationOutput
%   so the suite runs without the toolbox.
%
%   GitHub issue: #3977 (parent #3976).

    methods (TestClassSetup)
        function add_paths(testCase)
            here = fileparts(mfilename('fullpath'));
            mlDir = fileparts(here);   % .../MachineLearning/matlab
            addpath(mlDir);
            testCase.addTeardown(@() rmpath(mlDir));
        end
    end

    methods (Test)

        %% ---- parse_target_column ---------------------------------------
        function test_parse_simple_grouped_column(testCase)
            info = frame_search.parse_target_column('ClubLogs_CHGlobalPosition_1');
            testCase.verifyEqual(info.group, 'ClubLogs');
            testCase.verifyEqual(info.signal, 'CHGlobalPosition');
            testCase.verifyEqual(info.busPath, ["ClubLogs", "CHGlobalPosition"]);
            testCase.verifyEqual(info.index, 1);
        end

        function test_parse_unindexed_column_yields_nan_index(testCase)
            info = frame_search.parse_target_column('TorsoLogs_AngularPosition');
            testCase.verifyEqual(info.group, 'TorsoLogs');
            testCase.verifyEqual(info.signal, 'AngularPosition');
            testCase.verifyTrue(isnan(info.index));
        end

        function test_parse_signal_with_underscore_preserved(testCase)
            % LSLogs_AngularPosition_Z legacy column has underscore in signal.
            info = frame_search.parse_target_column('LSLogs_AngularPosition_Z');
            testCase.verifyEqual(info.group, 'LSLogs');
            % The trailing _Z is not a numeric index, so the parser keeps it.
            testCase.verifyEqual(info.signal, 'AngularPosition_Z');
            testCase.verifyTrue(isnan(info.index));
        end

        %% ---- control_column_to_polynomial_base ------------------------
        function test_known_control_columns_map(testCase)
            cases = { ...
                "LSLogs_ActuatorTorqueX",  "LSInputX"; ...
                "RSLogs_ActuatorTorqueZ",  "RSInputZ"; ...
                "HipLogs_HipTorqueXInput", "HipInputX"; ...
                "HipLogs_TranslationForceYInput", "TranslationInputY"};
            for k = 1:size(cases, 1)
                got = frame_search.control_column_to_polynomial_base(cases{k, 1});
                testCase.verifyEqual(got, cases{k, 2});
            end
        end

        function test_unknown_control_column_returns_empty(testCase)
            got = frame_search.control_column_to_polynomial_base("NotAColumn");
            testCase.verifyEqual(got, "");
        end

        %% ---- apply_constant_torque ------------------------------------
        function test_apply_constant_torque_sets_only_constant_term(testCase)
            controls = {'LSLogs_ActuatorTorqueX', 'RSLogs_ActuatorTorqueZ'};
            torque = [12.5, -3.75];
            vars = frame_search.apply_constant_torque(torque, controls);

            % Higher-order coefficients zero, constant 'G' = torque.
            testCase.verifyEqual(vars.LSInputXA, 0.0);
            testCase.verifyEqual(vars.LSInputXF, 0.0);
            testCase.verifyEqual(vars.LSInputXG, 12.5);
            testCase.verifyEqual(vars.RSInputZG, -3.75);
            testCase.verifyEqual(vars.RSInputZB, 0.0);

            % Each control contributes 7 variables (A..G).
            testCase.verifyEqual(numel(fieldnames(vars)), 14);
        end

        function test_apply_constant_torque_size_mismatch_errors(testCase)
            testCase.verifyError(@() frame_search.apply_constant_torque( ...
                [1.0, 2.0], {'LSLogs_ActuatorTorqueX'}), ...
                'frame_search:apply_constant_torque:sizeMismatch');
        end

        function test_apply_constant_torque_unknown_column_errors(testCase)
            testCase.verifyError(@() frame_search.apply_constant_torque( ...
                [1.0], {'NotAColumn'}), ...
                'frame_search:apply_constant_torque:unknownControl');
        end

        %% ---- frame_horizon --------------------------------------------
        function test_horizon_uses_target_time_when_available(testCase)
            cfg = struct('search', struct('horizon_frames', 1), ...
                         'validation', struct('median_step_seconds', 0.001));
            state = struct('time', 0.010);
            target = struct('time', 0.012);
            [t0, t1] = frame_search.frame_horizon(state, target, cfg);
            testCase.verifyEqual(t0, 0.010);
            testCase.verifyEqual(t1, 0.012);
        end

        function test_horizon_falls_back_to_min_step(testCase)
            cfg = struct('search', struct('horizon_frames', 2), ...
                         'validation', struct('median_step_seconds', 0.005));
            state = struct('time', 0.10);
            target = struct();   % no time field
            [t0, t1] = frame_search.frame_horizon(state, target, cfg);
            testCase.verifyEqual(t0, 0.10);
            testCase.verifyEqual(t1, 0.10 + 2 * 0.005, 'AbsTol', 1e-12);
        end

        function test_horizon_protects_against_non_monotonic_target(testCase)
            cfg = struct('search', struct('horizon_frames', 1), ...
                         'validation', struct('median_step_seconds', 0.001));
            state = struct('time', 0.020);
            target = struct('time', 0.020);   % equal => collapse
            [t0, t1] = frame_search.frame_horizon(state, target, cfg);
            testCase.verifyGreaterThan(t1, t0);
        end

        %% ---- lookup_signal_value via fake simOut ----------------------
        function test_lookup_combined_signal_bus_with_index(testCase)
            simOut = struct();
            simOut.CombinedSignalBus = struct( ...
                'ClubLogs', struct( ...
                    'CHGlobalPosition', [1.0 2.0 3.0; 4.0 5.0 6.0]));
            info = frame_search.parse_target_column('ClubLogs_CHGlobalPosition_2');
            v = frame_search.lookup_signal_value(simOut, info, []);
            testCase.verifyEqual(v, 5.0);
        end

        function test_lookup_returns_last_sample_by_default(testCase)
            simOut = struct();
            simOut.CombinedSignalBus = struct( ...
                'ClubLogs', struct( ...
                    'CHGlobalPosition', [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]));
            info = frame_search.parse_target_column('ClubLogs_CHGlobalPosition_3');
            v = frame_search.lookup_signal_value(simOut, info, []);
            testCase.verifyEqual(v, 9.0);
        end

        function test_lookup_missing_column_errors(testCase)
            simOut = struct('CombinedSignalBus', struct());
            info = frame_search.parse_target_column('ClubLogs_CHGlobalPosition_1');
            testCase.verifyError(@() frame_search.lookup_signal_value( ...
                simOut, info, []), 'frame_search:lookup_signal_value:notFound');
        end

        %% ---- extract_predicted ----------------------------------------
        function test_extract_predicted_pulls_each_target(testCase)
            simOut = struct();
            simOut.CombinedSignalBus = struct( ...
                'ClubLogs', struct( ...
                    'CHGlobalPosition', [0.1 0.2 0.3; 0.4 0.5 0.6], ...
                    'CHGlobalVelocity', [10.0 20.0 30.0]));
            target = struct( ...
                'time', 0.123, ...
                'ClubLogs_CHGlobalPosition_1', 0.0, ...
                'ClubLogs_CHGlobalVelocity_2', 0.0);
            predicted = frame_search.extract_predicted(simOut, target);

            testCase.verifyEqual(predicted.ClubLogs_CHGlobalPosition_1, 0.4);
            testCase.verifyEqual(predicted.ClubLogs_CHGlobalVelocity_2, 20.0);
            testCase.verifyFalse(isfield(predicted, 'time'));
        end

        function test_extract_predicted_missing_column_errors(testCase)
            simOut = struct('CombinedSignalBus', struct());
            target = struct('ClubLogs_CHGlobalPosition_1', 0.0);
            testCase.verifyError(@() frame_search.extract_predicted( ...
                simOut, target), 'frame_search:extract_predicted:missingColumn');
        end

        %% ---- extract_state --------------------------------------------
        function test_extract_state_records_xfinal_and_time(testCase)
            simOut = struct();
            simOut.xFinal = struct('placeholder', 1);
            simOut.tout = (0:0.001:0.020).';
            simOut.CombinedSignalBus = struct();
            cfg = struct('columns', struct());
            previous = struct('frame_index', 3, 'starting_state_file', "");
            next = frame_search.extract_state(simOut, previous, cfg);

            testCase.verifyTrue(isstruct(next.xFinal));
            testCase.verifyEqual(next.time, 0.020, 'AbsTol', 1e-12);
            testCase.verifyEqual(next.frame_index, 3);  % runner increments
        end

    end
end
