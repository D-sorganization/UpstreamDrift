classdef test_gs3dx_track_learn_configuration < matlab.unittest.TestCase
%TEST_GS3DX_TRACK_LEARN_CONFIGURATION  Strengthened contract tests for GS3DX_TRACK_LEARN.
%
%   Verifies:
%     1. Configuration options: model and initialization ("legacy" | "configured").
%     2. Model ownership & lifecycle: preserves caller-loaded dirty models even on error;
%        valid native execution and helper-owned closing are checked separately.
%     3. Strict preflight contract ('gs3dx:tracklearn') with verified error messages:
%        - Tracking enable flag: UpperBodyTracking must equal 1.
%        - Time vector: real, finite, monotonically increasing, uniformly spaced, > 12 samples.
%        - Cutoff frequency: positive, finite, strictly below Nyquist.
%        - Learning gain: positive, finite, real scalar.
%        - Upper-body matrices (A, R, F) and feedforward overrides: real, finite, correct shape.
%        - Tracking gains (Kp, Kd): real, finite, non-negative, correct dimensions [n_axes, 1].
%        - Unknown feedforward override prefixes rejected.
%     4. Caller workspace and dirty state preservation on error without swallowing failures.
%
%   Limitation Note: Full valid configured-mode simulation with ground contact and
%   closed-loop balance requires native R2025b Simscape Multibody execution.

    properties
        FixtureModel (1,1) string = ""
    end

    methods (TestMethodSetup)
        function createTemporaryModelFixture(testCase)
            pid = feature('getpid');
            rnd = randi(1e6);
            testCase.FixtureModel = sprintf('tmp_gs3dx_fixture_%d_%d', pid, rnd);
            new_system(char(testCase.FixtureModel));
            set_param(char(testCase.FixtureModel), 'Dirty', 'on');
        end
    end

    methods (TestMethodTeardown)
        function cleanupTemporaryModelFixture(testCase)
            mdl = char(testCase.FixtureModel);
            if bdIsLoaded(mdl)
                close_system(mdl, 0);
            end
        end
    end

    methods (Test)
        function test_configured_mode_rejects_empty_model_name(testCase)
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', ""), ...
                "model");
        end

        function test_configured_mode_rejects_unloaded_model(testCase)
            unloaded_mdl = "nonexistent_unloaded_gs3dx_model";
            testCase.assertFalse(bdIsLoaded(char(unloaded_mdl)));
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', unloaded_mdl), ...
                "loaded");
        end

        function test_configured_mode_rejects_missing_model_text(testCase)
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(),'initialization',"configured",'model',string(missing)), ...
                "missing");
        end

        function test_legacy_mode_rejects_explicit_different_model(testCase)
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "legacy", 'model', "GS3DX_Human"), ...
                "model");
        end

        function test_preflight_rejects_missing_upper_body_tracking_flag(testCase)
            mdl = char(testCase.FixtureModel);
            ws = get_param(mdl, 'ModelWorkspace');
            testCase.populateValidUpperSpecWorkspace(ws, 50);
            ws.evalin('clear UpperBodyTracking');
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl), ...
                "UpperBodyTracking");
        end

        function test_preflight_rejects_disabled_upper_body_tracking(testCase)
            mdl = char(testCase.FixtureModel);
            ws = get_param(mdl, 'ModelWorkspace');
            testCase.populateValidUpperSpecWorkspace(ws, 50);
            ws.assignin('UpperBodyTracking', 0);
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl), ...
                "UpperBodyTracking");
        end

        function test_preflight_rejects_missing_upper_body_track_time(testCase)
            mdl = char(testCase.FixtureModel);
            ws = get_param(mdl, 'ModelWorkspace');
            testCase.populateValidUpperSpecWorkspace(ws, 50);
            ws.evalin('clear UpperBodyTrackTime');
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl), ...
                "UpperBodyTrackTime");
        end

        function test_preflight_rejects_nonfinite_or_complex_time(testCase)
            mdl = char(testCase.FixtureModel);
            ws = get_param(mdl, 'ModelWorkspace');
            testCase.populateValidUpperSpecWorkspace(ws, 50);
            T_orig = ws.getVariable('UpperBodyTrackTime');

            % Test NaN time
            T_nan = T_orig; T_nan(10) = NaN;
            ws.assignin('UpperBodyTrackTime', T_nan);
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl), ...
                "UpperBodyTrackTime");

            % Test complex time
            T_cplx = T_orig; T_cplx(10) = T_orig(10) + 1i;
            ws.assignin('UpperBodyTrackTime', T_cplx);
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl), ...
                "UpperBodyTrackTime");
        end

        function test_preflight_rejects_non_increasing_time(testCase)
            mdl = char(testCase.FixtureModel);
            ws = get_param(mdl, 'ModelWorkspace');
            testCase.populateValidUpperSpecWorkspace(ws, 30);
            T = ws.getVariable('UpperBodyTrackTime');
            T(10) = T(9);
            ws.assignin('UpperBodyTrackTime', T);
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl), ...
                "UpperBodyTrackTime");
        end

        function test_preflight_rejects_non_uniformly_spaced_time(testCase)
            mdl = char(testCase.FixtureModel);
            ws = get_param(mdl, 'ModelWorkspace');
            testCase.populateValidUpperSpecWorkspace(ws, 30);
            T = ws.getVariable('UpperBodyTrackTime');
            T(15) = T(15) + 0.005;
            ws.assignin('UpperBodyTrackTime', T);
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl), ...
                "UpperBodyTrackTime");
        end

        function test_preflight_rejects_insufficient_samples_for_filtfilt(testCase)
            mdl = char(testCase.FixtureModel);
            ws = get_param(mdl, 'ModelWorkspace');
            testCase.populateValidUpperSpecWorkspace(ws, 8);
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl), ...
                "filtfilt");
        end

        function test_preflight_rejects_cutoff_at_or_above_nyquist_or_nonfinite(testCase)
            mdl = char(testCase.FixtureModel);
            ws = get_param(mdl, 'ModelWorkspace');
            testCase.populateValidUpperSpecWorkspace(ws, 50);

            % Exceeds Nyquist
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl, 'cutoff_hz', 300), ...
                "cutoff");

            % Exactly at Nyquist
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl, 'cutoff_hz', 250), ...
                "cutoff");

            % Non-positive or nonfinite cutoff
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl, 'cutoff_hz', NaN), ...
                "cutoff");
        end

        function test_preflight_rejects_nonfinite_or_nonpositive_learning_gain(testCase)
            mdl = char(testCase.FixtureModel);
            ws = get_param(mdl, 'ModelWorkspace');
            testCase.populateValidUpperSpecWorkspace(ws, 50);

            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl, 'learning_gain', NaN), ...
                "learning_gain");
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl, 'learning_gain', -0.5), ...
                "learning_gain");
        end

        function test_preflight_rejects_unknown_feedforward_override_prefix(testCase)
            mdl = char(testCase.FixtureModel);
            ws = get_param(mdl, 'ModelWorkspace');
            testCase.populateValidUpperSpecWorkspace(ws, 50);

            invalid_ff = struct('NonExistentJoint', zeros(1, 50));
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl, 'feedforward', invalid_ff), ...
                "prefix");
        end

        function test_preflight_rejects_mismatched_feedforward_override_shape(testCase)
            mdl = char(testCase.FixtureModel);
            ws = get_param(mdl, 'ModelWorkspace');
            testCase.populateValidUpperSpecWorkspace(ws, 50);

            % Torso is revolute (1 axis). Override with 2 rows.
            bad_ff = struct('Torso', zeros(2, 50));
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl, 'feedforward', bad_ff), ...
                "Torso");
        end

        function test_preflight_rejects_corrupted_reference_angle_rate_torque(testCase)
            mdl = char(testCase.FixtureModel);
            ws = get_param(mdl, 'ModelWorkspace');
            testCase.populateValidUpperSpecWorkspace(ws, 50);

            % Mismatched dimension in angle (Spine is 2 axes, corrupt to 3)
            ws.assignin('SpineTrackAngle', zeros(3, 50));
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl), ...
                "SpineTrackAngle");

            % Nonfinite/complex in reference rate
            testCase.populateValidUpperSpecWorkspace(ws, 50);
            R = ws.getVariable('LETrackRate');
            R(1, 5) = NaN;
            ws.assignin('LETrackRate', R);
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl), ...
                "LETrackRate");

            % Nonfinite/complex in reference torque
            testCase.populateValidUpperSpecWorkspace(ws, 50);
            F = ws.getVariable('RFTrackTorque');
            F(1, 5) = 2 + 3i;
            ws.assignin('RFTrackTorque', F);
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl), ...
                "RFTrackTorque");
        end

        function test_preflight_rejects_negative_nonfinite_or_nd_gains(testCase)
            mdl = char(testCase.FixtureModel);
            ws = get_param(mdl, 'ModelWorkspace');
            testCase.populateValidUpperSpecWorkspace(ws, 50);

            % LS (Left Shoulder) has 3 axes: provide scalar instead of 3x1
            ws.assignin('LSTrackKp', 100);
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl), ...
                "LSTrackKp");

            % Negative gain
            ws.assignin('LSTrackKp', [-10; 100; 100]);
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl), ...
                "LSTrackKp");

            % N-D / 3D gain array
            ws.assignin('LSTrackKp',ones(3,1)*100);
            ws.assignin('LSTrackKd', zeros(3, 1, 2));
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl), ...
                "LSTrackKd");
        end

        function test_caller_loaded_dirty_model_survives_errors(testCase)
            mdl = char(testCase.FixtureModel);
            ws = get_param(mdl, 'ModelWorkspace');
            testCase.populateValidUpperSpecWorkspace(ws, 50);

            sentinel_val = 1337.42;
            ws.assignin('CallerDirtySentinel', sentinel_val);

            % Trigger preflight error (cutoff above Nyquist) and assert exact failure
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl, 'cutoff_hz', 9999), ...
                "cutoff");

            % Assert model remains loaded and sentinel variable is intact
            testCase.assertTrue(bdIsLoaded(mdl), 'Caller model must remain loaded after helper error');
            testCase.assertEqual(get_param(mdl, 'Dirty'), 'on');
            testCase.assertTrue(ws.hasVariable('CallerDirtySentinel'), 'Caller workspace variables must survive');
            testCase.assertEqual(ws.getVariable('CallerDirtySentinel'), sentinel_val);
        end

        function test_preflight_fails_before_dynamics_on_fixture(testCase)
            mdl = char(testCase.FixtureModel);
            ws = get_param(mdl, 'ModelWorkspace');
            testCase.populateValidUpperSpecWorkspace(ws, 50);

            % Invalid stop_time outside reference
            testCase.assertTrackLearnError( ...
                @() gs3dx_track_learn(struct(), 'initialization', "configured", 'model', mdl, 'stop_time', -1.0), ...
                "stop");
        end
    end

    methods (Access = private)
        function assertTrackLearnError(testCase, fn, expectedSubstrs)
            % Executes fn, asserting 'gs3dx:tracklearn' with specific message content
            try
                fn();
                testCase.verifyFail('Expected gs3dx:tracklearn error, but no error was thrown.');
            catch ME
                testCase.assertEqual(ME.identifier, 'gs3dx:tracklearn', ...
                    sprintf('Expected ID "gs3dx:tracklearn", got "%s": %s', ME.identifier, ME.message));
                if nargin >= 3 && ~isempty(expectedSubstrs)
                    substrs = string(expectedSubstrs);
                    matched = any(arrayfun(@(s) contains(ME.message, s, 'IgnoreCase', true), substrs));
                    testCase.assertTrue(matched, ...
                        sprintf('Expected error message to contain one of [%s], but got: "%s"', ...
                            strjoin(substrs, ', '), ME.message));
                end
            end
        end

        function populateValidUpperSpecWorkspace(testCase, ws, num_samples)
            if nargin < 3
                num_samples = 50;
            end
            dt = 0.002;
            T = (0:num_samples-1)' * dt;
            ws.assignin('UpperBodyTrackTime', T);
            ws.assignin('UpperBodyTracking', 1);

            spec = gs3dx_upper_body_joints();

            for k = 1:numel(spec)
                P = spec(k).prefix;
                axes_str = strtrim(spec(k).axes);
                if isempty(axes_str)
                    na = 1;
                else
                    na = numel(axes_str);
                end

                ws.assignin([P 'TrackAngle'], zeros(na, num_samples));
                ws.assignin([P 'TrackRate'], zeros(na, num_samples));
                ws.assignin([P 'TrackTorque'], zeros(na, num_samples));
                ws.assignin([P 'TrackKp'], ones(na, 1) * 100);
                ws.assignin([P 'TrackKd'], ones(na, 1) * 10);
            end
        end
    end
end
