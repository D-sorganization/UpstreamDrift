classdef test_gs3dx_fit_track < matlab.unittest.TestCase
%TEST_GS3DX_FIT_TRACK  Upper body tracking the capture, GS3DX_FitTrack (#10979).
%
%   GS3DX_FitTrack is GS3DX_FitLegs with its twelve upper-body charts
%   driven by a learned feedforward plus PD (GS3DX_BUILD_FIT_TRACK,
%   GS3DX_TRACK_LEARN) toward the whole-body IK of the capture.  The model
%   is built from a whole-trial regularized IK and a learning run of hours
%   (docs/FIT.md), so this test checks the saved model and replays a short
%   stretch rather than rebuilding it.

    properties
        info struct
        mdl char
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
            testCase.mdl = char(gs3dx_names().variants.fit_track);
            testCase.assumeTrue(isfile(fullfile(testCase.info.models_dir, [testCase.mdl '.slx'])), ...
                'GS3DX_FitTrack is built by GS3DX_BUILD_FIT_TRACK (docs/FIT.md)');
            load_system(testCase.mdl);
            testCase.addTeardown(@() close_system(testCase.mdl, 0));
        end
    end

    methods (Test)
        function track_torque_interpolates_and_holds(testCase)
            T = [0 1 2];
            A = [0 10 20; 0 -10 -20];
            R = [10 10 10; -10 -10 -10];
            F = [1 2 3; 4 5 6];
            Kp = [2; 3]; Kd = [0.5; 0.1];
            tau = gs3dx_track_torque(0.5, T, A, R, F, Kp, Kd, [1; 1], [0; 0]);
            testCase.verifyEqual(tau, [1.5 + 2 * (5 - 1) + 0.5 * 10; 4.5 + 3 * (-5 - 1) + 0.1 * -10], 'AbsTol', 1e-12);
            testCase.verifyEqual(gs3dx_track_torque(-1, T, A, R, F, Kp, Kd, [0; 0], [10; -10]), F(:, 1), 'AbsTol', 1e-12);
            testCase.verifyEqual(gs3dx_track_torque(5, T, A, R, F, Kp, Kd, [20; -20], [10; -10]), F(:, 3), 'AbsTol', 1e-12);
        end

        function gains_are_critically_damped_on_the_inertia(testCase)
            g = gs3dx_track_gains();
            w = 2 * pi * 6;
            testCase.verifyEqual(g.Spine(1, :), deg2rad([3 * w ^ 2, 2 * 3 * w]), 'RelTol', 1e-12);
            testCase.verifySize(g.LS, [3 2]);
            testCase.verifySize(g.Torso, [1 2]);
            for f = fieldnames(g).'
                testCase.verifyEqual(g.(f{1})(:, 2) ./ g.(f{1})(:, 1), repmat(2 / w, size(g.(f{1}), 1), 1), ...
                    'RelTol', 1e-12, f{1});
            end
        end

        function every_chart_reads_its_own_joint(testCase)
            % The original model feeds the LW chart the left scapula's angles
            % and the LE, LF, RF and RW charts tags no Goto writes.
            m = testCase.mdl;
            for j = gs3dx_upper_body_joints().'
                ch = find_system([m '/' j.prefix ' Input Function'], 'SearchDepth', 1, 'SFBlockType', 'MATLAB Function');
                chart = sfroot().find('-isa', 'Stateflow.EMChart', 'Path', ch{1});
                ph = get_param(ch{1}, 'PortHandles');
                ax = cellstr(j.axes(:)).';
                if isempty(ax)
                    ax = {''};
                end
                for kind = {'Position', 'Velocity'}
                    for a = ax
                        d = chart.find('-isa', 'Stateflow.Data', 'Name', [j.prefix kind{1} a{1}], 'Scope', 'Input');
                        src = get_param(get_param(get_param(ph.Inport(d.Port), 'Line'), 'SrcPortHandle'), 'Parent');
                        testCase.verifyEqual(get_param(src, 'GotoTag'), [j.prefix 'Angular' kind{1} a{1}], ...
                            [j.prefix ' ' kind{1} a{1}]);
                    end
                end
                testCase.verifySubstring(chart.Script, 'if UpperBodyTracking', j.prefix);
            end
        end

        function model_holds_the_tracking_data(testCase)
            m = testCase.mdl;
            src = char(gs3dx_names().variants.fit_legs);
            load_system(src);
            testCase.addTeardown(@() close_system(src, 0));
            count = @(x) numel(find_system(x, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
                'LookInsideSubsystemReference', 'on', 'Virtual', 'off'));
            testCase.verifyEqual(count(m), count(src), 'no block added');
            ws = get_param(m, 'ModelWorkspace');
            testCase.verifyEqual(ws.getVariable('UpperBodyTracking'), 1);
            t = ws.getVariable('UpperBodyTrackTime');
            testCase.verifyEqual(t, ws.getVariable('LegReferenceTime'), 'AbsTol', 1e-9);
            for j = gs3dx_upper_body_joints().'
                n = max(1, numel(j.axes));
                for v = {'TrackAngle', 'TrackRate', 'TrackTorque'}
                    testCase.verifySize(ws.getVariable([j.prefix v{1}]), [n numel(t)], [j.prefix v{1}]);
                end
                testCase.verifyGreaterThan(max(abs(ws.getVariable([j.prefix 'TrackTorque'])), [], 'all'), 0, ...
                    [j.prefix ' has no learned feedforward']);
            end
        end

        function learned_feedforward_tracks_the_capture(testCase)
            % One simulation of the first 0.3 s with the saved feedforward:
            % the upper body follows the capture with the PD nearly idle.  The
            % bounds pin the 2026-09-27 result (0.283 deg, worst 0.491 deg,
            % 16.7 N*m; docs/FIT.md).
            out = gs3dx_track_learn(testCase.info, iterations=1, stop_time=0.3);
            fprintf('0.3 s replay: angle RMS %.2f deg (worst joint %.2f), PD RMS %.1f N*m\n', ...
                out.error, max(out.joint_error), out.pd);
            testCase.verifyEmpty(out.status, 'the simulation stopped');
            testCase.verifyLessThan(out.error, 0.35, 'angle RMS');
            testCase.verifyLessThan(max(out.joint_error), 0.6, 'worst joint RMS');
            testCase.verifyLessThan(out.pd, 20, 'PD RMS');
        end
    end
end
