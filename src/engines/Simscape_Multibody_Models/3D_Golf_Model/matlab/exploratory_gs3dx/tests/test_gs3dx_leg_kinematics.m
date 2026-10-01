classdef test_gs3dx_leg_kinematics < matlab.unittest.TestCase
%TEST_GS3DX_LEG_KINEMATICS  GS3DX_LEG_FK against Simscape, and GS3DX_LEG_IK.
%
%   FK: in GS3DX_FullBody the feet are welded, so the foot pose the FK
%   computes from the logged pelvis pose and leg angles must equal the weld
%   target (LFootGroundRotation/Offset) to round-off.  IK: it recovers
%   poses FK produced, and reports an out-of-reach target as an error.

    properties
        info struct
        geom struct
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
            p = gs3dx_leg_table().params;
            testCase.geom = struct('mount_R', eye(3), 'mount_p', [0; 0.09; -0.1], ...
                'thigh', p.ThighLength, 'shank', p.ShankLength);
        end
    end

    methods (Test)
        function fk_matches_welded_fullbody_feet(testCase)
            full = char(gs3dx_names().variants.fullbody);
            testCase.assumeTrue(isfile(fullfile(testCase.info.models_dir, [full '.slx'])), 'GS3DX_FullBody not built');
            load_system(full);
            ws = get_param(full, 'ModelWorkspace');
            p = gs3dx_leg_table().params;
            for P = 'LR'   % read before simulating: GS3DX_STANCE_FRAMES closes the model
                geom.(P) = struct('mount_R', ws.getVariable([P 'HipMountRotation']), ...
                    'mount_p', ws.getVariable([P 'HipMountOffset']), 'thigh', p.ThighLength, 'shank', p.ShankLength);
                weld.(P) = {ws.getVariable([P 'FootGroundRotation']), ws.getVariable([P 'FootGroundOffset'])};
            end
            vars = gs3dx_drive(testCase.info, "impact", full);
            frames = gs3dx_stance_frames(full, vars, stop_time=0.002);
            logs = frames.out.logsout;
            s = frames.series;
            for P = 'LR'
                hip = logs.get([P 'HipLogs']).Values;
                knee = logs.get([P 'KneeLogs']).Values;
                ankle = logs.get([P 'AnkleLogs']).Values;
                k = numel(s.t);
                q = [local_last(hip.AngularPositionX); local_last(hip.AngularPositionY); ...
                     local_last(hip.AngularPosition_Z); local_last(knee.AngularPosition); ...
                     local_last(ankle.AngularPositionX); local_last(ankle.AngularPositionY)];
                [R, x] = gs3dx_leg_fk(geom.(P), s.pelvis_R(:, :, k), s.pelvis_p(:, k), q);
                testCase.verifyEqual(R, weld.(P){1}, 'AbsTol', 1e-9, [P ' rotation']);
                testCase.verifyEqual(x, weld.(P){2}, 'AbsTol', 1e-9, [P ' position']);
            end
        end

        function ik_recovers_fk_pose(testCase)
            q_true = [5; -20; 3; -40; 2; -18];
            pR = eye(3);
            pp = [0; 0; 1];
            [fR, fp] = gs3dx_leg_fk(testCase.geom, pR, pp, q_true);
            [q, res] = gs3dx_leg_ik(testCase.geom, pR, pp, fR, fp, q_true + [3; -4; 2; 5; -2; 3]);
            testCase.verifyLessThan(res, 1e-9);
            [R, x] = gs3dx_leg_fk(testCase.geom, pR, pp, q);
            testCase.verifyEqual(R, fR, 'AbsTol', 1e-9);
            testCase.verifyEqual(x, fp, 'AbsTol', 1e-9);
        end

        function ik_tracks_a_moving_pelvis(testCase)
            q0 = [0; -15; 0; -30; 0; -15];
            [fR, fp] = gs3dx_leg_fk(testCase.geom, eye(3), [0; 0; 1], q0);
            n = 5;
            pR = repmat(eye(3), 1, 1, n);
            pp = [0; 0; 1] + [linspace(0, 0.02, n); zeros(1, n); linspace(0, -0.02, n)];
            [q, res] = gs3dx_leg_ik(testCase.geom, pR, pp, fR, fp, q0);
            testCase.verifySize(q, [6 n]);
            testCase.verifyLessThan(max(res), 1e-9);
            testCase.verifyLessThan(q(4, :), 0, 'knee stays on the forward-bent branch');
        end

        function ik_rejects_out_of_reach_target(testCase)
            [fR, ~] = gs3dx_leg_fk(testCase.geom, eye(3), [0; 0; 1], zeros(6, 1));
            testCase.verifyError(@() gs3dx_leg_ik(testCase.geom, eye(3), [0; 0; 1], fR, [0; 0; -5], zeros(6, 1)), ...
                'gs3dx:ik');
        end
    end
end

function x = local_last(ts)
    d = ts.Data;
    x = double(d(end));
end
