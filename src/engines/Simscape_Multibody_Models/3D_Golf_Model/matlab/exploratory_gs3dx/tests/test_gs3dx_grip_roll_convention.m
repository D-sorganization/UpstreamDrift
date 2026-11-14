classdef test_gs3dx_grip_roll_convention < matlab.unittest.TestCase
%TEST_GS3DX_GRIP_ROLL_CONVENTION  SHOW-A14 (#14739) checks of the A7 grip zero (#14731).
%
%   The A7 hand roll (0 = back of the wrist on top of the shaft, + = CCW,
%   lead 15 deg, trail 195 deg) lives in
%   src/shared/python/motion_matching/grip_roll.py. This suite does not copy
%   that formula. It checks that the gs3dx grip topology frames the model
%   consumes are club-attached, so the A7 wrist re-zero cannot move them.
%   The address hand-roll measurement on GS3DX_Human is the deferred R2025b
%   step recorded in docs/development/deferred-catalog-bundle.json.

    properties (Constant)
        Tol = 1e-6
    end

    properties
        Topo
    end

    methods (TestClassSetup)
        function setup(testCase)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(root, 'tools'));
            testCase.Topo = gs3dx_read_grip_topology();
        end
    end

    methods (Test)
        function test_hand_frames_follow_the_club_shaft(testCase)
            % Each hand frame's shaft axis is the club's +y shaft.
            for side = {'L', 'R'}
                frame = testCase.Topo.hand_frames.(side{1});
                testCase.verifyEqual(frame.shaft_axis(:)', [0 1 0], ...
                    'AbsTol', testCase.Tol, sprintf('%s shaft_axis', side{1}));
            end
        end

        function test_hand_frames_are_proper_rotations(testCase)
            for side = {'L', 'R'}
                R = testCase.Topo.hand_frames.(side{1}).rotation;
                testCase.verifyEqual(R' * R, eye(3), 'AbsTol', testCase.Tol, ...
                    sprintf('%s rotation is orthonormal', side{1}));
                testCase.verifyEqual(det(R), 1, 'AbsTol', testCase.Tol, ...
                    sprintf('%s rotation is right-handed', side{1}));
            end
        end

        function test_both_hands_share_the_club_roll(testCase)
            % The contact frames carry one club roll for both hands. The A7
            % hand roll (lead 15 deg, trail 195 deg) is a wrist quantity,
            % not a difference between these frames.
            testCase.verifyEqual(testCase.Topo.hand_frames.L.rotation, ...
                testCase.Topo.hand_frames.R.rotation, 'AbsTol', testCase.Tol);
        end
    end
end
