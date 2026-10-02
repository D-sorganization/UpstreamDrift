classdef test_gs3dx_contact_wrench < matlab.unittest.TestCase
%TEST_GS3DX_CONTACT_WRENCH  Foot-width contact wrench (#11173).
%
%   Synthetic geometry only; no capture, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function vertical_grf_at_com(testCase)
            f = [0; 0; 800];
            com = [0; 0; 1];
            points = local_square_points();
            w = gs3dx_contact_wrench(f, com, points, 0.8);
            testCase.verifyEqual(w.moment, 0, 'AbsTol', 1e-9);
            testCase.verifyEqual(w.force, 0, 'AbsTol', 1e-9);
            testCase.verifyEqual(sum(w.f, 2), f, 'AbsTol', 1e-6);
            testCase.verifyEqual(local_contact_moment(com, points, w.f), 0, 'AbsTol', 1e-4);
        end

        function shear_cop_outside_hull(testCase)
            f = [160; 0; 800];
            com = [0; 0; 1];
            points = local_square_points();
            w = gs3dx_contact_wrench(f, com, points, 0.8);
            testCase.verifyEqual(w.moment, 80, 'AbsTol', 1e-9);
            testCase.verifyEqual(w.force, 0, 'AbsTol', 1e-9);
            testCase.verifyEqual(sum(w.f, 2), f, 'AbsTol', 1e-6);
            testCase.verifyEqual(local_contact_moment(com, points, w.f), 80, 'AbsTol', 1e-4);
        end

        function negative_vertical_grf(testCase)
            f = [0; 0; -100];
            com = [0; 0; 1];
            points = local_square_points();
            w = gs3dx_contact_wrench(f, com, points, 0.8);
            testCase.verifyEqual(w.f, zeros(size(w.f)), 'AbsTol', 1e-9);
            testCase.verifyEqual(w.force, 100, 'AbsTol', 1e-9);
            testCase.verifyEqual(w.moment, 0, 'AbsTol', 1e-9);
        end

        function diagonal_shear_cop_outside_hull(testCase)
            f = [160; 160; 800];
            com = [0; 0; 1];
            points = local_square_points();
            w = gs3dx_contact_wrench(f, com, points, 0.8);
            testCase.verifyEqual(w.force, 0, 'AbsTol', 1e-9);
            testCase.verifyEqual(sum(w.f, 2), f, 'AbsTol', 1e-6);
            [mTip, mVert] = local_contact_moment_parts(com, points, w.f);
            testCase.verifyEqual(mTip, w.moment, 'AbsTol', 1e-4);
            testCase.verifyGreaterThan(abs(mVert), 1e-6);
        end
    end
end

function points = local_square_points()
    points = zeros(3, 4, 1);
    points(1, :, 1) = [-0.1, 0.1, 0.1, -0.1];
    points(2, :, 1) = [-0.05, -0.05, 0.05, 0.05];
    points(3, :, 1) = 0;
end

function m = local_contact_moment(com, points, forces)
    [mTip, ~] = local_contact_moment_parts(com, points, forces);
    m = mTip;
end

function [mTip, mVert] = local_contact_moment_parts(com, points, forces)
    p = size(points, 2);
    zGround = mean(points(3, :, 1));
    mVec = zeros(3, 1);
    for i = 1:p
        ri = points(:, i) - com;
        rGround = [points(1:2, i); zGround] - com;
        fi = forces(:, i);
        mVec = mVec + cross(rGround, [fi(1:2); 0]) + cross(ri, [0; 0; fi(3)]);
    end
    mTip = hypot(mVec(1), mVec(2));
    mVert = mVec(3);
end
