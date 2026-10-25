classdef test_gs3dx_foot_split < matlab.unittest.TestCase
%TEST_GS3DX_FOOT_SPLIT  Closed-form foot GRF split (#11173 step 4).
%
%   Synthetic stance geometry only; no capture, model or Simscape.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root, fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function quiet_stance_equal_split(testCase)
            M = 80;
            g = 9.81;
            fz = M * g;
            f_required = [0; 0; fz];
            p_com = [0; 0; 1];
            feet = local_feet(-0.15, 0.15);
            s = gs3dx_foot_split(f_required, p_com, feet);
            testCase.verifyEqual(s.f_L(3), 0.5 * fz, 'AbsTol', 1e-9);
            testCase.verifyEqual(s.f_R(3), 0.5 * fz, 'AbsTol', 1e-9);
            testCase.verifyLessThan(norm(s.force), 1e-9);
            testCase.verifyLessThan(norm(s.moment), 1e-9);
            testCase.verifyEqual(s.support, 1, 'AbsTol', 1e-12);
            testCase.verifyEqual(s.fz_required, fz, 'AbsTol', 1e-12);
        end

        function com_over_left_foot(testCase)
            M = 80;
            g = 9.81;
            fz = M * g;
            f_required = [0; 0; fz];
            p_com = [-0.15; 0; 1];
            feet = local_feet(-0.15, 0.15);
            s = gs3dx_foot_split(f_required, p_com, feet);
            testCase.verifyEqual(s.f_L(3), fz, 'AbsTol', 1e-9);
            testCase.verifyEqual(s.f_R(3), 0, 'AbsTol', 1e-12);
            testCase.verifyLessThan(norm(s.moment), 1e-9);
        end

        function com_outside_left_residual_moment(testCase)
            M = 80;
            g = 9.81;
            fz = M * g;
            f_required = [0; 0; fz];
            p_com = [-0.25; 0; 1];
            feet = local_feet(-0.15, 0.15);
            s = gs3dx_foot_split(f_required, p_com, feet);
            testCase.verifyEqual(s.f_L(3), fz, 'AbsTol', 1e-9);
            testCase.verifyEqual(s.f_R(3), 0, 'AbsTol', 1e-12);
            testCase.verifyEqual(norm(s.moment), fz * 0.1, 'AbsTol', 1e-6);
        end

        function downward_requirement_no_pull(testCase)
            f_required = [0; 0; -100];
            p_com = [0; 0; 1];
            feet = local_feet(-0.15, 0.15);
            s = gs3dx_foot_split(f_required, p_com, feet);
            testCase.verifyEqual(s.f_L(3), 0, 'AbsTol', 1e-12);
            testCase.verifyEqual(s.f_R(3), 0, 'AbsTol', 1e-12);
            testCase.verifyEqual(s.force, f_required, 'AbsTol', 1e-9);
        end

        function horizontal_excess_goes_to_residual(testCase)
            M = 80;
            g = 9.81;
            fz = M * g;
            mu = 0.8;
            f_required = [800; 0; fz];
            p_com = [0; 0; 1];
            feet = local_feet(-0.15, 0.15);
            s = gs3dx_foot_split(f_required, p_com, feet, mu);
            testCase.verifyEqual(s.f_L(3), 0.5 * fz, 'AbsTol', 1e-9);
            testCase.verifyEqual(s.f_R(3), 0.5 * fz, 'AbsTol', 1e-9);
            cap = mu * 0.5 * fz;
            testCase.verifyEqual(norm(s.f_L(1:2)), cap, 'AbsTol', 1e-9);
            testCase.verifyEqual(norm(s.f_R(1:2)), cap, 'AbsTol', 1e-9);
            testCase.verifyEqual(s.force(1), 800 - 2 * cap, 'AbsTol', 1e-9);
            testCase.verifyEqual(s.force(2:3), [0; 0], 'AbsTol', 1e-9);
        end
    end
end

function feet = local_feet(xL, xR)
    feet.p_L = [xL; 0; 0];
    feet.p_R = [xR; 0; 0];
end
