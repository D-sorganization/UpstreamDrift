classdef test_gs3dx_contact_solve < matlab.unittest.TestCase
%TEST_GS3DX_CONTACT_SOLVE  Verified capture contact wrench (#11173).
%
%   Pins gs3dx_contact_wrench and gs3dx_com_shift on the tour-average driver
%   capture with the foot-width contact hull.  Skipped when MATLAB's Python
%   has no ezc3d.

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            gs3dx_setup();
            try
                py.importlib.import_module('ezc3d');
                has_ezc3d = true;
            catch
                has_ezc3d = false;
            end
            testCase.assumeTrue(has_ezc3d, 'Python ezc3d is not available to MATLAB (pyenv)');
        end
    end

    methods (Test)
        function capture_wrench_matches_the_foot_width_residual(testCase)
            % Foot-width kinematic residuals from gs3dx_contact_wrench on the
            % capture; not model impact-support acceptance.
            [k, com, points] = local_capture_foot_width_contact_inputs();
            w = gs3dx_contact_wrench(k.grf, com, points);

            impact = k.impact_frame;
            pre = 1:impact;
            moment_rms = sqrt(mean(w.moment(pre) .^ 2));
            testCase.verifyEqual(moment_rms, 21.863, 'AbsTol', 0.02);
            testCase.verifyEqual(w.moment(impact), 81.427, 'AbsTol', 0.02);
            testCase.verifyLessThan(sqrt(mean(w.force(pre) .^ 2)), 1e-6);
        end

        function capture_com_shift_matches_the_foot_width_lever(testCase)
            % Horizontal COM shift from gs3dx_com_shift on the capture hull;
            % not applied to references or the model.
            [k, com, points] = local_capture_foot_width_contact_inputs();
            shift = gs3dx_com_shift(k.grf, com, points);
            h = hypot(shift(1, :), shift(2, :));

            impact = k.impact_frame;
            pre = 1:impact;
            testCase.verifyEqual(h(1), 0, 'AbsTol', 1e-3);
            testCase.verifyEqual(sqrt(mean(h(pre) .^ 2)), 0.0172505, 'AbsTol', 1e-3);
            testCase.verifyEqual(h(impact), 0.0732181, 'AbsTol', 1e-3);
            [hmax, idx] = max(h(pre));
            testCase.verifyEqual(hmax, 0.105126, 'AbsTol', 1e-3);
            t = k.t(:);
            testCase.verifyEqual(t(idx), 1.30278, 'AbsTol', 0.01);
        end
    end
end

function [k, com, points] = local_capture_foot_width_contact_inputs()
    k = gs3dx_kinematic_grf(trunk="joint_centres");
    cap = gs3dx_capture_markers();
    jc = gs3dx_capture_joint_centres(cap);
    tf = gs3dx_capture_address_transform(cap);
    com = k.com - tf.S.' * tf.origin;

    leg = gs3dx_leg_table();
    foot_length = leg.params.FootLength;
    foot_width = leg.stance.foot_width;
    sole_drop = leg.params.AnkleHeight;

    t_k = k.t(:).';
    ankle_L = local_resample_foot_track(t_k, jc.t, jc.ankle_L);
    ankle_R = local_resample_foot_track(t_k, jc.t, jc.ankle_R);
    toe_L = local_resample_foot_track(t_k, jc.t, jc.toe_L);
    toe_R = local_resample_foot_track(t_k, jc.t, jc.toe_R);

    ankle_L(3, :) = ankle_L(3, :) - sole_drop;
    ankle_R(3, :) = ankle_R(3, :) - sole_drop;
    toe_L(3, :) = toe_L(3, :) - sole_drop;
    toe_R(3, :) = toe_R(3, :) - sole_drop;

    pL = gs3dx_foot_contacts(ankle_L, toe_L, foot_length, foot_width);
    pR = gs3dx_foot_contacts(ankle_R, toe_R, foot_length, foot_width);
    points = cat(2, pL, pR);
end

function q = local_resample_foot_track(t_k, t_src, q_src)
    if numel(t_k) == size(q_src, 2)
        q = q_src;
    else
        t_src = t_src(:).';
        q = interp1(t_src, q_src.', t_k, 'linear', 'extrap').';
    end
end
