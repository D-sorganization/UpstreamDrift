classdef test_gs3dx_capture < matlab.unittest.TestCase
%TEST_GS3DX_CAPTURE  The tour-average capture and the stance taken from it (#10985).
%
%   Pins what the data audit found in data/C3D_TA_Driver.c3d (identity,
%   rate, no force plates) and that GS3DX_LEG_TABLE's .stance is what
%   GS3DX_CAPTURE_STANCE measures, and the force-plate-free total ground
%   reaction force (GS3DX_KINEMATIC_GRF, #11011).  Skipped when MATLAB's
%   Python has no ezc3d.

    properties
        cap struct
    end

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
            testCase.cap = gs3dx_capture_stance();
        end
    end

    methods (Test)
        function capture_is_the_audited_file(testCase)
            c = testCase.cap;
            testCase.verifyEqual(c.sha256, 'cbcb4d84aca0a1f558073f0c2328c3f501c4726aaf699e382cf20c5861a0971d');
            testCase.verifyEqual(c.n_frames, 654);
            testCase.verifyEqual(c.rate_hz, 360);
        end

        function kinematic_grf_carries_body_weight_at_address(testCase)
            % Static equilibrium: a check of the segment table, the marker
            % proxies and the axes together (#11011).
            k = gs3dx_kinematic_grf();
            still = k.t <= 0.1;
            testCase.verifyEqual(k.address, 1, 'AbsTol', 0.02);
            testCase.verifyLessThan(max(vecnorm(k.grf_bw(1:2, still))), 0.05, 'horizontal force at address');
        end

        function kinematic_grf_peaks_in_the_late_downswing(testCase)
            % The vertical force peaks just before impact at every cutoff
            % in the tested band; its level depends on the cutoff.
            for fc = [6 10]
                k = gs3dx_kinematic_grf(cutoff_hz=fc);
                testCase.verifyGreaterThan(k.peak.vertical_bw, 1.15, sprintf('%d Hz', fc));
                testCase.verifyLessThan(k.peak.vertical_bw, 1.7, sprintf('%d Hz', fc));
                testCase.verifyGreaterThan(k.peak.time_to_impact, -0.15, sprintf('%d Hz', fc));
                testCase.verifyLessThan(k.peak.time_to_impact, 0, sprintf('%d Hz', fc));
            end
        end

        function joint_centre_trunk_moves_only_the_trunk(testCase)
            % The trunk proxy alone moves the peak (docs/SHAPE.md): the
            % address equilibrium holds with either, the peak rises.
            k = gs3dx_kinematic_grf();
            kj = gs3dx_kinematic_grf(trunk="joint_centres");
            testCase.verifyEqual(kj.mass, k.mass);
            testCase.verifyEqual(kj.address, 1, 'AbsTol', 0.02);
            testCase.verifyEqual(k.peak.vertical_bw, 1.27, 'AbsTol', 0.01);
            testCase.verifyEqual(kj.peak.vertical_bw, 1.85, 'AbsTol', 0.01);
            testCase.verifyError(@() gs3dx_kinematic_grf(trunk="sternum"), 'MATLAB:validators:mustBeMember');
        end

        function ball_contact_follows_peak_speed(testCase)
            % Peak head speed comes a frame before the ball; at ball contact
            % the head is back at its address position and loses speed.
            c = gs3dx_capture_markers();
            testCase.verifyEqual(c.impact_frame, 476);
            testCase.verifyEqual(c.ball_frame, 477);
            testCase.verifyGreaterThan(c.ball_time, c.impact_frame);
            testCase.verifyLessThan(c.ball_time, c.impact_frame + 3);
            y = c.target_frame(:, 2).' * (c.club_head - c.club_head(:, 1));
            testCase.verifyLessThan(abs(y(c.ball_frame)), 0.06, 'head along the target line at ball contact (m)');
            v = vecnorm(diff(c.club_head, 1, 2));
            testCase.verifyLessThan(v(c.ball_frame + 1), 0.9 * v(c.impact_frame), 'speed lost to the ball');
        end

        function capture_has_no_ground_reaction_data(testCase)
            testCase.verifyEqual(testCase.cap.force_plates_used, 0);
            testCase.verifyEqual(testCase.cap.n_analog, 0);
        end

        function leg_and_foot_markers_are_complete(testCase)
            m = testCase.cap.missing;
            for name = ["LKneeOut", "RKneeOut", "LAnkleOut", "RAnkleOut", "LToeIn", "LToeOut", "RToeIn", "RToeOut"]
                testCase.verifyEqual(m.(name), 0, name);
            end
        end

        function leg_table_stance_matches_capture(testCase)
            c = testCase.cap;
            st = gs3dx_leg_table().stance;
            for s = ["L", "R"]
                testCase.verifyEqual(st.("ankle_" + s), c.("ankle_" + s)(1:2), 'AbsTol', 1e-3, "ankle_" + s);
                testCase.verifyEqual(st.("foot_yaw_" + s), c.("foot_yaw_" + s), 'AbsTol', 0.1, "foot_yaw_" + s);
            end
            drop = -mean([c.ankle_L(3), c.ankle_R(3)]);
            testCase.verifyEqual(st.drop, drop, 'AbsTol', 1e-3);
        end
    end
end
