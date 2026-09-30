classdef test_gs3dx_club_match < matlab.unittest.TestCase
%TEST_GS3DX_CLUB_MATCH  How closely a simulated club follows a captured one (#11160).
%
%   GS3DX_CLUB_MATCH compares two already-extracted club tracks (head, grip,
%   face normal) against the #11160 targets.  Synthetic tracks only; nothing
%   is captured or simulated.

    methods (TestClassSetup)
        function setup(~)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            gs3dx_setup();
        end
    end

    methods (Test)
        function identical_tracks_have_no_error(testCase)
            [ref, t, phases] = local_data();
            M = gs3dx_club_match(ref, ref, t, phases);
            testCase.verifyEqual(M.frame.head_err_m, zeros(numel(t), 1), 'AbsTol', 1e-12);
            testCase.verifyEqual(M.frame.grip_err_m, zeros(numel(t), 1), 'AbsTol', 1e-12);
            testCase.verifyEqual(M.frame.shaft_err_deg, zeros(numel(t), 1), 'AbsTol', 1e-9);
            testCase.verifyEqual(M.frame.face_err_deg, zeros(numel(t), 1), 'AbsTol', 1e-9);
            testCase.verifyTrue(M.ok.all);
        end

        function a_constant_head_offset_costs_its_own_size(testCase)
            [ref, t, phases] = local_data();
            sim = ref;
            sim.head = ref.head + 0.005 * [0; 0; 1];
            M = gs3dx_club_match(ref, sim, t, phases);
            a = M.phase(M.phase.phase == "address_to_impact", :);
            testCase.verifyEqual(a.head_rms_mm, 5, 'AbsTol', 1e-9);
            testCase.verifyEqual(a.head_max_mm, 5, 'AbsTol', 1e-9);
            testCase.verifyTrue(M.ok.all, 'a 5 mm offset is within every #11160 target');

            sim.head = ref.head + 0.025 * [0; 0; 1];
            M = gs3dx_club_match(ref, sim, t, phases);
            a = M.phase(M.phase.phase == "address_to_impact", :);
            testCase.verifyEqual(a.head_rms_mm, 25, 'AbsTol', 1e-9);
            testCase.verifyFalse(M.ok.all, 'a 25 mm offset exceeds head_rms_mm and head_max_mm');
        end

        function a_rotated_shaft_is_measured_in_degrees(testCase)
            [ref, t, phases] = local_data();
            sim = ref;
            dphi = deg2rad(3);
            len = vecnorm(ref.head - ref.grip);
            theta = atan2(ref.head(2, :) - ref.grip(2, :), ref.head(1, :) - ref.grip(1, :)) + dphi;
            sim.head = ref.grip + len .* [cos(theta); sin(theta); zeros(size(theta))];
            M = gs3dx_club_match(ref, sim, t, phases);
            testCase.verifyEqual(M.frame.shaft_err_deg, 3 * ones(numel(t), 1), 'AbsTol', 1e-9);
            testCase.verifyFalse(M.ok.all, 'a 3 deg shaft error exceeds the 2 deg target');
        end

        function a_face_rotated_about_the_shaft_is_measured_in_degrees(testCase)
            [ref, t, phases] = local_data();
            sim = ref;
            phi = deg2rad(8.6);
            u = (ref.head - ref.grip) ./ vecnorm(ref.head - ref.grip);
            sim.face_normal = ref.face_normal * cos(phi) + cross(u, ref.face_normal, 1) * sin(phi);
            M = gs3dx_club_match(ref, sim, t, phases);
            testCase.verifyEqual(M.frame.face_err_deg, 8.6 * ones(numel(t), 1), 'AbsTol', 1e-9);
        end

        function a_scaled_head_speed_is_a_percent_error_at_impact(testCase)
            [ref, t, phases] = local_data();
            sim = ref;
            sim.head = ref.head(:, phases.address) + 1.03 * (ref.head - ref.head(:, phases.address));
            M = gs3dx_club_match(ref, sim, t, phases);
            a = M.phase(M.phase.phase == "address_to_impact", :);
            testCase.verifyEqual(a.impact_speed_err_pct, 3, 'AbsTol', 1e-9);
            testCase.verifyFalse(M.ok.speed);
            testCase.verifyFalse(M.ok.all);
        end

        function a_size_mismatch_is_refused(testCase)
            [ref, t, phases] = local_data();
            sim = ref;
            sim.head = sim.head(:, 1:end - 1);
            testCase.verifyError(@() gs3dx_club_match(ref, sim, t, phases), 'gs3dx:club_match');
        end

        function a_non_unit_face_normal_is_refused(testCase)
            [ref, t, phases] = local_data();
            sim = ref;
            sim.face_normal(:, 1) = 2 * sim.face_normal(:, 1);
            testCase.verifyError(@() gs3dx_club_match(ref, sim, t, phases), 'gs3dx:club_match');
        end

        function unordered_phases_are_refused(testCase)
            [ref, t, phases] = local_data();
            phases.top = phases.address;
            testCase.verifyError(@() gs3dx_club_match(ref, ref, t, phases), 'gs3dx:club_match');
        end
    end
end

function [ref, t, phases] = local_data()
% A synthetic swing: head and grip sweep the same angle on concentric
% circles (constant 0.5 m shaft length) from address to impact, with a
% club face normal held out of the swing plane (always shaft-perpendicular).
    n = 11;
    theta = linspace(-pi / 4, pi / 2, n);
    ref.grip = 0.9 * [cos(theta); sin(theta); zeros(1, n)];
    ref.head = 1.4 * [cos(theta); sin(theta); zeros(1, n)];
    ref.face_normal = repmat([0; 0; 1], 1, n);
    t = linspace(0, 0.3, n);
    phases = struct('address', 1, 'top', 5, 'impact', n);
end
