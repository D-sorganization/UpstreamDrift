classdef test_gs3dx_club_acceptance < matlab.unittest.TestCase
%TEST_GS3DX_CLUB_ACCEPTANCE Contact orientation must participate in #11160.
    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(root, 'tools'));
        end
    end
    methods (Test)
        function face_error_at_contact_rejects_perfect_path(testCase)
            [ref, t, phases] = local_tracks();
            sim = ref;
            a = deg2rad(8.6);
            sim.face_normal = repmat([0; sin(a); cos(a)], 1, numel(t));
            m = gs3dx_club_match(ref, sim, t, phases);
            testCase.verifyFalse(m.ok.all);
        end
        function face_gate_uses_contact_not_peak_speed(testCase)
            [ref, t, phases] = local_tracks();
            sim = ref;
            phases.contact = 7;
            a = deg2rad(8.6);
            sim.face_normal(:, phases.contact) = [0; sin(a); cos(a)];
            m = gs3dx_club_match(ref, sim, t, phases);
            testCase.verifyFalse(m.ok.all);
        end
        function missing_contact_cannot_qualify(testCase)
            [ref, t, phases] = local_tracks();
            phases = rmfield(phases, 'contact');
            m = gs3dx_club_match(ref, ref, t, phases);
            testCase.verifyFalse(m.ok.all);
        end
        function perfect_measured_contact_passes(testCase)
            [ref, t, phases] = local_tracks();
            m = gs3dx_club_match(ref, ref, t, phases);
            testCase.verifyTrue(m.ok.all);
        end
        function degenerate_shaft_is_rejected(testCase)
            [ref, t, phases] = local_tracks();
            ref.head = ref.grip;
            testCase.verifyError(@() gs3dx_club_match(ref, ref, t, phases), 'gs3dx:club_match');
        end
        function degenerate_face_projection_is_rejected(testCase)
            [ref, t, phases] = local_tracks();
            ref.face_normal = repmat([1; 0; 0], 1, numel(t));
            testCase.verifyError(@() gs3dx_club_match(ref, ref, t, phases), 'gs3dx:club_match');
        end
        function fractional_phase_is_rejected(testCase)
            [ref, t, phases] = local_tracks();
            phases.top = 4.5;
            testCase.verifyError(@() gs3dx_club_match(ref, ref, t, phases), 'gs3dx:club_match');
        end
        function stationary_impact_has_no_relative_speed(testCase)
            [ref, t, phases] = local_tracks();
            ref.head(:,:) = repmat([1; 0; 0], 1, numel(t));
            ref.grip(:,:) = repmat([0.5; 0; 0], 1, numel(t));
            testCase.verifyError(@() gs3dx_club_match(ref, ref, t, phases), 'gs3dx:club_match');
        end
    end
end

function [ref, t, phases] = local_tracks()
    t = linspace(0, 1, 11);
    ref.grip = [t; zeros(2, numel(t))];
    ref.head = ref.grip + [0.5; 0; 0];
    ref.face_normal = repmat([0; 0; 1], 1, numel(t));
    phases = struct('address', 1, 'top', 5, 'impact', 11, 'contact', 10);
end
