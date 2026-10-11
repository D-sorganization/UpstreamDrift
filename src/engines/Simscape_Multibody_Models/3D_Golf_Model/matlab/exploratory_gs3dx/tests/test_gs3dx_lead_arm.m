classdef test_gs3dx_lead_arm < matlab.unittest.TestCase
%TEST_GS3DX_LEAD_ARM  The lead arm stays near straight through the swing (#11156).
%
%   GS3DX_GOLF_ROM narrows the lead elbow to the golf band; the whole-body
%   IK holds it there by the same continuation penalty as the anatomical
%   range (test_gs3dx_joint_rom) without losing the marker fit.

    properties
        info struct
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
        end
    end

    methods (Test)
        function the_golf_band_narrows_only_the_lead_elbow(testCase)
            rom = gs3dx_joint_rom();
            golf = gs3dx_golf_rom(rom);
            le = rom.joint == "LE";
            testCase.verifyEqual(golf.max_deg(le), 20);
            testCase.verifyEqual(golf.min_deg(le), rom.min_deg(le));
            testCase.verifyEqual(golf.span_deg(le), 25);
            testCase.verifySubstring(char(golf.source(le)), '#11156');
            same = golf(~le, :);
            testCase.verifyEqual(same, rom(~le, :), 'Every other row is the anatomical one');
        end

        function the_ik_keeps_the_lead_arm_straight(testCase)
            % Every 10th frame from address to impact (as the anatomical
            % penalty test): with the golf band at weight 3 the lead elbow
            % flexes at most 20 deg (1 deg slack for the soft penalty), every
            % other bounded joint stays in range, and the fit costs at most
            % 15 mm RMS over the unpenalized chain.
            fit = char(gs3dx_names().variants.fit);
            testCase.assumeTrue(isfile(fullfile(testCase.info.models_dir, [fit '.slx'])), 'GS3DX_Fit is not built');
            jc = gs3dx_capture_joint_centres(gs3dx_capture_markers());
            f0 = jc.impact_frame;
            frames = unique([1:10:f0 f0]);
            reg = {'posture_weight', 0.01, 'smooth_weight', 0.02, 'backward', false, 'gap_weight', 0.5};
            % Explicitly pin historically Fit-based lead arm test calls to variants.fit
            off = gs3dx_whole_body_ik(jc, 'model', fit, frames=1, calibration_frames=1:15:f0 - 90).offsets;
            golf = gs3dx_golf_rom();
            free = gs3dx_whole_body_ik(jc, 'model', fit, 'frames', frames, 'offsets', off, reg{:});
            held = gs3dx_whole_body_ik(jc, 'model', fit, 'frames', frames, 'offsets', off, reg{:}, ...
                'rom_weight', 3, 'rom', golf);
            r0 = gs3dx_rom_check(golf, gs3dx_rom_from_ik(free, golf));
            r = gs3dx_rom_check(golf, gs3dx_rom_from_ik(held, golf));
            le = golf.joint == "LE";
            testCase.log(1, sprintf('Lead elbow flexion, deg: free %.1f to %.1f, held %.1f to %.1f', ...
                r0.min_deg(le), r0.max_deg(le), r.min_deg(le), r.max_deg(le)));
            testCase.verifyLessThanOrEqual(r.max_deg(le), 21, 'Lead elbow flexion beyond the golf band');
            bounded = ~isnan(r.min_deg);
            cols = {'joint', 'motion', 'min_deg', 'max_deg', 'excess_deg'};
            testCase.verifyLessThanOrEqual(r.excess_deg(bounded), 1, formattedDisplayText(r(bounded & r.excess_deg > 1, cols)));
            testCase.verifyLessThanOrEqual(mean(held.rms), mean(free.rms) + 0.015, ...
                sprintf('RMS %.1f mm with the band, %.1f mm without', 1000 * mean(held.rms), 1000 * mean(free.rms)));
        end
    end
end
