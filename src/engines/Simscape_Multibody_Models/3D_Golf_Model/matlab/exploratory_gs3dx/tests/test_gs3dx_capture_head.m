classdef test_gs3dx_capture_head < matlab.unittest.TestCase
%TEST_GS3DX_CAPTURE_HEAD  Head orientation and position from capture markers (#10979).
%
%   Verifies the head frame constructed from HeadTop, HeadFront, and HeadSide
%   markers in the tour-average capture (GS3DX_CAPTURE_HEAD_FRAME):
%   - Every rotation matrix is right-handed orthonormal (det = +1 to 1e-12).
%   - Address frame matches canonical anatomical sign conventions.
%   - Relative rotation from address starts at identity (to 1e-12).
%   - Rotation rate is physically bounded across measured frames to impact.
%   - Output structure fields, dimensions, and Design-by-Contract validation.

    properties
        cap struct
        hf struct
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
            testCase.cap = gs3dx_capture_markers();
            testCase.hf = gs3dx_capture_head_frame(testCase.cap);
        end
    end

    methods (Test)
        function output_structure_has_expected_fields_and_sizes(testCase)
            % Output fields must conform to specification with consistent sizing.
            hf = testCase.hf;
            n = testCase.cap.n_frames;

            testCase.verifyTrue(isfield(hf, 'centre'), 'hf must have .centre');
            testCase.verifyTrue(isfield(hf, 'R'), 'hf must have .R');
            testCase.verifyTrue(isfield(hf, 'R_rel'), 'hf must have .R_rel');
            testCase.verifyTrue(isfield(hf, 'gap'), 'hf must have .gap');
            testCase.verifyTrue(isfield(hf, 't'), 'hf must have .t');
            testCase.verifyTrue(isfield(hf, 'impact_frame'), 'hf must have .impact_frame');

            testCase.verifyEqual(size(hf.centre), [3, n], 'hf.centre must be 3 x n');
            testCase.verifyEqual(size(hf.R), [3, 3, n], 'hf.R must be 3 x 3 x n');
            testCase.verifyEqual(size(hf.R_rel), [3, 3, n], 'hf.R_rel must be 3 x 3 x n');
            testCase.verifyEqual(size(hf.gap), [1, n], 'hf.gap must be 1 x n');
            testCase.verifyTrue(islogical(hf.gap), 'hf.gap must be logical');
            testCase.verifyEqual(size(hf.t), [1, n], 'hf.t must be 1 x n');
            testCase.verifyEqual(hf.impact_frame, testCase.cap.impact_frame, 'hf.impact_frame must match cap');
        end

        function every_rotation_matrix_is_orthonormal_and_proper(testCase)
            % Every frame's R(:,:,k) must satisfy R' * R = I and det(R) = +1 to 1e-12.
            hf = testCase.hf;
            n = testCase.cap.n_frames;

            % Vectorized orthogonality check: R.' * R == I_3
            RtR = pagemtimes(permute(hf.R, [2 1 3]), hf.R);
            expected_I = repmat(eye(3), [1, 1, n]);
            testCase.verifyEqual(RtR, expected_I, 'AbsTol', 1e-12, ...
                'Every R(:,:,k) must be orthogonal to 1e-12');

            % Proper rotation check: det(R) == +1
            dets = zeros(1, n);
            for k = 1:n
                dets(k) = det(hf.R(:, :, k));
            end
            testCase.verifyEqual(dets, ones(1, n), 'AbsTol', 1e-12, ...
                'Every R(:,:,k) must have determinant +1 to 1e-12');
        end

        function address_orientation_sign_conventions(testCase)
            % At address (frame 1), forward column 1 must have positive component
            % along +X, up column 3 positive component along +Z, and left column 2
            % positive component along +Y.
            R1 = testCase.hf.R(:, :, 1);

            testCase.verifyGreaterThan(R1(1, 1), 0, ...
                'R(:,1,1) (forward) must have positive component along +X');
            testCase.verifyGreaterThan(R1(3, 3), 0, ...
                'R(:,3,1) (up) must have positive component along +Z');
            testCase.verifyGreaterThan(R1(2, 2), 0, ...
                'R(:,2,1) (left) must have positive component along +Y');
        end

        function relative_rotation_at_address_is_identity(testCase)
            % R_rel(:,:,1) must be identity to 1e-12, and R_rel(:,:,t) == R(:,:,t) * R(:,:,1)'.
            hf = testCase.hf;
            n = testCase.cap.n_frames;

            testCase.verifyEqual(hf.R_rel(:, :, 1), eye(3), 'AbsTol', 1e-12, ...
                'R_rel at address (frame 1) must be identity to 1e-12');

            % Verify definition R_rel(:,:,t) = R(:,:,t) * R(:,:,1)' across all frames
            expected_R_rel = pagemtimes(hf.R, permute(hf.R(:, :, 1), [2 1 3]));
            testCase.verifyEqual(hf.R_rel, expected_R_rel, 'AbsTol', 1e-12, ...
                'R_rel(:,:,t) must equal R(:,:,t) * R(:,:,1)'' across all frames');
        end

        function head_rotates_smoothly_up_to_impact(testCase)
            % The largest frame-to-frame rotation angle of R_rel over measured
            % (non-gap) frames up to impact must remain below a physically
            % motivated bound.
            %
            % Physical justification:
            % The capture sampling rate is 360 Hz (dt ~ 2.78 ms). During a golf
            % swing, torso rotational velocity peaks around 700-900 deg/s. The head
            % is stabilized by cervical reflex and gaze fixation on the ball, so
            % head angular velocity rarely exceeds 300-500 deg/s during the backswing
            % and downswing up to impact.
            % An upper physical limit of 720 deg/s corresponds to 2.0 degrees per
            % frame at 360 Hz (2.0 deg * 360 s^-1 = 720 deg/s). Any frame-to-frame
            % jump exceeding 2.0 degrees would indicate marker dropout, mislabeling,
            % or numerical discontinuity.
            hf = testCase.hf;
            imp = hf.impact_frame;
            d_ang = zeros(1, imp - 1);
            for t = 1:imp - 1
                if hf.gap(t) || hf.gap(t + 1)
                    continue; % Skip gap-filled intervals
                end
                R_step = hf.R(:, :, t + 1) * hf.R(:, :, t)';
                tr = trace(R_step);
                d_ang(t) = acosd(max(-1, min(1, (tr - 1) / 2)));
            end

            max_d_ang = max(d_ang);
            testCase.verifyLessThan(max_d_ang, 2.0, ...
                'Frame-to-frame head rotation angle must be below 2.0 deg (720 deg/s at 360 Hz)');
            testCase.verifyGreaterThan(max_d_ang, 0.1, ...
                'Head must show measurable rotational motion');
        end

        function head_markers_are_complete_in_driver_capture(testCase)
            % HeadTop, HeadFront, and HeadSide have 0 missing frames in the driver capture.
            testCase.verifyFalse(any(testCase.hf.gap), ...
                'All head markers in the driver capture are measured (no gaps)');
        end

        function input_validation_and_dbc_guards(testCase)
            % Argument validation and Design-by-Contract error reporting.
            testCase.verifyError(@() gs3dx_capture_head_frame(123), ...
                ?MException, 'Non-struct input must fail argument validation');

            % Missing head markers must throw an error with id gs3dx:head
            bad_cap = testCase.cap;
            bad_cap.labels = ["WaistLeft", "WaistRight", "WaistLBack", "WaistRBack"];
            testCase.verifyError(@() gs3dx_capture_head_frame(bad_cap), 'gs3dx:head');
        end
    end
end
