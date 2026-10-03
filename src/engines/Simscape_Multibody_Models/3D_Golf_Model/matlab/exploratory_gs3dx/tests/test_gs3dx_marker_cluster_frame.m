classdef test_gs3dx_marker_cluster_frame < matlab.unittest.TestCase
%TEST_GS3DX_MARKER_CLUSTER_FRAME  Unit tests for pure rigid 3-marker frame helper (#10979).
%   Proves independent known frames, rigid rotation/translation invariance,
%   positive uniform scale invariance, proper rotation matrix det/orthogonality,
%   fail-closed handling of gaps and nonfinite samples, and scale-relative
%   collinear and coincident relative guards without imputation or interpolation.

    methods (TestClassSetup)
        function setup(testCase)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function testIndependentKnownFrame(testCase)
            % Canonical standard basis: origin at [0;0;0], axis at +X, plane at +Y
            p0 = [0; 0; 0];
            p1 = [1; 0; 0];
            p2 = [0; 2; 0];
            [F, valid] = gs3dx_marker_cluster_frame(p0, p1, p2);
            testCase.verifyTrue(valid);
            testCase.verifyEqual(F, eye(3), 'AbsTol', 1e-12);

            % Canonical permuted frame: origin at [1; 2; 3], axis along +Z, plane along +X
            % Expect x = [0;0;1], y = [1;0;0], z = [0;1;0]
            p0 = [1; 2; 3];
            p1 = p0 + [0; 0; 4];
            p2 = p0 + [2; 0; 0];
            [F2, valid2] = gs3dx_marker_cluster_frame(p0, p1, p2);
            testCase.verifyTrue(valid2);
            testCase.verifyEqual(F2, [0 1 0; 0 0 1; 1 0 0], 'AbsTol', 1e-12);
        end

        function testRigidTransformAndTranslation(testCase)
            % Known arbitrary marker cluster
            p0 = [0.1; 0.2; 0.3];
            p1 = [0.4; 0.1; 0.5];
            p2 = [0.0; 0.5; 0.2];
            [F_ref, valid_ref] = gs3dx_marker_cluster_frame(p0, p1, p2);
            testCase.verifyTrue(valid_ref);

            % Pure translation leaves frame invariant
            t = [-3.2; 15.1; 9.4];
            [F_trans, valid_trans] = gs3dx_marker_cluster_frame(p0 + t, p1 + t, p2 + t);
            testCase.verifyTrue(valid_trans);
            testCase.verifyEqual(F_trans, F_ref, 'AbsTol', 1e-12);

            % Rigid rotation R in SO(3) and translation t2
            R = local_euler_rotation_matrix(30, -25, 45);
            t2 = [2.0; -1.5; 0.8];
            p0_rot = R * p0 + t2;
            p1_rot = R * p1 + t2;
            p2_rot = R * p2 + t2;
            [F_rot, valid_rot] = gs3dx_marker_cluster_frame(p0_rot, p1_rot, p2_rot);
            testCase.verifyTrue(valid_rot);
            testCase.verifyEqual(F_rot, R * F_ref, 'AbsTol', 1e-12);
        end

        function testPositiveScaleInvariance(testCase)
            % Uniform positive scaling must preserve frame and validity
            p0 = [0.05; -0.02; 0.10];
            p1 = [0.18; 0.07; 0.12];
            p2 = [0.03; 0.15; 0.04];
            [F_base, valid_base] = gs3dx_marker_cluster_frame(p0, p1, p2);
            testCase.verifyTrue(valid_base);

            scales = [1e-4, 0.001, 0.05, 1.0, 2.5, 50.0, 1e4];
            for s = scales
                [F_scaled, valid_scaled] = gs3dx_marker_cluster_frame(s * p0, s * p1, s * p2);
                testCase.verifyTrue(valid_scaled, sprintf('Validity failed at scale %g', s));
                testCase.verifyEqual(F_scaled, F_base, 'AbsTol', 1e-12, sprintf('Frame mismatch at scale %g', s));
            end
        end

        function testTinyAndLargeMetamorphicScaleInvariance(testCase)
            % Metamorphic test: frame must be scale-invariant for tiny and large
            % finite positive uniform scales (at least 1e-20, down to 1e-300, up to 1e300).
            % Floating-point limits: translation of tiny clusters loses relative
            % precision due to mantissa cancellation, so scale invariance is
            % evaluated without arbitrary large translations that erase coordinates.
            p0 = [0.1; 0.2; 0.3];
            p1 = [0.4; 0.1; 0.5];
            p2 = [0.0; 0.5; 0.2];
            [F_ref, valid_ref] = gs3dx_marker_cluster_frame(p0, p1, p2);
            testCase.verifyTrue(valid_ref);

            % Tiny scales (at least 1e-20, down to 1e-300) and large scales (up to 1e300)
            scales = [1e-20, 1e-50, 1e-100, 1e-250, 1e-300, 1e20, 1e100, 1e250, 1e300];
            for s = scales
                [F_scaled, valid_scaled] = gs3dx_marker_cluster_frame(s * p0, s * p1, s * p2);
                testCase.verifyTrue(valid_scaled, sprintf('Validity failed at scale %g', s));
                testCase.verifyEqual(F_scaled, F_ref, 'AbsTol', 1e-12, sprintf('Frame mismatch at scale %g', s));
            end

            % Extreme case: finite points whose direct difference overflows without max-abs normalization
            p0_ovf = [-1.0e308; 0; 0];
            p1_ovf = [1.0e308; 0; 0];
            p2_ovf = [0; 1.0e308; 0];
            [F_ovf, valid_ovf] = gs3dx_marker_cluster_frame(p0_ovf, p1_ovf, p2_ovf);
            testCase.verifyTrue(valid_ovf);
            testCase.verifyEqual(F_ovf, [1 0 0; 0 1 0; 0 0 1], 'AbsTol', 1e-12);
        end

        function testProperRotationDetAndOrthogonality(testCase)
            % Valid frames must strictly belong to SO(3): det(F)=1 and F'*F = I
            rng(10979);
            N = 25;
            p0 = randn(3, N);
            p1 = p0 + [1; 0.2; -0.1] + 0.05 * randn(3, N);
            p2 = p0 + [-0.1; 1; 0.2] + 0.05 * randn(3, N);

            [F, valid] = gs3dx_marker_cluster_frame(p0, p1, p2);
            testCase.verifyTrue(all(valid));
            testCase.verifySize(F, [3, 3, N]);
            for k = 1:N
                Rk = F(:, :, k);
                testCase.verifyEqual(det(Rk), 1.0, 'AbsTol', 1e-12);
                testCase.verifyEqual(Rk' * Rk, eye(3), 'AbsTol', 1e-12);
                testCase.verifyEqual(Rk * Rk', eye(3), 'AbsTol', 1e-12);
            end
        end

        function testOneGapBetweenValidSamples(testCase)
            % Missing or nonfinite samples must be valid=false and all-NaN without imputation
            p0 = repmat([0; 0; 0], 1, 5);
            p1 = repmat([1; 0; 0], 1, 5);
            p2 = repmat([0; 1; 0], 1, 5);
            p0(1, 2) = NaN;   % NaN in origin
            p1(2, 3) = Inf;   % Inf in axis_marker
            p2(3, 4) = -Inf;  % -Inf in plane_marker

            [F, valid] = gs3dx_marker_cluster_frame(p0, p1, p2);
            testCase.verifyEqual(valid, [true, false, false, false, true]);
            testCase.verifyEqual(F(:, :, 1), eye(3), 'AbsTol', 1e-12);
            testCase.verifyEqual(F(:, :, 5), eye(3), 'AbsTol', 1e-12);
            for g = 2:4
                testCase.verifyTrue(all(isnan(F(:, :, g)), 'all'), sprintf('Frame %d must be all-NaN', g));
            end
        end

        function testCoincidentAndCollinearRelativeGuard(testCase)
            % Degenerate geometries yield valid=false and all-NaN
            p0 = [1; 2; 3];
            p1 = p0 + [1; 0; 0];
            p2 = p0 + [0; 1; 0];

            % Coincident markers
            [F1, v1] = gs3dx_marker_cluster_frame(p0, p0, p2); % p1 == p0
            testCase.verifyFalse(v1); testCase.verifyTrue(all(isnan(F1), 'all'));

            [F2, v2] = gs3dx_marker_cluster_frame(p0, p1, p0); % p2 == p0
            testCase.verifyFalse(v2); testCase.verifyTrue(all(isnan(F2), 'all'));

            [F3, v3] = gs3dx_marker_cluster_frame(p0, p1, p1); % p2 == p1
            testCase.verifyFalse(v3); testCase.verifyTrue(all(isnan(F3), 'all'));

            % Collinear markers (parallel and anti-parallel)
            [F4, v4] = gs3dx_marker_cluster_frame(p0, p0 + [1; 0; 0], p0 + [2; 0; 0]);
            testCase.verifyFalse(v4); testCase.verifyTrue(all(isnan(F4), 'all'));

            [F5, v5] = gs3dx_marker_cluster_frame(p0, p0 + [1; 0; 0], p0 - [1.5; 0; 0]);
            testCase.verifyFalse(v5); testCase.verifyTrue(all(isnan(F5), 'all'));

            % Near-collinear marker (relative perpendicular offset < declared tol 1e-6)
            eps_rel = 1e-7;
            [F6, v6] = gs3dx_marker_cluster_frame(p0, p0 + [1; 0; 0], p0 + [1; eps_rel; 0]);
            testCase.verifyFalse(v6); testCase.verifyTrue(all(isnan(F6), 'all'));

            % Scale invariance of degeneracy guard
            for s = [1e-4, 1e4]
                [~, vs_coll] = gs3dx_marker_cluster_frame(s * p0, s * (p0 + [1; 0; 0]), s * (p0 + [2; 0; 0]));
                testCase.verifyFalse(vs_coll);
                [~, vs_near] = gs3dx_marker_cluster_frame(s * p0, s * (p0 + [1; 0; 0]), s * (p0 + [1; eps_rel; 0]));
                testCase.verifyFalse(vs_near);
            end
        end

        function testWrongDimensionsAndComplexDbC(testCase)
            p0 = [0; 0; 0]; p1 = [1; 0; 0]; p2 = [0; 1; 0];
            % Complex inputs rejected
            testCase.verifyError(@() gs3dx_marker_cluster_frame(p0 + 1i, p1, p2), ?MException);
            testCase.verifyError(@() gs3dx_marker_cluster_frame(p0, p1 + 1i, p2), ?MException);
            testCase.verifyError(@() gs3dx_marker_cluster_frame(p0, p1, p2 + 1i), ?MException);

            % Wrong dimensions rejected
            testCase.verifyError(@() gs3dx_marker_cluster_frame([0; 0], [1; 0], [0; 1]), ?MException);
            testCase.verifyError(@() gs3dx_marker_cluster_frame([0 0 0], [1 0 0], [0 1 0]), ?MException);
            testCase.verifyError(@() gs3dx_marker_cluster_frame(zeros(4, 1), zeros(4, 1), zeros(4, 1)), ?MException);
            testCase.verifyError(@() gs3dx_marker_cluster_frame(zeros(3, 2, 2), zeros(3, 2, 2), zeros(3, 2, 2)), ?MException);

            % Mismatched frame counts rejected
            testCase.verifyError(@() gs3dx_marker_cluster_frame(zeros(3, 2), zeros(3, 3), zeros(3, 2)), ?MException);
            testCase.verifyError(@() gs3dx_marker_cluster_frame(zeros(3, 2), zeros(3, 2), zeros(3, 1)), ?MException);

            % Empty input (N=0) rejected
            testCase.verifyError(@() gs3dx_marker_cluster_frame(zeros(3, 0), zeros(3, 0), zeros(3, 0)), ?MException);

            % Non-numeric rejected
            testCase.verifyError(@() gs3dx_marker_cluster_frame("invalid", p1, p2), ?MException);
        end
    end
end

function R = local_euler_rotation_matrix(yaw_deg, pitch_deg, roll_deg)
    y = deg2rad(yaw_deg); p = deg2rad(pitch_deg); r = deg2rad(roll_deg);
    Rz = [cos(y) -sin(y) 0; sin(y) cos(y) 0; 0 0 1];
    Ry = [cos(p) 0 sin(p); 0 1 0; -sin(p) 0 cos(p)];
    Rx = [1 0 0; 0 cos(r) -sin(r); 0 sin(r) cos(r)];
    R = Rz * Ry * Rx;
end
