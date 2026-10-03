classdef test_gs3dx_fit_sphere < matlab.unittest.TestCase
%TEST_GS3DX_FIT_SPHERE  Unit tests for pure functional sphere fitter gs3dx_fit_sphere.
%
%   Verifies exact analytic sphere reconstruction, rigid transform and large
%   translation invariance, noise robustness, minimal 4-point tetrahedron solve,
%   and fail-closed behavior on coplanar, colinear, repeated, deficient, or non-finite inputs.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function test_exact_analytic_sphere(testCase)
            % Points on known sphere must be reconstructed to machine precision.
            c_true = [0.15; -0.22; 0.85];
            r_true = 0.045;
            % 50 points distributed on the sphere via spherical coordinates
            n = 50;
            phi = linspace(0, 2*pi, n);
            theta = linspace(-pi/2.5, pi/2.5, n);
            [P_grid, T_grid] = meshgrid(phi, theta);
            P = [c_true(1) + r_true * cos(T_grid(:)') .* cos(P_grid(:)'); ...
                 c_true(2) + r_true * cos(T_grid(:)') .* sin(P_grid(:)'); ...
                 c_true(3) + r_true * sin(T_grid(:)')];

            [c_fit, r_fit, rms_fit, diag_fit] = gs3dx_fit_sphere(P);

            testCase.verifyEqual(c_fit, c_true, 'AbsTol', 1e-12, 'Center matches analytic center');
            testCase.verifyEqual(r_fit, r_true, 'AbsTol', 1e-12, 'Radius matches analytic radius');
            testCase.verifyLessThan(rms_fit, 1e-12, 'RMS error is near machine zero');
            testCase.verifyEqual(diag_fit.rank, 4, 'Rank is full (4)');
            testCase.verifyLessThan(diag_fit.condition, 20, 'Condition number is small');
            testCase.verifyEqual(diag_fit.n_points, size(P, 2));
        end

        function test_rigid_transform_invariance(testCase)
            % Rigid rotation and translation of points must transform center identically.
            c_true = [0.05; 0.10; -0.15];
            r_true = 0.035;
            [X, Y, Z] = sphere(12);
            P0 = c_true + r_true * [X(:)'; Y(:)'; Z(:)'];

            % Rodrigues rotation: 45 deg about [1; 2; 3]
            axis_rot = [1; 2; 3] / norm([1; 2; 3]);
            ang = pi / 4;
            K = [0 -axis_rot(3) axis_rot(2); axis_rot(3) 0 -axis_rot(1); -axis_rot(2) axis_rot(1) 0];
            R = eye(3) + sin(ang) * K + (1 - cos(ang)) * (K * K);
            t = [-5.0; 12.0; 3.5];

            P_rot = R * P0 + t;
            c_expected = R * c_true + t;

            [c_fit, r_fit, rms_fit] = gs3dx_fit_sphere(P_rot);

            testCase.verifyEqual(c_fit, c_expected, 'AbsTol', 1e-11, 'Center transforms rigidly');
            testCase.verifyEqual(r_fit, r_true, 'AbsTol', 1e-11, 'Radius is rotation/translation invariant');
            testCase.verifyLessThan(rms_fit, 1e-11, 'RMS error is near machine zero');
        end

        function test_translation_invariance_huge_offset(testCase)
            % Centering prevents catastrophic cancellation when points are far from origin.
            c_true = [1000.0; -2000.0; 5000.0];
            r_true = 0.05;
            [X, Y, Z] = sphere(15);
            P = c_true + r_true * [X(:)'; Y(:)'; Z(:)'];

            [c_fit, r_fit, rms_fit] = gs3dx_fit_sphere(P);

            testCase.verifyEqual(c_fit, c_true, 'AbsTol', 1e-10, 'Center recovered with large offset');
            testCase.verifyEqual(r_fit, r_true, 'AbsTol', 1e-10, 'Radius recovered with large offset');
            testCase.verifyLessThan(rms_fit, 1e-10, 'RMS error is near zero despite huge coordinates');
        end

        function test_noisy_sphere(testCase)
            % Sphere with 0.5 mm noise should recover center and radius accurately.
            rng(42);
            c_true = [0.1; 0.2; 0.3];
            r_true = 0.050; % 50 mm
            [X, Y, Z] = sphere(16);
            dirs = [X(:)'; Y(:)'; Z(:)'];
            dirs = dirs ./ sqrt(sum(dirs .^ 2, 1));
            noise = 0.0005 * randn(1, size(dirs, 2)); % 0.5 mm noise on radii
            P = c_true + (r_true + noise) .* dirs;

            [c_fit, r_fit, rms_fit] = gs3dx_fit_sphere(P);

            testCase.verifyLessThan(norm(c_fit - c_true), 0.002, 'Center within 2 mm');
            testCase.verifyLessThan(abs(r_fit - r_true), 0.002, 'Radius within 2 mm');
            testCase.verifyGreaterThan(rms_fit, 0.0002, 'RMS reflects noise floor');
            testCase.verifyLessThan(rms_fit, 0.0010, 'RMS bounded by noise scale');
        end

        function test_minimal_four_points(testCase)
            % Minimal case: 4 vertices of a regular tetrahedron inscribed in a sphere.
            c_true = [0.5; -0.3; 1.2];
            r_true = 0.040;
            % Vertices of regular tetrahedron
            V = [1  1  1; ...
                 1 -1 -1; ...
                -1  1 -1; ...
                -1 -1  1].';
            V = V ./ sqrt(3); % on unit sphere
            P = c_true + r_true * V;

            [c_fit, r_fit, rms_fit, diag_fit] = gs3dx_fit_sphere(P);

            testCase.verifyEqual(c_fit, c_true, 'AbsTol', 1e-12, 'Exact center for 4 points');
            testCase.verifyEqual(r_fit, r_true, 'AbsTol', 1e-12, 'Exact radius for 4 points');
            testCase.verifyLessThan(rms_fit, 1e-12, 'Zero RMS for 4 points');
            testCase.verifyEqual(diag_fit.n_points, 4);
            testCase.verifyEqual(diag_fit.rank, 4);
        end

        function test_failclosed_fewer_than_four_points(testCase)
            % Must fail closed when N < 4.
            P3 = rand(3, 3);
            testCase.verifyError(@() gs3dx_fit_sphere(P3), 'gs3dx:fit_sphere:too_few_points');

            P2 = rand(3, 2);
            testCase.verifyError(@() gs3dx_fit_sphere(P2), 'gs3dx:fit_sphere:too_few_points');

            P1 = rand(3, 1);
            testCase.verifyError(@() gs3dx_fit_sphere(P1), 'gs3dx:fit_sphere:too_few_points');

            P0 = zeros(3, 0);
            testCase.verifyError(@() gs3dx_fit_sphere(P0), 'gs3dx:fit_sphere:too_few_points');
        end

        function test_failclosed_nan_inputs(testCase)
            % Must fail closed on NaN inputs.
            [X, Y, Z] = sphere(4);
            P = [X(:)'; Y(:)'; Z(:)'];
            P(1, 3) = NaN;
            testCase.verifyError(@() gs3dx_fit_sphere(P), 'gs3dx:fit_sphere:non_finite');
        end

        function test_failclosed_inf_inputs(testCase)
            % Must fail closed on Inf inputs.
            [X, Y, Z] = sphere(4);
            P = [X(:)'; Y(:)'; Z(:)'];
            P(2, 5) = Inf;
            testCase.verifyError(@() gs3dx_fit_sphere(P), 'gs3dx:fit_sphere:non_finite');

            P(2, 5) = -Inf;
            testCase.verifyError(@() gs3dx_fit_sphere(P), 'gs3dx:fit_sphere:non_finite');
        end

        function test_failclosed_complex_inputs(testCase)
            % Must fail closed on complex inputs.
            [X, Y, Z] = sphere(4);
            P = [X(:)'; Y(:)'; Z(:)'] + 1i * 0.01;
            testCase.verifyError(@() gs3dx_fit_sphere(P), 'gs3dx:fit_sphere:invalid_type');
        end

        function test_failclosed_wrong_dimensions(testCase)
            % Must fail closed when dimensions are not 3xN.
            testCase.verifyError(@() gs3dx_fit_sphere(rand(2, 10)), 'MATLAB:validation:IncompatibleSize');
            testCase.verifyError(@() gs3dx_fit_sphere(rand(4, 10)), 'MATLAB:validation:IncompatibleSize');
            testCase.verifyError(@() gs3dx_fit_sphere(rand(1, 10)), 'MATLAB:validation:IncompatibleSize');
        end

        function test_failclosed_identical_points(testCase)
            % Repeated identical points are degenerate and must fail closed.
            P = repmat([0.5; 0.2; -0.1], 1, 10);
            testCase.verifyError(@() gs3dx_fit_sphere(P), 'gs3dx:fit_sphere:degenerate_points');
        end

        function test_failclosed_colinear_points(testCase)
            % Points along a 3D line must fail closed (rank deficient).
            t = linspace(0, 1, 20);
            P = [0.1; 0.2; 0.3] + [1; -1; 2] * t;
            testCase.verifyError(@() gs3dx_fit_sphere(P), 'gs3dx:fit_sphere:rank_deficient');
        end

        function test_failclosed_coplanar_circle(testCase)
            % Points on a circle lying strictly in a plane (z = 0) must fail closed.
            theta = linspace(0, 2*pi, 20);
            P = [0.05 * cos(theta); 0.05 * sin(theta); zeros(1, 20)];
            testCase.verifyError(@() gs3dx_fit_sphere(P), 'gs3dx:fit_sphere:rank_deficient');
        end

        function test_failclosed_coplanar_arc(testCase)
            % Points on an arc lying in an arbitrary 3D plane must fail closed.
            theta = linspace(0, pi/2, 15);
            P_2d = [0.05 * cos(theta); 0.05 * sin(theta); zeros(1, 15)];
            % Rotate into arbitrary plane
            R = [0.36 0.48 -0.80; -0.80 0.60 0; 0.48 0.64 0.60];
            P_3d = R * P_2d + [0.2; -0.4; 0.1];
            testCase.verifyError(@() gs3dx_fit_sphere(P_3d), 'gs3dx:fit_sphere:rank_deficient');
        end

        function test_diagnostic_fields_contract(testCase)
            % Verify all fields of the diagnostic struct are populated and valid.
            c_true = [0.1; 0.2; 0.3];
            r_true = 0.05;
            [X, Y, Z] = sphere(8);
            P = c_true + r_true * [X(:)'; Y(:)'; Z(:)'];

            [~, ~, rms_val, diag_val] = gs3dx_fit_sphere(P);

            testCase.verifyTrue(isstruct(diag_val));
            testCase.verifyTrue(isfield(diag_val, 'rank'));
            testCase.verifyTrue(isfield(diag_val, 'condition'));
            testCase.verifyTrue(isfield(diag_val, 'singular_values'));
            testCase.verifyTrue(isfield(diag_val, 'rank_tolerance'));
            testCase.verifyTrue(isfield(diag_val, 'scale'));
            testCase.verifyTrue(isfield(diag_val, 'center_mean'));
            testCase.verifyTrue(isfield(diag_val, 'n_points'));
            testCase.verifyTrue(isfield(diag_val, 'rms'));
            testCase.verifyTrue(isfield(diag_val, 'geometric_residual'));
            testCase.verifyTrue(isfield(diag_val, 'algebraic_residual'));

            testCase.verifyEqual(diag_val.rank, 4);
            testCase.verifyGreaterThan(diag_val.condition, 0);
            testCase.verifyEqual(numel(diag_val.singular_values), 4);
            testCase.verifyEqual(diag_val.n_points, size(P, 2));
            testCase.verifyEqual(diag_val.rms, rms_val);
            testCase.verifySize(diag_val.geometric_residual, [1 size(P, 2)]);
            testCase.verifySize(diag_val.algebraic_residual, [size(P, 2) 1]);
        end
    end
end
