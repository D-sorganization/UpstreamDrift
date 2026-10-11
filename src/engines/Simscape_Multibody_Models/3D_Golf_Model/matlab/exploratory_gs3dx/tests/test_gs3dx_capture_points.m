classdef test_gs3dx_capture_points < matlab.unittest.TestCase
%TEST_GS3DX_CAPTURE_POINTS  Unit tests for gs3dx_capture_points pure helper (#10985, #11011).
%
%   Verifies:
%     1. Exact SI conversion with (x, -z, y) Z-up mapping for 'm', 'mm', 'cm'
%     2. Row 4 (homogeneous coordinate 1) is never treated as residual
%     3. Negative residuals (< 0) are masked to NaN
%     4. Nonfinite residuals (NaN, Inf) are masked to NaN
%     5. Nonfinite XYZ coordinates are masked to NaN across all 3 axes
%     6. Zero residual (0.0) is treated as valid (non-negative)
%     7. Defective samples are isolated per-marker/frame and never replaced with 0
%     8. Fail-closed contract rejection for unsupported units, bad shapes, and complex inputs

    methods (TestClassSetup)
        function setup(~)
            here = fileparts(mfilename('fullpath'));
            tools_dir = fullfile(fileparts(here), 'tools');
            addpath(tools_dir);
        end
    end

    methods (Test)
        function test_exact_conversion_meters(testCase)
            % Units 'm': scale = 1.0, coordinate transform (x, -z, y)
            % Points: 4 x 2 x 2
            pts = zeros(4, 2, 2);
            % Frame 1, Marker 1: [1.2; 3.4; 5.6; 1.0]
            pts(:, 1, 1) = [1.2; 3.4; 5.6; 1.0];
            % Frame 1, Marker 2: [-0.5; -2.1; 4.0; 1.0]
            pts(:, 2, 1) = [-0.5; -2.1; 4.0; 1.0];
            % Frame 2, Marker 1: [2.0; 4.0; 6.0; 1.0]
            pts(:, 1, 2) = [2.0; 4.0; 6.0; 1.0];
            % Frame 2, Marker 2: [0.1; 0.2; 0.3; 1.0]
            pts(:, 2, 2) = [0.1; 0.2; 0.3; 1.0];

            res = ones(1, 2, 2) * 0.05;

            [out, mask] = gs3dx_capture_points(pts, 'm', res);

            testCase.verifyEqual(size(out), [3, 2, 2]);
            testCase.verifyEqual(size(mask), [1, 2, 2]);
            testCase.verifyFalse(any(mask, 'all'));

            % Frame 1, Marker 1: x_si = x = 1.2, y_si = -z = -5.6, z_si = y = 3.4
            testCase.verifyEqual(out(:, 1, 1), [1.2; -5.6; 3.4], 'AbsTol', 1e-15);
            % Frame 1, Marker 2: x_si = -0.5, y_si = -4.0, z_si = -2.1
            testCase.verifyEqual(out(:, 2, 1), [-0.5; -4.0; -2.1], 'AbsTol', 1e-15);
            % Frame 2, Marker 1: x_si = 2.0, y_si = -6.0, z_si = 4.0
            testCase.verifyEqual(out(:, 1, 2), [2.0; -6.0; 4.0], 'AbsTol', 1e-15);
            % Frame 2, Marker 2: x_si = 0.1, y_si = -0.3, z_si = 0.2
            testCase.verifyEqual(out(:, 2, 2), [0.1; -0.3; 0.2], 'AbsTol', 1e-15);
        end

        function test_exact_conversion_millimeters(testCase)
            % Units 'mm': scale = 1e-3, coordinate transform (x, -z, y)
            pts = reshape([1500; 2500; 3500; 1.0], [4, 1, 1]);
            res = reshape(0.1, [1, 1, 1]);

            % Test with char vector 'mm'
            [out_char, mask_char] = gs3dx_capture_points(pts, 'mm', res);
            testCase.verifyEqual(out_char(:, 1, 1), [1.5; -3.5; 2.5], 'AbsTol', 1e-14);
            testCase.verifyFalse(mask_char(1, 1, 1));

            % Test with string scalar "mm"
            [out_str, mask_str] = gs3dx_capture_points(pts, "mm", res);
            testCase.verifyEqual(out_str(:, 1, 1), [1.5; -3.5; 2.5], 'AbsTol', 1e-14);
            testCase.verifyFalse(mask_str(1, 1, 1));
        end

        function test_exact_conversion_centimeters(testCase)
            % Units 'cm': scale = 1e-2, coordinate transform (x, -z, y)
            pts = reshape([150; 250; 350; 1.0], [4, 1, 1]);
            res = reshape(0.1, [1, 1, 1]);

            % Test with char vector 'cm'
            [out_char, mask_char] = gs3dx_capture_points(pts, 'cm', res);
            testCase.verifyEqual(out_char(:, 1, 1), [1.5; -3.5; 2.5], 'AbsTol', 1e-14);
            testCase.verifyFalse(mask_char(1, 1, 1));

            % Test with string scalar "cm"
            [out_str, mask_str] = gs3dx_capture_points(pts, "cm", res);
            testCase.verifyEqual(out_str(:, 1, 1), [1.5; -3.5; 2.5], 'AbsTol', 1e-14);
            testCase.verifyFalse(mask_str(1, 1, 1));
        end

        function test_row4_ignored_not_treated_as_residual(testCase)
            % Official ezc3d specification: row 4 is homogeneous 1, not residual.
            % Test that negative, zero, positive, or NaN values in row 4 do NOT
            % mask points to NaN when the actual residual is valid (> 0).
            res = reshape(0.05, [1, 1, 1]);
            test_row4_values = [-999.0, 0.0, 1.0, 100.0, NaN];

            for k = 1:numel(test_row4_values)
                pts = reshape([1.0; 2.0; 3.0; test_row4_values(k)], [4, 1, 1]);
                [out, mask] = gs3dx_capture_points(pts, 'm', res);
                testCase.verifyFalse(mask(1, 1, 1), sprintf('Failed with row4 = %g', test_row4_values(k)));
                testCase.verifyEqual(out(:, 1, 1), [1.0; -3.0; 2.0], 'AbsTol', 1e-15);
            end
        end

        function test_negative_residual_masks_to_nan(testCase)
            % Residual < 0 is invalid per ezc3d specification.
            % Coordinates are finite, but negative residual must force coordinates to NaN.
            pts = reshape([1.0; 2.0; 3.0; 1.0], [4, 1, 1]);
            res_neg = reshape(-1.0, [1, 1, 1]);

            [out, mask] = gs3dx_capture_points(pts, 'm', res_neg);

            testCase.verifyTrue(mask(1, 1, 1));
            testCase.verifyTrue(all(isnan(out(:, 1, 1))));
            % Ensure missing values are never replaced with zero
            testCase.verifyFalse(any(out(:, 1, 1) == 0));
        end

        function test_nonfinite_residual_masks_to_nan(testCase)
            % Residual containing NaN, +Inf, or -Inf must be masked to NaN.
            pts = reshape([1.0; 2.0; 3.0; 1.0], [4, 1, 1]);
            nonfinite_res = [NaN, Inf, -Inf];

            for k = 1:numel(nonfinite_res)
                res = reshape(nonfinite_res(k), [1, 1, 1]);
                [out, mask] = gs3dx_capture_points(pts, 'm', res);
                testCase.verifyTrue(mask(1, 1, 1), sprintf('Failed with res = %g', nonfinite_res(k)));
                testCase.verifyTrue(all(isnan(out(:, 1, 1))));
            end
        end

        function test_nonfinite_xyz_masks_all_coords_to_nan(testCase)
            % If any coordinate in XYZ is NaN or Inf, the entire marker sample
            % across all 3 axes must become NaN.
            res = reshape(0.05, [1, 1, 1]);

            % NaN in X
            pts_nan_x = reshape([NaN; 2.0; 3.0; 1.0], [4, 1, 1]);
            [out_x, mask_x] = gs3dx_capture_points(pts_nan_x, 'm', res);
            testCase.verifyTrue(mask_x(1, 1, 1));
            testCase.verifyTrue(all(isnan(out_x(:, 1, 1))));

            % NaN in Y
            pts_nan_y = reshape([1.0; NaN; 3.0; 1.0], [4, 1, 1]);
            [out_y, mask_y] = gs3dx_capture_points(pts_nan_y, 'm', res);
            testCase.verifyTrue(mask_y(1, 1, 1));
            testCase.verifyTrue(all(isnan(out_y(:, 1, 1))));

            % NaN in Z
            pts_nan_z = reshape([1.0; 2.0; NaN; 1.0], [4, 1, 1]);
            [out_z, mask_z] = gs3dx_capture_points(pts_nan_z, 'm', res);
            testCase.verifyTrue(mask_z(1, 1, 1));
            testCase.verifyTrue(all(isnan(out_z(:, 1, 1))));

            % +Inf in X
            pts_inf = reshape([Inf; 2.0; 3.0; 1.0], [4, 1, 1]);
            [out_inf, mask_inf] = gs3dx_capture_points(pts_inf, 'm', res);
            testCase.verifyTrue(mask_inf(1, 1, 1));
            testCase.verifyTrue(all(isnan(out_inf(:, 1, 1))));
        end

        function test_zero_residual_is_valid(testCase)
            % Residual of exactly 0.0 is non-negative and valid in ezc3d.
            pts = reshape([1.0; 2.0; 3.0; 1.0], [4, 1, 1]);
            res_zero = reshape(0.0, [1, 1, 1]);

            [out, mask] = gs3dx_capture_points(pts, 'm', res_zero);
            testCase.verifyFalse(mask(1, 1, 1));
            testCase.verifyEqual(out(:, 1, 1), [1.0; -3.0; 2.0], 'AbsTol', 1e-15);
        end

        function test_multi_marker_multi_frame_isolation(testCase)
            % 3 markers, 4 frames: verify that defects in one marker/frame do NOT
            % pollute other markers or frames.
            pts = repmat(reshape([1.0; 2.0; 3.0; 1.0], [4, 1, 1]), [1, 3, 4]);
            res = repmat(reshape(0.1, [1, 1, 1]), [1, 3, 4]);

            % Defect 1: Marker 2, Frame 2 has negative residual
            res(1, 2, 2) = -1.0;
            % Defect 2: Marker 1, Frame 3 has NaN in Y
            pts(2, 1, 3) = NaN;
            % Defect 3: Marker 3, Frame 4 has Inf residual
            res(1, 3, 4) = Inf;

            [out, mask] = gs3dx_capture_points(pts, 'm', res);

            % Exactly 3 samples should be masked
            testCase.verifyEqual(nnz(mask), 3);
            testCase.verifyTrue(mask(1, 2, 2));
            testCase.verifyTrue(mask(1, 1, 3));
            testCase.verifyTrue(mask(1, 3, 4));

            % Check that masked samples are all NaN
            testCase.verifyTrue(all(isnan(out(:, 2, 2))));
            testCase.verifyTrue(all(isnan(out(:, 1, 3))));
            testCase.verifyTrue(all(isnan(out(:, 3, 4))));

            % Check that unmasked samples have exact converted coordinates
            testCase.verifyEqual(out(:, 1, 1), [1.0; -3.0; 2.0], 'AbsTol', 1e-15);
            testCase.verifyEqual(out(:, 2, 1), [1.0; -3.0; 2.0], 'AbsTol', 1e-15);
            testCase.verifyEqual(out(:, 3, 1), [1.0; -3.0; 2.0], 'AbsTol', 1e-15);
            testCase.verifyEqual(out(:, 1, 2), [1.0; -3.0; 2.0], 'AbsTol', 1e-15);
            testCase.verifyEqual(out(:, 3, 2), [1.0; -3.0; 2.0], 'AbsTol', 1e-15);
            testCase.verifyEqual(out(:, 2, 3), [1.0; -3.0; 2.0], 'AbsTol', 1e-15);
            testCase.verifyEqual(out(:, 3, 3), [1.0; -3.0; 2.0], 'AbsTol', 1e-15);
            testCase.verifyEqual(out(:, 1, 4), [1.0; -3.0; 2.0], 'AbsTol', 1e-15);
            testCase.verifyEqual(out(:, 2, 4), [1.0; -3.0; 2.0], 'AbsTol', 1e-15);
        end

        function test_rejection_bad_units(testCase)
            % Reject unsupported unit strings and invalid unit types fail-closed.
            pts = reshape([1.0; 2.0; 3.0; 1.0], [4, 1, 1]);
            res = reshape(0.1, [1, 1, 1]);

            testCase.verifyError(@() gs3dx_capture_points(pts, 'in', res), 'gs3dx:capture_points:unsupportedUnits');
            testCase.verifyError(@() gs3dx_capture_points(pts, 'inches', res), 'gs3dx:capture_points:unsupportedUnits');
            testCase.verifyError(@() gs3dx_capture_points(pts, 'ft', res), 'gs3dx:capture_points:unsupportedUnits');
            testCase.verifyError(@() gs3dx_capture_points(pts, 'km', res), 'gs3dx:capture_points:unsupportedUnits');
            testCase.verifyError(@() gs3dx_capture_points(pts, '', res), 'gs3dx:capture_points:unsupportedUnits');
            testCase.verifyError(@() gs3dx_capture_points(pts, 123, res), 'gs3dx:capture_points:invalidUnits');
            testCase.verifyError(@() gs3dx_capture_points(pts, {'m'}, res), 'gs3dx:capture_points:invalidUnits');
            testCase.verifyError(@() gs3dx_capture_points(pts, ["m", "mm"], res), 'gs3dx:capture_points:invalidUnits');
        end

        function test_rejection_bad_shapes(testCase)
            % Reject inputs with invalid dimensions or mismatched sizes.
            pts_valid = repmat([1.0; 2.0; 3.0; 1.0], [1, 2, 3]);
            res_valid = ones(1, 2, 3);

            % Points row count != 4
            testCase.verifyError(@() gs3dx_capture_points(zeros(3, 2, 3), 'm', res_valid), 'gs3dx:capture_points:badPoints');
            testCase.verifyError(@() gs3dx_capture_points(zeros(5, 2, 3), 'm', res_valid), 'gs3dx:capture_points:badPoints');

            % Residuals row count != 1
            testCase.verifyError(@() gs3dx_capture_points(pts_valid, 'm', zeros(2, 2, 3)), 'gs3dx:capture_points:badResiduals');

            % Marker count mismatch (dim 2)
            testCase.verifyError(@() gs3dx_capture_points(pts_valid, 'm', ones(1, 3, 3)), 'gs3dx:capture_points:dimensionMismatch');

            % Frame count mismatch (dim 3)
            testCase.verifyError(@() gs3dx_capture_points(pts_valid, 'm', ones(1, 2, 4)), 'gs3dx:capture_points:dimensionMismatch');

            % 4D inputs
            testCase.verifyError(@() gs3dx_capture_points(zeros(4, 2, 3, 2), 'm', ones(1, 2, 3, 2)), 'gs3dx:capture_points:badPoints');
        end

        function test_rejection_non_real_or_non_numeric(testCase)
            % Reject complex and non-numeric inputs fail-closed.
            res_valid = reshape(0.1, [1, 1, 1]);

            % Complex points
            pts_complex = reshape([1.0 + 1.0i; 2.0; 3.0; 1.0], [4, 1, 1]);
            testCase.verifyError(@() gs3dx_capture_points(pts_complex, 'm', res_valid), 'gs3dx:capture_points:badPoints');

            % Complex residuals
            pts_valid = reshape([1.0; 2.0; 3.0; 1.0], [4, 1, 1]);
            res_complex = reshape(0.1 + 0.1i, [1, 1, 1]);
            testCase.verifyError(@() gs3dx_capture_points(pts_valid, 'm', res_complex), 'gs3dx:capture_points:badResiduals');

            % Cell array points
            testCase.verifyError(@() gs3dx_capture_points({1, 2, 3, 4}, 'm', res_valid), 'gs3dx:capture_points:badPoints');
        end

        function test_single_frame_operation(testCase)
            % Verify that single-frame static captures (T = 1) work properly.
            pts_single = [1.0; 2.0; 3.0; 1.0];
            res_single = 0.05;

            [out, mask] = gs3dx_capture_points(pts_single, 'm', res_single);
            testCase.verifyEqual(size(out), [3, 1]);
            testCase.verifyEqual(size(mask), [1, 1]);
            testCase.verifyFalse(mask(1, 1));
            testCase.verifyEqual(out, [1.0; -3.0; 2.0], 'AbsTol', 1e-15);
        end
    end
end
