function tests = test_gs3dx_capture_observation_mask
%TEST_GS3DX_CAPTURE_OBSERVATION_MASK  Unit tests for gs3dx_capture_points observation mask extension.
%
%   Synthetic fixtures only. Verifies Design-by-Contract for:
%     1. Old/default parity (omitted vs empty [] vs logical empty)
%     2. All-true mask equivalence with 3-argument baseline
%     3. Selective false mask augmentation and NaN coordinate masking
%     4. SI unit scaling with mask ('m', 'mm', 'cm')
%     5. Existing invalid residuals / nonfinite coordinates cannot be revived
%     6. Singleton dimensions (singleton frame Nx1, singleton marker 1xT, scalar 1x1)
%     7. Malformed mask types rejection (numeric, char, struct, cell)
%     8. Malformed mask shapes rejection (4D, wrong marker count, wrong frame count, bad dim 1)
%
%   RED Phase: Calling with 4 arguments fails against the old 3-argument API with
%   MATLAB:TooManyInputs (legitimate contract failure, not fabricated harness error).

    tests = functiontests(localfunctions);
end

function test_old_default_parity(testCase)
    % 3 markers, 4 frames synthetic fixture
    pts = zeros(4, 3, 4);
    pts(1:3, :, :) = repmat(reshape([1.0; 2.0; 3.0], [3, 1, 1]), [1, 3, 4]);
    pts(4, :, :) = 1.0; % Homogeneous coordinate 1
    res = ones(1, 3, 4) * 0.05;

    % Existing natural defects
    res(1, 2, 2) = -1.0;   % negative residual
    pts(2, 1, 3) = NaN;    % nonfinite coordinate

    % Baseline: 3-argument call
    [pts_base, mask_base] = gs3dx_capture_points(pts, 'm', res);

    % Parity: 4-argument call with numeric empty []
    [pts_empty, mask_empty] = gs3dx_capture_points(pts, 'm', res, []);
    verifyEqual(testCase, mask_empty, mask_base);
    verifyEqual(testCase, pts_empty, pts_base);

    % Parity: 4-argument call with logical empty false(0)
    [pts_log_empty, mask_log_empty] = gs3dx_capture_points(pts, 'm', res, false(0));
    verifyEqual(testCase, mask_log_empty, mask_base);
    verifyEqual(testCase, pts_log_empty, pts_base);
end

function test_all_true_mask(testCase)
    % All-true mask must produce identical results to old 3-arg call
    pts = zeros(4, 2, 3);
    pts(1:3, :, :) = repmat(reshape([10.0; -20.0; 30.0], [3, 1, 1]), [1, 2, 3]);
    pts(4, :, :) = 1.0;
    res = ones(1, 2, 3) * 0.02;

    [pts_base, mask_base] = gs3dx_capture_points(pts, 'm', res);

    obs_mask = true(1, 2, 3);
    [pts_masked, mask_masked] = gs3dx_capture_points(pts, 'm', res, obs_mask);

    verifyEqual(testCase, mask_masked, mask_base);
    verifyEqual(testCase, pts_masked, pts_base);
    verifyFalse(testCase, any(mask_masked, 'all'));
end

function test_selective_false_mask(testCase)
    % False entries in observed_mask must augment missing_mask and set all 3 XYZ to NaN
    pts = zeros(4, 3, 4);
    pts(1:3, :, :) = repmat(reshape([1.0; 2.0; 3.0], [3, 1, 1]), [1, 3, 4]);
    pts(4, :, :) = 1.0;
    res = ones(1, 3, 4) * 0.05;

    obs_mask = true(1, 3, 4);
    obs_mask(1, 1, 2) = false; % marker 1, frame 2 unobserved
    obs_mask(1, 3, 4) = false; % marker 3, frame 4 unobserved

    [pts_si, mask] = gs3dx_capture_points(pts, 'm', res, obs_mask);

    % Exactly 2 samples must be marked missing
    verifyEqual(testCase, nnz(mask), 2);
    verifyTrue(testCase, mask(1, 1, 2));
    verifyTrue(testCase, mask(1, 3, 4));

    % Masked samples must have all 3 XYZ coordinates set to NaN
    verifyTrue(testCase, all(isnan(pts_si(:, 1, 2))));
    verifyTrue(testCase, all(isnan(pts_si(:, 3, 4))));

    % Unmasked samples must remain finite and correctly transformed (x, -z, y)
    verifyFalse(testCase, mask(1, 1, 1));
    verifyEqual(testCase, pts_si(:, 1, 1), [1.0; -3.0; 2.0], 'AbsTol', 1e-15);
    verifyFalse(testCase, mask(1, 2, 2));
    verifyEqual(testCase, pts_si(:, 2, 2), [1.0; -3.0; 2.0], 'AbsTol', 1e-15);
end

function test_unit_transform_with_mask(testCase)
    % Coordinates must scale correctly according to declared units ('mm')
    % and Simscape Z-up mapping (x, -z, y), with masked entries remaining NaN
    pts = zeros(4, 2, 1);
    pts(:, 1, 1) = [1500; 2500; 3500; 1.0];
    pts(:, 2, 1) = [1000; 2000; 3000; 1.0];
    res = ones(1, 2, 1) * 0.01;

    obs_mask = [true, false]; % Marker 1 valid, Marker 2 masked

    [pts_si, mask] = gs3dx_capture_points(pts, 'mm', res, obs_mask);

    % Marker 1: x_si = 1.5, y_si = -3.5, z_si = 2.5
    verifyFalse(testCase, mask(1, 1, 1));
    verifyEqual(testCase, pts_si(:, 1, 1), [1.5; -3.5; 2.5], 'AbsTol', 1e-14);

    % Marker 2: masked to NaN
    verifyTrue(testCase, mask(1, 2, 1));
    verifyTrue(testCase, all(isnan(pts_si(:, 2, 1))));
end

function test_existing_invalid_residual_cannot_be_revived(testCase)
    % Samples with negative residuals or nonfinite data MUST NOT be revived by true mask
    pts = zeros(4, 3, 1);
    pts(:, 1, 1) = [1.0; 2.0; 3.0; 1.0];
    pts(:, 2, 1) = [1.0; 2.0; 3.0; 1.0];
    pts(:, 3, 1) = [1.0; NaN; 3.0; 1.0]; % nonfinite coordinate

    res = zeros(1, 3, 1);
    res(1, 1, 1) = -1.0; % negative residual
    res(1, 2, 1) = NaN;  % nonfinite residual
    res(1, 3, 1) = 0.05; % valid residual, but coordinate is NaN

    % Pass all-true observation mask
    obs_mask = true(1, 3, 1);

    [pts_si, mask] = gs3dx_capture_points(pts, 'm', res, obs_mask);

    % All 3 samples must remain missing despite true observation mask
    verifyTrue(testCase, mask(1, 1, 1));
    verifyTrue(testCase, mask(1, 2, 1));
    verifyTrue(testCase, mask(1, 3, 1));
    verifyTrue(testCase, all(isnan(pts_si(:, 1, 1))));
    verifyTrue(testCase, all(isnan(pts_si(:, 2, 1))));
    verifyTrue(testCase, all(isnan(pts_si(:, 3, 1))));
end

function test_singleton_dimensions(testCase)
    % Test singleton frame, singleton marker, and 1x1 scalar capture
    % 1. Singleton frame: T = 1, N = 3
    pts_f1 = zeros(4, 3, 1);
    pts_f1(1:3, :, 1) = repmat([1.0; 2.0; 3.0], [1, 3]);
    pts_f1(4, :, 1) = 1.0;
    res_f1 = ones(1, 3, 1) * 0.05;

    % Singleton frame: canonical 1 x N
    [~, mask_canon] = gs3dx_capture_points(pts_f1, 'm', res_f1, [true, false, true]);
    verifyTrue(testCase, mask_canon(1, 2, 1));
    verifyFalse(testCase, mask_canon(1, 1, 1));

    % Singleton frame: N x 1 frame column vector
    [~, mask_col] = gs3dx_capture_points(pts_f1, 'm', res_f1, [true; false; true]);
    verifyEqual(testCase, mask_col, mask_canon);

    % 2. Singleton marker: N = 1, T = 3
    pts_m1 = repmat(reshape([1.0; 2.0; 3.0; 1.0], [4, 1, 1]), [1, 1, 3]);
    res_m1 = ones(1, 1, 3) * 0.05;

    % Singleton marker: 1 x 1 x T canonical
    mask_m1_in = reshape([false; true; false], [1, 1, 3]);
    [~, mask_m1_out] = gs3dx_capture_points(pts_m1, 'm', res_m1, mask_m1_in);
    verifyTrue(testCase, mask_m1_out(1, 1, 1));
    verifyFalse(testCase, mask_m1_out(1, 1, 2));
    verifyTrue(testCase, mask_m1_out(1, 1, 3));

    % Singleton marker: 1 x T row vector
    [~, mask_m1_row] = gs3dx_capture_points(pts_m1, 'm', res_m1, [false, true, false]);
    verifyEqual(testCase, mask_m1_out, mask_m1_row);

    % 3. Scalar singleton: N = 1, T = 1
    pts_s1 = [1.0; 2.0; 3.0; 1.0];
    res_s1 = 0.05;
    [pts_s1_out, mask_s1] = gs3dx_capture_points(pts_s1, 'm', res_s1, false);
    verifyTrue(testCase, mask_s1(1, 1));
    verifyTrue(testCase, all(isnan(pts_s1_out)));
end

function test_malformed_mask_types(testCase)
    % Reject non-logical types fail-closed with badObservedMask
    pts = repmat(reshape([1.0; 2.0; 3.0; 1.0], [4, 1, 1]), [1, 2, 2]);
    res = ones(1, 2, 2) * 0.05;

    % Double numeric
    verifyError(testCase, @() gs3dx_capture_points(pts, 'm', res, ones(1, 2, 2)), ...
        'gs3dx:capture_points:badObservedMask');
    % Single numeric
    verifyError(testCase, @() gs3dx_capture_points(pts, 'm', res, single(ones(1, 2, 2))), ...
        'gs3dx:capture_points:badObservedMask');
    % Integer
    verifyError(testCase, @() gs3dx_capture_points(pts, 'm', res, int32(ones(1, 2, 2))), ...
        'gs3dx:capture_points:badObservedMask');
    % Char
    verifyError(testCase, @() gs3dx_capture_points(pts, 'm', res, 'true'), ...
        'gs3dx:capture_points:badObservedMask');
    % Struct
    verifyError(testCase, @() gs3dx_capture_points(pts, 'm', res, struct('val', true)), ...
        'gs3dx:capture_points:badObservedMask');
    % Cell
    verifyError(testCase, @() gs3dx_capture_points(pts, 'm', res, {true, true; true, true}), ...
        'gs3dx:capture_points:badObservedMask');
end

function test_malformed_mask_shapes(testCase)
    % Reject 4D arrays and mismatched dimensions fail-closed
    pts = repmat(reshape([1.0; 2.0; 3.0; 1.0], [4, 1, 1]), [1, 2, 3]);
    res = ones(1, 2, 3) * 0.05;

    % 4D mask
    verifyError(testCase, @() gs3dx_capture_points(pts, 'm', res, repmat(true(1, 2, 3), [1, 1, 1, 2])), ...
        'gs3dx:capture_points:badObservedMask');

    % Wrong marker count (N = 3 instead of 2)
    verifyError(testCase, @() gs3dx_capture_points(pts, 'm', res, true(1, 3, 3)), ...
        'gs3dx:capture_points:dimensionMismatch');

    % Wrong frame count (T = 4 instead of 3)
    verifyError(testCase, @() gs3dx_capture_points(pts, 'm', res, true(1, 2, 4)), ...
        'gs3dx:capture_points:dimensionMismatch');

    % Wrong first dimension (2 instead of 1)
    verifyError(testCase, @() gs3dx_capture_points(pts, 'm', res, true(2, 2, 3)), ...
        'gs3dx:capture_points:dimensionMismatch');
end

function test_empty_mask_contract(testCase)
    pts = repmat([1; 2; 3; 1], [1, 2, 3]);
    res = zeros(1, 2, 3);
    bad = {int32([]), single([]), false(0, 2), false(0, 0, 1, 2)};
    for k = 1:numel(bad)
        verifyError(testCase, @() gs3dx_capture_points(pts, 'm', res, bad{k}), ...
            'gs3dx:capture_points:badObservedMask');
    end
end
