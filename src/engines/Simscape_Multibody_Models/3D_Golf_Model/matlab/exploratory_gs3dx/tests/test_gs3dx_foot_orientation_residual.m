function tests = test_gs3dx_foot_orientation_residual
%TEST_GS3DX_FOOT_ORIENTATION_RESIDUAL  Unit tests for pure foot orientation residual helper (#10979, #11161).
%   Proves chordal SO(3) metric, full roll/pitch/yaw rejection (including sole-inversion
%   and backward yaw), analytical 0/90/180-deg errors, SO(3) validation, and gap masking.
    tests = functiontests(localfunctions);
end

function setupOnce(t)
    t.TestData.original_path = path;
    tools_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'tools');
    addpath(tools_dir);
end

function teardownOnce(t)
    path(t.TestData.original_path);
end

%% 1. Zero Rotation Yields Zero Residual and Zero Error
function testZeroRotationYieldsZeroResidual(t)
    R_pred = repmat(eye(3), [1 1 2]);
    R_target = repmat(eye(3), [1 1 2]);
    gaps = [false, false];
    weight = 0.05;

    [r, info] = gs3dx_foot_orientation_residual(R_pred, R_target, gaps, weight);

    verifySize(t, r, [18, 1]);
    verifyEqual(t, r, zeros(18, 1), 'AbsTol', 1e-12);
    verifyEqual(t, info.err_chordal, [0.0, 0.0], 'AbsTol', 1e-12);
    verifyEqual(t, info.err_deg, [0.0, 0.0], 'AbsTol', 1e-12);
    verifyEqual(t, info.err_rad, [0.0, 0.0], 'AbsTol', 1e-12);
    verifyEqual(t, info.valid, [true, true]);
end

%% 2. Analytical Chordal and Angular Error at 90 Deg Yaw
function testAnalyticalChordalError90Deg(t)
    R_target = repmat(eye(3), [1 1 2]);
    % Left foot aligned, Right foot yaw rotated by 90 deg
    R_L = eye(3);
    R_R = [0 -1 0; 1 0 0; 0 0 1];
    R_pred = cat(3, R_L, R_R);
    gaps = [false, false];
    weight = 0.04;

    [r, info] = gs3dx_foot_orientation_residual(R_pred, R_target, gaps, weight);

    % Left foot: 0 residual
    verifyEqual(t, r(1:9), zeros(9, 1), 'AbsTol', 1e-12);
    verifyEqual(t, info.err_chordal(1), 0.0, 'AbsTol', 1e-12);
    verifyEqual(t, info.err_deg(1), 0.0, 'AbsTol', 1e-12);

    % Right foot: chordal error is sqrt(2), angular error is 90 deg
    verifyEqual(t, info.err_chordal(2), sqrt(2), 'AbsTol', 1e-12);
    verifyEqual(t, info.err_deg(2), 90.0, 'AbsTol', 1e-12);
    verifyEqual(t, info.err_rad(2), pi / 2, 'AbsTol', 1e-12);
    verifyEqual(t, norm(r(10:18)), weight * sqrt(2), 'AbsTol', 1e-12);
end

%% 3. Analytical Chordal and Angular Error at 180 Deg Yaw (Backward Foot)
function testAnalyticalChordalError180DegYaw(t)
    R_target = repmat(eye(3), [1 1 2]);
    % Right foot flipped backward (yaw 180 deg)
    R_R = [-1 0 0; 0 -1 0; 0 0 1];
    R_pred = cat(3, eye(3), R_R);
    gaps = [false, false];
    weight = 0.05;

    [r, info] = gs3dx_foot_orientation_residual(R_pred, R_target, gaps, weight);

    verifyEqual(t, info.err_chordal(2), 2.0, 'AbsTol', 1e-12);
    verifyEqual(t, info.err_deg(2), 180.0, 'AbsTol', 1e-12);
    verifyEqual(t, norm(r(10:18)), 2.0 * weight, 'AbsTol', 1e-12);
end

%% 4. Full Rejection of Sole-Inverted 180 Deg Roll
function testRejectsSoleInvertedRoll180Deg(t)
    R_target = repmat(eye(3), [1 1 2]);
    % Right foot sole turned completely upside-down (roll 180 deg about +x)
    % Notice forward axis +x remains [1; 0; 0], but sole is inverted!
    R_R = [1 0 0; 0 -1 0; 0 0 -1];
    R_pred = cat(3, eye(3), R_R);
    gaps = [false, false];
    weight = 0.06;

    [r, info] = gs3dx_foot_orientation_residual(R_pred, R_target, gaps, weight);

    % Proves the full orientation residual catches inverted soles that forward-only missed:
    verifyEqual(t, info.err_chordal(2), 2.0, 'AbsTol', 1e-12);
    verifyEqual(t, info.err_deg(2), 180.0, 'AbsTol', 1e-12);
    verifyEqual(t, norm(r(10:18)), 2.0 * weight, 'AbsTol', 1e-12);
end

%% 5. Full Rejection of 180 Deg Pitch
function testRejectsPitch180Deg(t)
    R_target = repmat(eye(3), [1 1 2]);
    % Right foot pitched 180 deg about +y
    R_R = [-1 0 0; 0 1 0; 0 0 -1];
    R_pred = cat(3, eye(3), R_R);
    gaps = [false, false];
    weight = 0.05;

    [r, info] = gs3dx_foot_orientation_residual(R_pred, R_target, gaps, weight);

    verifyEqual(t, info.err_chordal(2), 2.0, 'AbsTol', 1e-12);
    verifyEqual(t, info.err_deg(2), 180.0, 'AbsTol', 1e-12);
    verifyEqual(t, norm(r(10:18)), 2.0 * weight, 'AbsTol', 1e-12);
end

%% 6. Explicit Gap Masking Leaves Zero Residual and NaN Metrics
function testExplicitGapMasksMeasurement(t)
    R_pred = repmat(eye(3), [1 1 2]);
    % Left foot target has NaNs / garbage, but explicit gap=true
    R_target = cat(3, nan(3, 3), eye(3));
    gaps = [true, false];
    weight = 0.05;

    [r, info] = gs3dx_foot_orientation_residual(R_pred, R_target, gaps, weight);

    % Left foot: masked
    verifyEqual(t, r(1:9), zeros(9, 1));
    verifyFalse(t, info.valid(1));
    verifyTrue(t, isnan(info.err_chordal(1)));
    verifyTrue(t, isnan(info.err_deg(1)));

    % Right foot: valid
    verifyEqual(t, r(10:18), zeros(9, 1));
    verifyTrue(t, info.valid(2));
    verifyEqual(t, info.err_chordal(2), 0.0, 'AbsTol', 1e-12);
    verifyEqual(t, info.err_deg(2), 0.0, 'AbsTol', 1e-12);
end

%% 7. Non-SO(3) Predicted Rotation Throws Error
function testNonSO3PredictedRotationThrows(t)
    R_target = repmat(eye(3), [1 1 2]);
    % Sheared/scaled predicted matrix
    R_bad = [2 0 0; 0 1 0; 0 0 1];
    R_pred = cat(3, R_bad, eye(3));

    verifyError(t, @() gs3dx_foot_orientation_residual(R_pred, R_target, [false, false], 0.05), ...
        'gs3dx:foot_residual:invalid_rotation');

    % All-zeros predicted matrix
    R_zero = zeros(3, 3, 2);
    verifyError(t, @() gs3dx_foot_orientation_residual(R_zero, R_target, [false, false], 0.05), ...
        'gs3dx:foot_residual:invalid_rotation');
end

%% 8. Non-SO(3) Target Rotation Without Gap Throws Error
function testNonSO3TargetRotationWithoutGapThrows(t)
    R_pred = repmat(eye(3), [1 1 2]);

    % Reflection matrix (det = -1) with gap=false
    R_refl = diag([1, 1, -1]);
    R_target = cat(3, eye(3), R_refl);
    verifyError(t, @() gs3dx_foot_orientation_residual(R_pred, R_target, [false, false], 0.05), ...
        'gs3dx:foot_residual:invalid_rotation');

    % NaN matrix with gap=false
    R_nan = cat(3, eye(3), nan(3, 3));
    verifyError(t, @() gs3dx_foot_orientation_residual(R_pred, R_nan, [false, false], 0.05), ...
        'gs3dx:foot_residual:invalid_rotation');
end

%% 9. Invalid Weight Parameters Throws Error
function testInvalidWeightsThrows(t)
    R = repmat(eye(3), [1 1 2]);

    % Negative weight
    verifyError(t, @() gs3dx_foot_orientation_residual(R, R, [false, false], -0.05), ?MException);
    % Infinite weight
    verifyError(t, @() gs3dx_foot_orientation_residual(R, R, [false, false], Inf), ?MException);
    % Complex weight
    verifyError(t, @() gs3dx_foot_orientation_residual(R, R, [false, false], 1 + 2i), ?MException);
end

%% 10. Zero Weight Passes Error Diagnostics while Returning Zero Residual
function testZeroWeightPassesDiagnostics(t)
    % 90 deg yaw with opt-out default weight=0
    R_L = eye(3);
    R_R = [0 -1 0; 1 0 0; 0 0 1];
    R_pred = cat(3, R_L, R_R);
    R_target = repmat(eye(3), [1 1 2]);
    gaps = [false, false];
    weight = 0.0;

    [r, info] = gs3dx_foot_orientation_residual(R_pred, R_target, gaps, weight);

    verifyEqual(t, r, zeros(18, 1));
    verifyTrue(t, info.valid(2));
    verifyEqual(t, info.err_chordal(2), sqrt(2), 'AbsTol', 1e-12);
    verifyEqual(t, info.err_deg(2), 90.0, 'AbsTol', 1e-12);
end
