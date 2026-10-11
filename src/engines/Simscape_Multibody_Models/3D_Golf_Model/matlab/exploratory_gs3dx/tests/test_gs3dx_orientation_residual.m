function tests = test_gs3dx_orientation_residual
%TEST_GS3DX_ORIENTATION_RESIDUAL  Unit tests for generic SO(3) orientation residual helper.
%   Proves chordal metric, multi-frame ordering (K=1,2,3), gap masking, fail-closed
%   SO(3) validation, dimension checking, and opt-out weights.
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

%% 1. Identity Yields Zero Residual and Metrics
function testIdentityYieldsZeroResidual(t)
    % Test K=1 single frame
    R1 = eye(3);
    [r1, info1] = gs3dx_orientation_residual(R1, R1, false, 0.05);
    verifySize(t, r1, [9, 1]);
    verifyEqual(t, r1, zeros(9, 1), 'AbsTol', 1e-12);
    verifyEqual(t, info1.err_chordal, 0.0, 'AbsTol', 1e-12);
    verifyEqual(t, info1.err_deg, 0.0, 'AbsTol', 1e-12);
    verifyEqual(t, info1.valid, true);

    % Test K=2 two frames
    R2 = repmat(eye(3), [1, 1, 2]);
    [r2, info2] = gs3dx_orientation_residual(R2, R2, [false, false], 0.05);
    verifySize(t, r2, [18, 1]);
    verifyEqual(t, r2, zeros(18, 1), 'AbsTol', 1e-12);
    verifyEqual(t, info2.err_chordal, [0.0, 0.0], 'AbsTol', 1e-12);
    verifyEqual(t, info2.valid, [true, true]);
end

%% 2. Canonical 90 Deg Rotations (X, Y, Z axes)
function testCanonical90DegRotations(t)
    Rx = [1 0 0; 0 0 -1; 0 1 0];
    Ry = [0 0 1; 0 1 0; -1 0 0];
    Rz = [0 -1 0; 1 0 0; 0 0 1];
    w = 0.04;
    expected_chordal = sqrt(2);

    for R_rot = {Rx, Ry, Rz}
        [r, info] = gs3dx_orientation_residual(R_rot{1}, eye(3), false, w);
        verifyEqual(t, info.err_chordal, expected_chordal, 'AbsTol', 1e-12);
        verifyEqual(t, info.err_deg, 90.0, 'AbsTol', 1e-12);
        verifyEqual(t, info.err_rad, pi / 2, 'AbsTol', 1e-12);
        verifyEqual(t, norm(r), w * expected_chordal, 'AbsTol', 1e-12);
    end
end

%% 3. Canonical 180 Deg Rotations (X, Y, Z axes)
function testCanonical180DegRotations(t)
    Rx = diag([1, -1, -1]);
    Ry = diag([-1, 1, -1]);
    Rz = diag([-1, -1, 1]);
    w = 0.05;

    for R_rot = {Rx, Ry, Rz}
        [r, info] = gs3dx_orientation_residual(R_rot{1}, eye(3), false, w);
        verifyEqual(t, info.err_chordal, 2.0, 'AbsTol', 1e-12);
        verifyEqual(t, info.err_deg, 180.0, 'AbsTol', 1e-12);
        verifyEqual(t, info.err_rad, pi, 'AbsTol', 1e-12);
        verifyEqual(t, norm(r), 2.0 * w, 'AbsTol', 1e-12);
    end
end

%% 4. Two-Frame and Three-Frame Tensor Ordering
function testTensorOrderingTwoAndThreeFrames(t)
    R0 = eye(3);
    R90 = [0 -1 0; 1 0 0; 0 0 1];
    R180 = diag([-1, -1, 1]);
    w = 0.1;

    % K=2: Frame 1=0 deg, Frame 2=90 deg
    R_pred2 = cat(3, R0, R90);
    R_targ2 = repmat(R0, [1, 1, 2]);
    [r2, info2] = gs3dx_orientation_residual(R_pred2, R_targ2, [false, false], w);
    verifySize(t, r2, [18, 1]);
    verifyEqual(t, r2(1:9), zeros(9, 1), 'AbsTol', 1e-12);
    verifyEqual(t, norm(r2(10:18)), w * sqrt(2), 'AbsTol', 1e-12);
    verifyEqual(t, info2.err_deg, [0.0, 90.0], 'AbsTol', 1e-12);

    % K=3: Frame 1=0 deg, Frame 2=90 deg, Frame 3=180 deg
    R_pred3 = cat(3, R0, R90, R180);
    R_targ3 = repmat(R0, [1, 1, 3]);
    [r3, info3] = gs3dx_orientation_residual(R_pred3, R_targ3, [false, false, false], w);
    verifySize(t, r3, [27, 1]);
    verifyEqual(t, r3(1:9), zeros(9, 1), 'AbsTol', 1e-12);
    verifyEqual(t, norm(r3(10:18)), w * sqrt(2), 'AbsTol', 1e-12);
    verifyEqual(t, norm(r3(19:27)), w * 2.0, 'AbsTol', 1e-12);
    verifyEqual(t, info3.err_deg, [0.0, 90.0, 180.0], 'AbsTol', 1e-12);
end

%% 5. Explicit Gap Masking Leaves Zero Residual and NaN Metrics
function testExplicitGapMasking(t)
    R_pred = cat(3, eye(3), eye(3));
    % Target 1 is non-SO3 NaN/garbage, Target 2 is identity
    R_targ = cat(3, nan(3, 3), eye(3));
    gaps = [true, false];
    w = 0.05;

    [r, info] = gs3dx_orientation_residual(R_pred, R_targ, gaps, w);

    verifySize(t, r, [18, 1]);
    verifyEqual(t, r(1:9), zeros(9, 1));
    verifyFalse(t, info.valid(1));
    verifyTrue(t, isnan(info.err_chordal(1)));
    verifyTrue(t, isnan(info.err_deg(1)));
    verifyTrue(t, isnan(info.err_rad(1)));

    verifyEqual(t, r(10:18), zeros(9, 1), 'AbsTol', 1e-12);
    verifyTrue(t, info.valid(2));
    verifyEqual(t, info.err_chordal(2), 0.0, 'AbsTol', 1e-12);
    verifyEqual(t, info.err_deg(2), 0.0, 'AbsTol', 1e-12);
end

%% 6. Non-SO(3) Predicted Rotation Throws EVEN IF Gap Is True
function testInvalidPredictedRotationThrowsEvenWhenGapTrue(t)
    % Sheared predicted matrix with gap=true
    R_sheared = [2 0 0; 0 1 0; 0 0 1];
    R_targ = repmat(eye(3), [1, 1, 2]);
    R_pred1 = cat(3, R_sheared, eye(3));
    verifyError(t, @() gs3dx_orientation_residual(R_pred1, R_targ, [true, false], 0.05), ...
        'gs3dx:orientation_residual:invalid_rotation');

    % Reflection predicted matrix (det = -1) with gap=true
    R_refl = diag([1, 1, -1]);
    R_pred2 = cat(3, eye(3), R_refl);
    verifyError(t, @() gs3dx_orientation_residual(R_pred2, R_targ, [false, true], 0.05), ...
        'gs3dx:orientation_residual:invalid_rotation');

    % Zero predicted matrix with gap=true
    R_zero = zeros(3, 3);
    verifyError(t, @() gs3dx_orientation_residual(R_zero, eye(3), true, 0.05), ...
        'gs3dx:orientation_residual:invalid_rotation');
end

%% 7. Measured Invalid Target Throws When Gap Is False
function testMeasuredInvalidTargetThrowsWithoutGap(t)
    R_pred = repmat(eye(3), [1, 1, 2]);

    % Target is reflection matrix (det = -1)
    R_refl = diag([1, 1, -1]);
    R_targ_refl = cat(3, eye(3), R_refl);
    verifyError(t, @() gs3dx_orientation_residual(R_pred, R_targ_refl, [false, false], 0.05), ...
        'gs3dx:orientation_residual:invalid_rotation');

    % Target is NaN matrix
    R_targ_nan = cat(3, eye(3), nan(3, 3));
    verifyError(t, @() gs3dx_orientation_residual(R_pred, R_targ_nan, [false, false], 0.05), ...
        'gs3dx:orientation_residual:invalid_rotation');

    % Target is complex matrix
    R_targ_cplx = cat(3, eye(3), eye(3) * (1 + 1i));
    verifyError(t, @() gs3dx_orientation_residual(R_pred, R_targ_cplx, [false, false], 0.05), ...
        'gs3dx:orientation_residual:invalid_rotation');
end

%% 8. Dimension and Length Mismatches Throw Invalid Dimensions
function testDimensionMismatchesThrow(t)
    R_2f = repmat(eye(3), [1, 1, 2]);
    R_3f = repmat(eye(3), [1, 1, 3]);

    % R_pred (2 frames) vs R_target (3 frames)
    verifyError(t, @() gs3dx_orientation_residual(R_2f, R_3f, [false, false], 0.05), ...
        'gs3dx:orientation_residual:invalid_dimensions');

    % R_pred (2 frames) vs gaps of length 3
    verifyError(t, @() gs3dx_orientation_residual(R_2f, R_2f, [false, false, false], 0.05), ...
        'gs3dx:orientation_residual:invalid_dimensions');

    % gaps is column vector instead of 1xK row
    verifyError(t, @() gs3dx_orientation_residual(R_2f, R_2f, [false; false], 0.05), ...
        'gs3dx:orientation_residual:invalid_dimensions');

    % R_pred has wrong spatial dimensions (e.g. 2x2 or 4x4)
    verifyError(t, @() gs3dx_orientation_residual(eye(2), eye(2), false, 0.05), ...
        'gs3dx:orientation_residual:invalid_dimensions');
end

%% 9. Weight Parameter Validation and Zero-Weight Diagnostics
function testWeightHandling(t)
    R = eye(3);
    % Invalid weights throw MException
    verifyError(t, @() gs3dx_orientation_residual(R, R, false, -0.05), ?MException);
    verifyError(t, @() gs3dx_orientation_residual(R, R, false, Inf), ?MException);
    verifyError(t, @() gs3dx_orientation_residual(R, R, false, 1 + 2i), ?MException);

    % Zero weight returns zeros residual while reporting valid metrics
    R90 = [0 -1 0; 1 0 0; 0 0 1];
    [r, info] = gs3dx_orientation_residual(R90, R, false, 0.0);
    verifyEqual(t, r, zeros(9, 1));
    verifyTrue(t, info.valid);
    verifyEqual(t, info.err_chordal, sqrt(2), 'AbsTol', 1e-12);
    verifyEqual(t, info.err_deg, 90.0, 'AbsTol', 1e-12);
end

%% 10. Default Arguments Functionality
function testDefaultArguments(t)
    R = repmat(eye(3), [1, 1, 2]);
    % Omit gaps and weight
    [r, info] = gs3dx_orientation_residual(R, R);
    verifySize(t, r, [18, 1]);
    verifyEqual(t, r, zeros(18, 1));
    verifyEqual(t, info.valid, [true, true]);

    % Omit weight only
    [r2, info2] = gs3dx_orientation_residual(R, R, [false, false]);
    verifyEqual(t, r2, zeros(18, 1));
    verifyEqual(t, info2.valid, [true, true]);
end

function testEmptyFramesRejected(t)
    empty_rotations = zeros(3, 3, 0);
    verifyError(t, @() gs3dx_orientation_residual(empty_rotations, ...
        empty_rotations, false(1, 0), 0.05), ...
        'gs3dx:orientation_residual:invalid_dimensions');
end
