function tests = test_gs3dx_ik_diagnostics
%TEST_GS3DX_IK_DIAGNOSTICS  Precondition and numerical tests for IK diagnostics (#11160).
%   Covers known geometry/residuals, gaps, units/time, phase ranges (single-sample, all-gap),
%   malformed names, complex inputs, and non-2D finite joints.
    tests = functiontests(localfunctions);
end

function setupOnce(t)
    t.TestData.original_path = path;
    test_dir = fileparts(mfilename('fullpath'));
    tools_dir = fullfile(fileparts(test_dir), 'tools');
    addpath(tools_dir);
end

function teardownOnce(t)
    path(t.TestData.original_path);
end

%% Helper: Create valid synthetic IK struct
function ik = local_make_valid_ik(n_targets, n_frames)
    if nargin < 1
        n_targets = 3;
    end
    if nargin < 2
        n_frames = 5;
    end

    ik = struct();
    ik.model = 'GS3DX_Fit';
    ik.names = arrayfun(@(k) sprintf('target_%d', k), 1:n_targets, 'UniformOutput', false);
    ik.frames = 10 * (1:n_frames);
    ik.t = (0:(n_frames - 1)) * 0.01;
    ik.status = ones(1, n_frames);
    ik.residual = 0.02 * ones(n_targets, n_frames);
    ik.points = zeros(3, n_targets, n_frames);
    for f = 1:n_frames
        for k = 1:n_targets
            ik.points(:, k, f) = [k * 0.1; (f - 1) * 0.05; 0.5];
        end
    end
    ik.joint = zeros(6, n_frames);
end

%% 1. Known Geometry and Residuals
function testKnownGeometryAndResiduals(t)
    % 2 targets, 3 frames, dt = 0.1 s
    ik = struct();
    ik.model = 'GS3DX_Fit';
    ik.names = {'pelvis', 'club_head'};
    ik.frames = [10, 20, 30];
    ik.t = [0.0, 0.1, 0.2];
    ik.status = [1, 1, 1];
    ik.joint = zeros(4, 3);

    % Target 1 (pelvis): moves 0.3 m per 0.1 s -> speed = 3.0 m/s
    % Target 2 (club_head): step 1 moves 0.5 m (5.0 m/s), step 2 moves 1.0 m (10.0 m/s) -> max speed = 10.0 m/s
    ik.points = zeros(3, 2, 3);
    ik.points(:, 1, 1) = [0.0; 0.0; 0.0];
    ik.points(:, 1, 2) = [0.3; 0.0; 0.0];
    ik.points(:, 1, 3) = [0.6; 0.0; 0.0];

    ik.points(:, 2, 1) = [1.0; 0.0; 0.0];
    ik.points(:, 2, 2) = [1.0; 0.5; 0.0];
    ik.points(:, 2, 3) = [1.0; 1.5; 0.0];

    % Residuals:
    % Pelvis: [0.03, 0.04, 0.05]
    % Club head: [0.01, 0.02, 0.03]
    ik.residual = [0.03, 0.04, 0.05; ...
                   0.01, 0.02, 0.03];

    diag = gs3dx_ik_diagnostics(ik);

    % Verify Pelvis metrics
    exp_pelvis_rms = sqrt((0.03^2 + 0.04^2 + 0.05^2) / 3);
    verifyEqual(t, diag.targets.pelvis.residual_rms, exp_pelvis_rms, 'AbsTol', 1e-12);
    verifyEqual(t, diag.targets.pelvis.residual_max, 0.05, 'AbsTol', 1e-12);
    verifyEqual(t, diag.targets.pelvis.worst_frame, 30);
    verifyEqual(t, diag.targets.pelvis.sample_count, 3);
    verifyEqual(t, diag.targets.pelvis.max_step_speed, 3.0, 'AbsTol', 1e-12);

    % Verify Club Head metrics
    exp_club_rms = sqrt((0.01^2 + 0.02^2 + 0.03^2) / 3);
    verifyEqual(t, diag.targets.club_head.residual_rms, exp_club_rms, 'AbsTol', 1e-12);
    verifyEqual(t, diag.targets.club_head.residual_max, 0.03, 'AbsTol', 1e-12);
    verifyEqual(t, diag.targets.club_head.worst_frame, 30);
    verifyEqual(t, diag.targets.club_head.sample_count, 3);
    verifyEqual(t, diag.targets.club_head.max_step_speed, 10.0, 'AbsTol', 1e-12);

    % Verify Aggregate metrics
    f10_rms = sqrt((0.03^2 + 0.01^2) / 2);
    f20_rms = sqrt((0.04^2 + 0.02^2) / 2);
    f30_rms = sqrt((0.05^2 + 0.03^2) / 2);
    exp_agg_rms_mean = mean([f10_rms, f20_rms, f30_rms]);
    exp_agg_rms_max = f30_rms;

    verifyEqual(t, diag.aggregate.rms_mean, exp_agg_rms_mean, 'AbsTol', 1e-12);
    verifyEqual(t, diag.aggregate.rms_max, exp_agg_rms_max, 'AbsTol', 1e-12);
    verifyEqual(t, diag.aggregate.worst_frame, 30);
    verifyEqual(t, diag.aggregate.sample_count, 6);
    verifyEqual(t, diag.aggregate.frame_count, 3);
    verifyEqual(t, diag.aggregate.max_step_speed, 10.0, 'AbsTol', 1e-12);
end

%% 2. Gaps and Missing Data (NaN handling)
function testGapsAndMissingData(t)
    ik = local_make_valid_ik(3, 4);
    ik.t = [0.0, 0.05, 0.10, 0.15];
    ik.frames = [100, 101, 102, 103];

    % Target 1: partial NaNs [0.02, NaN, 0.06, NaN]
    ik.residual(1, :) = [0.02, NaN, 0.06, NaN];

    % Target 2: completely unmeasured (all NaNs)
    ik.residual(2, :) = [NaN, NaN, NaN, NaN];

    % Target 3: fully measured
    ik.residual(3, :) = [0.01, 0.02, 0.03, 0.04];

    diag = gs3dx_ik_diagnostics(ik);

    % Target 1 check: computed only over 2 valid samples
    exp_t1_rms = sqrt((0.02^2 + 0.06^2) / 2);
    verifyEqual(t, diag.targets.target_1.residual_rms, exp_t1_rms, 'AbsTol', 1e-12);
    verifyEqual(t, diag.targets.target_1.residual_max, 0.06, 'AbsTol', 1e-12);
    verifyEqual(t, diag.targets.target_1.worst_frame, 102);
    verifyEqual(t, diag.targets.target_1.sample_count, 2);
    verifyTrue(t, isfinite(diag.targets.target_1.max_step_speed));

    % Target 2 check: all NaN -> NaN summary plus explicit zero count
    verifyTrue(t, isnan(diag.targets.target_2.residual_rms));
    verifyTrue(t, isnan(diag.targets.target_2.residual_max));
    verifyTrue(t, isnan(diag.targets.target_2.worst_frame));
    verifyEqual(t, diag.targets.target_2.sample_count, 0);
    % Speed must be predicted even when measurements absent
    verifyTrue(t, isfinite(diag.targets.target_2.max_step_speed));
    verifyGreaterThan(t, diag.targets.target_2.max_step_speed, 0);

    % Aggregate check: frame 101 has only target 3; frame 103 has only target 3
    verifyEqual(t, diag.aggregate.sample_count, 6);
    verifyEqual(t, diag.aggregate.frame_count, 4);
    verifyTrue(t, isfinite(diag.aggregate.rms_mean));
    verifyTrue(t, isfinite(diag.aggregate.rms_max));
    verifyTrue(t, ismember(diag.aggregate.worst_frame, ik.frames));
end

%% 3. Completely Unmeasured Scenario
function testCompletelyUnmeasured(t)
    ik = local_make_valid_ik(2, 3);
    ik.residual(:) = NaN; % All targets, all frames missing

    diag = gs3dx_ik_diagnostics(ik);

    % Both targets should report NaN and 0 sample count
    for k = 1:2
        nm = sprintf('target_%d', k);
        verifyTrue(t, isnan(diag.targets.(nm).residual_rms));
        verifyTrue(t, isnan(diag.targets.(nm).residual_max));
        verifyTrue(t, isnan(diag.targets.(nm).worst_frame));
        verifyEqual(t, diag.targets.(nm).sample_count, 0);
        % Predicted step speeds remain finite and calculated
        verifyTrue(t, isfinite(diag.targets.(nm).max_step_speed));
    end

    % Aggregate summary
    verifyTrue(t, isnan(diag.aggregate.rms_mean));
    verifyTrue(t, isnan(diag.aggregate.rms_max));
    verifyTrue(t, isnan(diag.aggregate.worst_frame));
    verifyEqual(t, diag.aggregate.sample_count, 0);
    verifyEqual(t, diag.aggregate.frame_count, 0);
    verifyTrue(t, isfinite(diag.aggregate.max_step_speed));
end

%% 4. Units, Non-Uniform Time, and Single-Frame Behavior
function testUnitsAndNonuniformTime(t)
    ik = local_make_valid_ik(1, 3);
    ik.names = {'marker'};
    ik.frames = [5, 10, 20];
    ik.t = [0.0, 0.02, 0.05]; % dt = [0.02, 0.03]
    ik.status = [1, 1, 1];
    ik.residual = [0.01, 0.03, 0.02];

    % Step 1: dx = [0.10, 0, 0] in 0.02 s -> 5.0 m/s
    % Step 2: dx = [0.06, 0, 0] in 0.03 s -> 2.0 m/s
    ik.points = zeros(3, 1, 3);
    ik.points(:, 1, 1) = [0.0; 0.0; 0.0];
    ik.points(:, 1, 2) = [0.10; 0.0; 0.0];
    ik.points(:, 1, 3) = [0.16; 0.0; 0.0];

    diag = gs3dx_ik_diagnostics(ik);
    verifyEqual(t, diag.targets.marker.max_step_speed, 5.0, 'AbsTol', 1e-12);
    verifyEqual(t, diag.aggregate.max_step_speed, 5.0, 'AbsTol', 1e-12);

    % Single-frame test: cannot compute step speed
    ik1 = local_make_valid_ik(1, 1);
    ik1.names = {'marker'};
    ik1.frames = 42;
    ik1.t = 0.5;
    ik1.status = 1;
    ik1.residual = 0.04;
    ik1.points = [0.1; 0.2; 0.3];

    diag1 = gs3dx_ik_diagnostics(ik1);
    verifyEqual(t, diag1.targets.marker.residual_rms, 0.04, 'AbsTol', 1e-12);
    verifyEqual(t, diag1.targets.marker.worst_frame, 42);
    verifyTrue(t, isnan(diag1.targets.marker.max_step_speed));
    verifyTrue(t, isnan(diag1.aggregate.max_step_speed));
end

%% 5. Phase Ranges (Valid Multi-Phase Partitioning)
function testPhaseRangesValid(t)
    ik = local_make_valid_ik(2, 6);
    ik.frames = [10, 20, 30, 40, 50, 60];
    ik.t = (0:5) * 0.1;
    ik.residual = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06; ...
                   0.06, 0.05, 0.04, 0.03, 0.02, 0.01];

    pr = struct();
    pr.backswing = [10, 30];    % Inclusive frames 10, 20, 30
    pr.downswing = [35, 60];    % Inclusive frames 40, 50, 60

    diag = gs3dx_ik_diagnostics(ik, phase_ranges=pr);

    verifyTrue(t, isfield(diag, 'phases'));
    verifyTrue(t, isfield(diag.phases, 'backswing'));
    verifyTrue(t, isfield(diag.phases, 'downswing'));

    % Verify backswing phase subset
    verifyEqual(t, diag.phases.backswing.frames, [10, 20, 30]);
    verifyEqual(t, diag.phases.backswing.t, [0.0, 0.1, 0.2]);
    exp_bs_t1_max = 0.03;
    verifyEqual(t, diag.phases.backswing.targets.target_1.residual_max, exp_bs_t1_max, 'AbsTol', 1e-12);
    verifyEqual(t, diag.phases.backswing.targets.target_1.worst_frame, 30);

    % Verify downswing phase subset
    verifyEqual(t, diag.phases.downswing.frames, [40, 50, 60]);
    verifyEqual(t, diag.phases.downswing.t, [0.3, 0.4, 0.5], 'AbsTol', 1e-14);
    exp_ds_t1_max = 0.06;
    verifyEqual(t, diag.phases.downswing.targets.target_1.residual_max, exp_ds_t1_max, 'AbsTol', 1e-12);
    verifyEqual(t, diag.phases.downswing.targets.target_1.worst_frame, 60);

    % Overall diagnostics must still cover all 6 frames
    verifyEqual(t, diag.frames, [10, 20, 30, 40, 50, 60]);
    verifyEqual(t, diag.targets.target_1.worst_frame, 60);
end

%% 6. Single-Sample Phase Coverage
function testSingleSamplePhase(t)
    ik = local_make_valid_ik(2, 5);
    ik.frames = [10, 20, 30, 40, 50];
    ik.residual = [0.02, 0.04, 0.06, 0.08, 0.10; ...
                   0.01, 0.03, 0.05, 0.07, 0.09];

    % Phase containing exactly frame 30
    pr = struct('impact', [30, 30]);
    diag = gs3dx_ik_diagnostics(ik, phase_ranges=pr);

    verifyEqual(t, diag.phases.impact.frames, 30);
    verifyEqual(t, diag.phases.impact.targets.target_1.sample_count, 1);
    verifyEqual(t, diag.phases.impact.targets.target_1.residual_rms, 0.06, 'AbsTol', 1e-12);
    verifyEqual(t, diag.phases.impact.targets.target_1.residual_max, 0.06, 'AbsTol', 1e-12);
    verifyEqual(t, diag.phases.impact.targets.target_1.worst_frame, 30);
    % Single frame within phase cannot produce step speed -> must be NaN
    verifyTrue(t, isnan(diag.phases.impact.targets.target_1.max_step_speed));
    verifyTrue(t, isnan(diag.phases.impact.aggregate.max_step_speed));
end

%% 7. All-Gap Phase Coverage
function testAllGapPhase(t)
    ik = local_make_valid_ik(2, 5);
    ik.frames = [10, 20, 30, 40, 50];
    ik.t = (0:4) * 0.01;
    % Frames 20 and 30 are completely missing (NaN)
    ik.residual(:, 2:3) = NaN;

    pr = struct('gap_window', [20, 30]);
    diag = gs3dx_ik_diagnostics(ik, phase_ranges=pr);

    verifyEqual(t, diag.phases.gap_window.frames, [20, 30]);
    verifyEqual(t, diag.phases.gap_window.targets.target_1.sample_count, 0);
    verifyTrue(t, isnan(diag.phases.gap_window.targets.target_1.residual_rms));
    verifyTrue(t, isnan(diag.phases.gap_window.targets.target_1.residual_max));
    verifyTrue(t, isnan(diag.phases.gap_window.targets.target_1.worst_frame));

    % Predicted speed is still computed from predicted points across the phase
    verifyTrue(t, isfinite(diag.phases.gap_window.targets.target_1.max_step_speed));
    verifyGreaterThan(t, diag.phases.gap_window.targets.target_1.max_step_speed, 0);
    verifyEqual(t, diag.phases.gap_window.aggregate.sample_count, 0);
    verifyEqual(t, diag.phases.gap_window.aggregate.frame_count, 0);
end

%% 8. Phase Ranges Empty Fail-Closed
function testPhaseRangesEmptyFailClosed(t)
    ik = local_make_valid_ik(2, 4);
    ik.frames = [10, 20, 30, 40];

    % Phase range outside all frames -> must fail closed
    pr = struct();
    pr.impact = [100, 200];

    verifyError(t, @() gs3dx_ik_diagnostics(ik, phase_ranges=pr), ...
        'gs3dx:ik_diagnostics:EmptyPhase');
end

%% 9. Phase Ranges Invalid Bounds
function testPhaseRangesInvalidBounds(t)
    ik = local_make_valid_ik(2, 4);
    ik.frames = [10, 20, 30, 40];

    % Start > End
    pr1 = struct('inverted', [30, 20]);
    verifyError(t, @() gs3dx_ik_diagnostics(ik, phase_ranges=pr1), ...
        'gs3dx:ik_diagnostics:InvalidPhaseRange');

    % Non-integer
    pr2 = struct('non_int', [10.5, 20]);
    verifyError(t, @() gs3dx_ik_diagnostics(ik, phase_ranges=pr2), ...
        'gs3dx:ik_diagnostics:InvalidPhaseRange');

    % Negative
    pr3 = struct('neg', [-10, 20]);
    verifyError(t, @() gs3dx_ik_diagnostics(ik, phase_ranges=pr3), ...
        'gs3dx:ik_diagnostics:InvalidPhaseRange');

    % Not 2 elements
    pr4 = struct('three_elem', [10, 20, 30]);
    verifyError(t, @() gs3dx_ik_diagnostics(ik, phase_ranges=pr4), ...
        'gs3dx:ik_diagnostics:InvalidPhaseRange');
end

%% 10. Malformed Names and Duplicate Normalization
function testMalformedNames(t)
    ik = local_make_valid_ik(2, 3);

    % Name with space
    ik_sp = ik;
    ik_sp.names = {'pelvis joint', 'club_head'};
    verifyError(t, @() gs3dx_ik_diagnostics(ik_sp), ...
        'gs3dx:ik_diagnostics:InvalidNames');

    % Name starting with digit
    ik_num = ik;
    ik_num.names = {'1_pelvis', 'club_head'};
    verifyError(t, @() gs3dx_ik_diagnostics(ik_num), ...
        'gs3dx:ik_diagnostics:InvalidNames');

    % Name with hyphen
    ik_hyph = ik;
    ik_hyph.names = {'pelvis', 'club-head'};
    verifyError(t, @() gs3dx_ik_diagnostics(ik_hyph), ...
        'gs3dx:ik_diagnostics:InvalidNames');

    % Empty string name
    ik_emp = ik;
    ik_emp.names = {'pelvis', ''};
    verifyError(t, @() gs3dx_ik_diagnostics(ik_emp), ...
        'gs3dx:ik_diagnostics:InvalidNames');

    % Non-scalar string element in cell
    ik_vec = ik;
    ik_vec.names = {'pelvis', ["club", "head"]};
    verifyError(t, @() gs3dx_ik_diagnostics(ik_vec), ...
        'gs3dx:ik_diagnostics:InvalidNames');

    % Duplicate after string/char normalization
    ik_dup = ik;
    ik_dup.names = {'pelvis', string("pelvis")};
    verifyError(t, @() gs3dx_ik_diagnostics(ik_dup), ...
        'gs3dx:ik_diagnostics:DuplicateNames');
end

%% 11. Complex Inputs Rejected
function testComplexInputsRejected(t)
    ik = local_make_valid_ik(2, 3);

    % Complex points
    ik_cpts = ik;
    ik_cpts.points(1, 1, 1) = 0.1 + 0.05i;
    verifyError(t, @() gs3dx_ik_diagnostics(ik_cpts), ...
        'gs3dx:ik_diagnostics:InvalidShape');

    % Complex residuals
    ik_cres = ik;
    ik_cres.residual(1, 1) = 0.02 + 0.01i;
    verifyError(t, @() gs3dx_ik_diagnostics(ik_cres), ...
        'gs3dx:ik_diagnostics:InvalidShape');

    % Complex time
    ik_ct = ik;
    ik_ct.t(1) = 0.01i;
    verifyError(t, @() gs3dx_ik_diagnostics(ik_ct), ...
        'gs3dx:ik_diagnostics:InvalidTime');

    % Complex frames
    ik_cf = ik;
    ik_cf.frames(1) = 10 + 1i;
    verifyError(t, @() gs3dx_ik_diagnostics(ik_cf), ...
        'gs3dx:ik_diagnostics:InvalidFrames');

    % Complex status
    ik_cst = ik;
    ik_cst.status(1) = 1 + 1i;
    verifyError(t, @() gs3dx_ik_diagnostics(ik_cst), ...
        'gs3dx:ik_diagnostics:InvalidStatus');
end

%% 12. Non-2D and Finite Joints Verification
function testNon2DFiniteJoints(t)
    ik = local_make_valid_ik(2, 3);

    % 3D joint array
    ik_3dj = ik;
    ik_3dj.joint = zeros(3, 4, 3);
    verifyError(t, @() gs3dx_ik_diagnostics(ik_3dj), ...
        'gs3dx:ik_diagnostics:InvalidShape');

    % Complex joints
    ik_cj = ik;
    ik_cj.joint = complex(zeros(6, 3));
    ik_cj.joint(1, 1) = 0.5i;
    verifyError(t, @() gs3dx_ik_diagnostics(ik_cj), ...
        'gs3dx:ik_diagnostics:InvalidShape');

    % Non-finite joints (NaN)
    ik_nanj = ik;
    ik_nanj.joint(1, 1) = NaN;
    verifyError(t, @() gs3dx_ik_diagnostics(ik_nanj), ...
        'gs3dx:ik_diagnostics:NonFiniteJoints');

    % Non-finite joints (Inf)
    ik_infj = ik;
    ik_infj.joint(2, 2) = Inf;
    verifyError(t, @() gs3dx_ik_diagnostics(ik_infj), ...
        'gs3dx:ik_diagnostics:NonFiniteJoints');
end

%% 13. Invalid Residuals (Negative and Infinite)
function testInvalidResidualNegativeAndInf(t)
    ik = local_make_valid_ik(2, 3);

    % Negative residual must be rejected
    ik_neg = ik;
    ik_neg.residual(1, 2) = -0.001;
    verifyError(t, @() gs3dx_ik_diagnostics(ik_neg), ...
        'gs3dx:ik_diagnostics:InvalidResidual');

    % Infinite residual must be rejected
    ik_inf = ik;
    ik_inf.residual(2, 1) = Inf;
    verifyError(t, @() gs3dx_ik_diagnostics(ik_inf), ...
        'gs3dx:ik_diagnostics:InvalidResidual');
end

%% 14. Invalid Shapes
function testInvalidShapes(t)
    ik = local_make_valid_ik(2, 4);

    % Residual row mismatch
    ik_bad_res = ik;
    ik_bad_res.residual = zeros(3, 4);
    verifyError(t, @() gs3dx_ik_diagnostics(ik_bad_res), ...
        'gs3dx:ik_diagnostics:InvalidShape');

    % Points 1st dim mismatch
    ik_bad_pts1 = ik;
    ik_bad_pts1.points = zeros(4, 2, 4);
    verifyError(t, @() gs3dx_ik_diagnostics(ik_bad_pts1), ...
        'gs3dx:ik_diagnostics:InvalidShape');

    % Points 2nd dim mismatch
    ik_bad_pts2 = ik;
    ik_bad_pts2.points = zeros(3, 3, 4);
    verifyError(t, @() gs3dx_ik_diagnostics(ik_bad_pts2), ...
        'gs3dx:ik_diagnostics:InvalidShape');

    % Points 3rd dim mismatch
    ik_bad_pts3 = ik;
    ik_bad_pts3.points = zeros(3, 2, 5);
    verifyError(t, @() gs3dx_ik_diagnostics(ik_bad_pts3), ...
        'gs3dx:ik_diagnostics:InvalidShape');

    % Time length mismatch
    ik_bad_t = ik;
    ik_bad_t.t = zeros(1, 3);
    verifyError(t, @() gs3dx_ik_diagnostics(ik_bad_t), ...
        'gs3dx:ik_diagnostics:InvalidTime');

    % Status length mismatch
    ik_bad_st = ik;
    ik_bad_st.status = [1, 1];
    verifyError(t, @() gs3dx_ik_diagnostics(ik_bad_st), ...
        'gs3dx:ik_diagnostics:InvalidStatus');

    % Joint column mismatch
    ik_bad_j = ik;
    ik_bad_j.joint = zeros(4, 2);
    verifyError(t, @() gs3dx_ik_diagnostics(ik_bad_j), ...
        'gs3dx:ik_diagnostics:InvalidShape');
end

%% 15. Non-Finite Points
function testNonFinitePoints(t)
    ik = local_make_valid_ik(2, 3);

    % Points with NaN
    ik_pts_nan = ik;
    ik_pts_nan.points(1, 1, 1) = NaN;
    verifyError(t, @() gs3dx_ik_diagnostics(ik_pts_nan), ...
        'gs3dx:ik_diagnostics:NonFinitePoints');

    % Points with Inf
    ik_pts_inf = ik;
    ik_pts_inf.points(2, 2, 2) = Inf;
    verifyError(t, @() gs3dx_ik_diagnostics(ik_pts_inf), ...
        'gs3dx:ik_diagnostics:NonFinitePoints');
end

%% 16. Invalid Time
function testInvalidTime(t)
    ik = local_make_valid_ik(2, 3);

    % Non-increasing time
    ik_t_dec = ik;
    ik_t_dec.t = [0.0, 0.2, 0.1];
    verifyError(t, @() gs3dx_ik_diagnostics(ik_t_dec), ...
        'gs3dx:ik_diagnostics:InvalidTime');

    % Constant time step (diff = 0)
    ik_t_const = ik;
    ik_t_const.t = [0.1, 0.1, 0.2];
    verifyError(t, @() gs3dx_ik_diagnostics(ik_t_const), ...
        'gs3dx:ik_diagnostics:InvalidTime');

    % Non-finite time
    ik_t_nan = ik;
    ik_t_nan.t = [0.0, NaN, 0.2];
    verifyError(t, @() gs3dx_ik_diagnostics(ik_t_nan), ...
        'gs3dx:ik_diagnostics:InvalidTime');
end

%% 17. Invalid Frames
function testInvalidFrames(t)
    ik = local_make_valid_ik(2, 3);

    % Non-integer frames
    ik_f_float = ik;
    ik_f_float.frames = [1.0, 2.5, 3.0];
    verifyError(t, @() gs3dx_ik_diagnostics(ik_f_float), ...
        'gs3dx:ik_diagnostics:InvalidFrames');

    % Zero or negative frames
    ik_f_zero = ik;
    ik_f_zero.frames = [0, 1, 2];
    verifyError(t, @() gs3dx_ik_diagnostics(ik_f_zero), ...
        'gs3dx:ik_diagnostics:InvalidFrames');

    % Non-increasing frames
    ik_f_dec = ik;
    ik_f_dec.frames = [10, 5, 20];
    verifyError(t, @() gs3dx_ik_diagnostics(ik_f_dec), ...
        'gs3dx:ik_diagnostics:InvalidFrames');
end

%% 18. Status Loop Closure Required
function testStatusLoopClosureRequired(t)
    ik = local_make_valid_ik(2, 3);

    % Status indicating loop closure failure (status == 0)
    ik_fail = ik;
    ik_fail.status = [1, 0, 1];
    verifyError(t, @() gs3dx_ik_diagnostics(ik_fail), ...
        'gs3dx:ik_diagnostics:LoopClosure');
end
