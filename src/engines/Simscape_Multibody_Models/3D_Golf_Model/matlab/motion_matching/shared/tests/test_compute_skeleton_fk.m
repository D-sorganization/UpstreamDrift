classdef test_compute_skeleton_fk < matlab.unittest.TestCase
%TEST_COMPUTE_SKELETON_FK  Regression tests for compute_skeleton_fk.
%
%   These tests assert that the sensor-anchored FK chain reproduces the
%   directly-logged wrist positions to within 20 mm at both the Impact
%   pose (3DModelInputs_Impact.mat) and the Top-of-Backswing pose
%   (3DModelInputs_TopofBackswing.mat), per issue #4079.  A synthetic
%   round-trip test verifies the chain reconstruction algebra in
%   isolation from the model run, using a hand-built CombinedSignalBus
%   with known body rotations and the calibrated parent-frame offsets.

    properties (TestParameter)
        pose = {'Impact', 'TopofBackswing'};
    end

    methods (TestClassSetup)
        function add_paths(testCase) %#ok<MANU>
            here = fileparts(mfilename('fullpath'));
            shared_dir = fileparts(here);
            matlab_dir = fileparts(fileparts(shared_dir));
            addpath(genpath(shared_dir));
            src_dir = fullfile(matlab_dir, 'src');
            if exist(src_dir, 'dir')
                addpath(genpath(src_dir));
            end
        end
    end

    methods (Test)
        function wrist_residual_under_20mm_at_impact(testCase)
            % Acceptance criterion from issue #4079: LW and RW residuals
            % below 20 mm at the Impact pose.
            skel = local_run_pose('Impact');
            testCase.verifyLessThan(skel.fk.lw_residual_mm, 20.0, ...
                sprintf('LW residual %.2f mm exceeds 20 mm at Impact', ...
                        skel.fk.lw_residual_mm));
            testCase.verifyLessThan(skel.fk.rw_residual_mm, 20.0, ...
                sprintf('RW residual %.2f mm exceeds 20 mm at Impact', ...
                        skel.fk.rw_residual_mm));
        end

        function wrist_residual_under_20mm_at_top_of_backswing(testCase)
            % Same criterion, second pose.
            skel = local_run_pose('TopofBackswing');
            testCase.verifyLessThan(skel.fk.lw_residual_mm, 20.0, ...
                sprintf('LW residual %.2f mm exceeds 20 mm at TopofBackswing', ...
                        skel.fk.lw_residual_mm));
            testCase.verifyLessThan(skel.fk.rw_residual_mm, 20.0, ...
                sprintf('RW residual %.2f mm exceeds 20 mm at TopofBackswing', ...
                        skel.fk.rw_residual_mm));
        end

        function elbow_residual_also_under_20mm(testCase, pose)
            % Cross-validate the upper-arm rung of the chain.
            skel = local_run_pose(pose);
            testCase.verifyLessThan(skel.fk.le_residual_mm, 20.0);
            testCase.verifyLessThan(skel.fk.re_residual_mm, 20.0);
        end

        function sensor_anchored_mode_used(testCase, pose)
            % The default path should pick the sensor-anchored chain on
            % any sim that has SimscapeLogType='all' (which the model now
            % saves persistently).
            skel = local_run_pose(pose);
            testCase.verifyEqual(skel.fk.mode, 'sensor_anchored');
        end

        function legacy_angle_chain_runs_without_error(testCase)
            % The angle-only fallback is intentionally less accurate, but
            % it must still run end-to-end for older sims with no
            % Transform Sensors.
            skel = local_run_pose('Impact');
            fk = compute_skeleton_fk(skel.sim_out, struct(), ...
                                     struct('frame', 1, 'verbose', false, ...
                                            'force_angle_chain', true));
            testCase.verifyEqual(fk.mode, 'angle_fallback');
            testCase.verifyTrue(all(~isnan(fk.lw)));
            testCase.verifyTrue(all(~isnan(fk.rw)));
        end

        function synthetic_round_trip_recovers_wrist(testCase)
            % Build a synthetic CombinedSignalBus with known body
            % rotations and global positions, run the FK chain, and
            % verify the reconstructed wrist matches the synthesized
            % wrist position to within numerical precision.
            csb = local_synthetic_csb();
            simOut = struct('CombinedSignalBus', csb);
            fk = compute_skeleton_fk(simOut, struct(), ...
                                     struct('frame', 1, 'verbose', false));
            % Synthetic wrist constructed at lf_origin + R_LF * offset.
            % LF_TO_LW_BODY in the calibrated table is [-0.0195; 0.0022; 0.1940].
            R_LF = squeeze(csb.LFLogs.Rotation_Transform.Data(:, :, 1));
            lf_origin = csb.LFLogs.GlobalPosition.Data(1, :);
            expected_lw = lf_origin + (R_LF * [-0.0195; 0.0022; 0.1940]).';
            testCase.verifyEqual(fk.lw, expected_lw, "AbsTol", 1e-9);
            % And it should match the synthetic logged wrist exactly.
            logged_lw = csb.LWLogs.LHGlobalPosition.Data(1, :);
            testCase.verifyEqual(fk.lw, logged_lw, "AbsTol", 1e-6);
        end

        function residuals_constant_within_tolerance_across_frames(testCase)
            % Spot-check the chain at frame 1 and the last logged frame
            % to confirm the calibrated offsets remain pose-invariant
            % during the short t<=stop_time integration window.
            skel = local_run_pose('Impact');
            n = numel(skel.sim_out.tout);
            if n >= 2
                fk_last = compute_skeleton_fk(skel.sim_out, struct(), ...
                                              struct('frame', n, 'verbose', false));
                testCase.verifyLessThan(fk_last.lw_residual_mm, 20.0);
                testCase.verifyLessThan(fk_last.rw_residual_mm, 20.0);
            end
        end
    end
end

%% =====================================================================
function skel = local_run_pose(pose_name)
%LOCAL_RUN_POSE  Run the model at the named pose and return the skeleton
%   struct produced by load_impact_starting_position.
    here = fileparts(mfilename('fullpath'));
    shared_dir = fileparts(here);
    matlab_dir = fileparts(fileparts(shared_dir));
    inputs_dir = fullfile(matlab_dir, 'src', 'model', 'inputs');
    switch pose_name
        case 'Impact'
            input_file = fullfile(inputs_dir, '3DModelInputs_Impact.mat');
        case 'TopofBackswing'
            input_file = fullfile(inputs_dir, '3DModelInputs_TopofBackswing.mat');
        otherwise
            error('test_compute_skeleton_fk:unknownPose', ...
                  'Unknown pose name "%s"', pose_name);
    end
    skel = load_impact_starting_position(struct( ...
        'verbose', false, 'input_file', input_file));
end

%% =====================================================================
function csb = local_synthetic_csb()
%LOCAL_SYNTHETIC_CSB  Build a minimal CombinedSignalBus with known body
%   rotations and consistent global positions for the synthetic
%   round-trip test.  Values are chosen so the chain has non-trivial
%   rotations on every link (no identity matrices).
    OFF = struct( ...
        'SPINE_TO_TORSO_BODY', [ 0.0000;  0.0000;  0.0610], ...
        'TORSO_TO_HUB_BODY',   [ 0.0000;  0.0508;  0.3048], ...
        'LSCAP_TO_LS_BODY',    [ 0.0000;  0.0000; -0.2540], ...
        'RSCAP_TO_RS_BODY',    [ 0.0000;  0.0000;  0.2540], ...
        'LS_TO_LE_BODY',       [ 0.3408;  0.0000;  0.1741], ...
        'RS_TO_RE_BODY',       [ 0.3528;  0.0000;  0.1712], ...
        'LF_TO_LW_BODY',       [-0.0195;  0.0022;  0.1940], ...
        'RF_TO_RW_BODY',       [ 0.0113; -0.0039;  0.2002]);

    Rx = @(d) [1 0 0; 0 cosd(d) -sind(d); 0 sind(d) cosd(d)];
    Ry = @(d) [cosd(d) 0 sind(d); 0 1 0; -sind(d) 0 cosd(d)];
    Rz = @(d) [cosd(d) -sind(d) 0; sind(d) cosd(d) 0; 0 0 1];

    % All landmark positions held as 1x3 row vectors.
    hip   = [0, 0, 0];
    spine = hip + [0, -0.183, 0.317];

    R_spine  = Rz(-30);
    R_torso  = R_spine  * Rx(20);
    R_LScap  = R_torso  * Rx(15)  * Ry(-10);
    R_RScap  = R_torso  * Rx(-12) * Ry(8);
    R_LS     = R_LScap  * Rx(40)  * Ry(-25) * Rz(-90);
    R_RS     = R_RScap  * Rx(-30) * Ry(20)  * Rz(85);
    R_LF     = R_LS     * Rz(70);
    R_RF     = R_RS     * Rz(60);

    torso = spine + (R_spine  * OFF.SPINE_TO_TORSO_BODY).';
    hub   = torso + (R_torso  * OFF.TORSO_TO_HUB_BODY).';
    ls    = hub   + (R_LScap  * OFF.LSCAP_TO_LS_BODY).';
    rs    = hub   + (R_RScap  * OFF.RSCAP_TO_RS_BODY).';
    le    = ls    + (R_LS     * OFF.LS_TO_LE_BODY).';
    re    = rs    + (R_RS     * OFF.RS_TO_RE_BODY).';
    lf_origin = le;     % The synthetic LF body sits at the elbow for this test
    rf_origin = re;
    lw    = lf_origin + (R_LF * OFF.LF_TO_LW_BODY).';
    rw    = rf_origin + (R_RF * OFF.RF_TO_RW_BODY).';

    % --- Compose CSB struct in the shape compute_skeleton_fk expects.
    ts_pos = @(p) struct('Data', reshape(p, 1, 3));
    ts_rot = @(R) struct('Data', reshape(R, 3, 3, 1));
    ts_sca = @(v) struct('Data', double(v));

    csb = struct();
    csb.AngularKinematicsLogs = struct( ...
        'HipPositionX', ts_sca(hip(1)), ...
        'HipPositionY', ts_sca(hip(2)), ...
        'HipPositionZ', ts_sca(hip(3)));
    csb.SpineLogs   = struct('GlobalPosition', ts_pos(spine),     'Rotation_Transform', ts_rot(R_spine));
    csb.TorsoLogs   = struct('GlobalPosition', ts_pos(torso),     'Rotation_Transform', ts_rot(R_torso));
    csb.HipLogs     = struct('HUBGlobalPosition', ts_pos(hub));
    csb.LScapLogs   = struct('GlobalPosition', ts_pos(ls),        'Rotation_Transform', ts_rot(R_LScap));
    csb.RScapLogs   = struct('GlobalPosition', ts_pos(rs),        'Rotation_Transform', ts_rot(R_RScap));
    csb.LSLogs      = struct('GlobalPosition', ts_pos(ls),        'Rotation_Transform', ts_rot(R_LS));
    csb.RSLogs      = struct('GlobalPosition', ts_pos(rs),        'Rotation_Transform', ts_rot(R_RS));
    csb.LFLogs      = struct('GlobalPosition', ts_pos(lf_origin), 'Rotation_Transform', ts_rot(R_LF));
    csb.RFLogs      = struct('GlobalPosition', ts_pos(rf_origin), 'Rotation_Transform', ts_rot(R_RF));
    csb.LWLogs      = struct('LHGlobalPosition', ts_pos(lw));
    csb.RWLogs      = struct('RHGlobalPosition', ts_pos(rw));
end
