classdef test_gs3dx_quat < matlab.unittest.TestCase
%TEST_GS3DX_QUAT  GS3DX_Quat: quaternion shoulders (#10955) and hip (#10956).
%
%   GS3DX_KDS_Spherical replaces GS3DX_KDS_Gimbal's Gimbal Joint with a
%   Spherical Joint and keeps the subsystem interface: the Torque X/Y/Z
%   inputs and every SignalBus element keep their Gimbal-axis meaning
%   (GS3DX_XYZ_MAP).  Equivalence is shown on an isolated, torqued joint
%   rig (GS3DX_JOINT_RIG), because the full model has no torqued,
%   well-conditioned drive (docs/SENSITIVITY_FINDINGS.md).  The hip's
%   Bushing Joint becomes a 6-DOF Joint (quaternion rotation, same three
%   prismatics) the same way, shown on GS3DX_HIP_RIG.

    properties
        info struct
        names struct
    end

    properties (Constant)
        Spherical = 'sm_lib/Joints/Spherical Joint'
        Gimbal    = 'sm_lib/Joints/Gimbal Joint'
        SixDof    = 'sm_lib/Joints/6-DOF Joint'
        Bushing   = 'sm_lib/Joints/Bushing Joint'
        Hip       = 'Hips and Torso Inputs/Hip Kinetically Driven'
        % Rig solved at RelTol = AbsTol = 1e-8; observed differences are
        % <= 2e-7 of each signal's peak (hip rig: <= 1.9e-6).
        RigRelTol = 1e-5
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info  = gs3dx_setup();
            testCase.names = gs3dx_names();
            quat = char(testCase.names.variants.quat);
            if ~isfile(fullfile(testCase.info.models_dir, [quat '.slx']))
                gs3dx_build_quat(testCase.info);
            end
        end
    end

    methods (Test)
        function spherical_subsystem_has_one_spherical_joint_and_no_gimbal(testCase)
            mdl = testCase.names.spherical_subsys;
            load_system(mdl);
            testCase.verifyEmpty(find_system(mdl, 'ReferenceBlock', testCase.Gimbal));
            joints = find_system(mdl, 'ReferenceBlock', testCase.Spherical);
            testCase.verifyEqual(numel(joints), 1);
            testCase.verifyEqual(get_param(joints{1}, 'Name'), 'Kinetically Driven');
        end

        function spherical_subsystem_keeps_the_gimbal_interface(testCase)
            gimbal = testCase.names.slim_subsys('Gimbal');
            sph = testCase.names.spherical_subsys;
            load_system(gimbal); load_system(sph);
            ports = @(m) sort(get_param(find_system(m, 'SearchDepth', 1, 'RegExp', 'on', ...
                'BlockType', '^(Inport|Outport|PMIOPort)$'), 'Name'));
            testCase.verifyEqual(ports(sph), ports(gimbal));
            bus = @(m) get_param(find_system(m, 'SearchDepth', 1, 'BlockType', 'BusCreator'), 'InputSignalNames');
            testCase.verifyEqual(bus(sph), bus(gimbal));
            params = @(m) {Simulink.Mask.get(m).Parameters.Name};
            testCase.verifyEqual(params(sph), params(gimbal));
        end

        function quat_model_references_spherical_only_at_the_shoulders(testCase)
            quat = char(testCase.names.variants.quat);
            load_system(quat);
            blocks = find_system(quat, 'LookUnderMasks', 'all', 'LookInsideSubsystemReference', 'off', ...
                'BlockType', 'SubSystem', 'ReferencedSubsystem', testCase.names.spherical_subsys);
            testCase.verifyEqual(sort(blocks), sort({[quat '/Left Shoulder Joint/Gimbal Joint']; ...
                [quat '/Right Shoulder Joint/Gimbal Joint']}));
            testCase.verifyEmpty(find_system(quat, 'LookUnderMasks', 'all', 'LookInsideSubsystemReference', 'off', ...
                'BlockType', 'SubSystem', 'ReferencedSubsystem', testCase.names.slim_subsys('Gimbal')));
        end

        function quat_hip_is_a_6dof_joint(testCase)
            quat = char(testCase.names.variants.quat);
            load_system(quat);
            hip = [quat '/' testCase.Hip];
            testCase.verifyEmpty(find_system(hip, 'LookUnderMasks', 'all', 'ReferenceBlock', testCase.Bushing));
            testCase.verifyEqual(find_system(hip, 'LookUnderMasks', 'all', 'ReferenceBlock', testCase.SixDof), ...
                {[hip '/Hip Joint']});
            slim = char(testCase.names.variants.slim);
            load_system(slim);
            bus = @(m) get_param(find_system([m '/' testCase.Hip], 'SearchDepth', 1, 'BlockType', 'BusCreator'), ...
                'InputSignalNames');
            testCase.verifyEqual(bus(quat), bus(slim));
        end

        function quat_model_saves_fifteen_blocks(testCase)
            % 10 from the two shoulders, 5 from the hip.
            slim = local_fresh_budget(char(testCase.names.variants.slim));
            quat = local_fresh_budget(char(testCase.names.variants.quat));
            testCase.verifyEqual(slim.nonvirtual_total - quat.nonvirtual_total, 15);
        end

        function builder_refuses_to_overwrite(testCase)
            testCase.verifyError(@() gs3dx_build_quat(testCase.info), 'gs3dx:quat');
        end
    end

    methods (Test, TestTags = {'Simulation'})
        function spherical_rig_matches_gimbal_rig(testCase)
            g = gs3dx_joint_rig(testCase.names.slim_subsys('Gimbal'));
            s = gs3dx_joint_rig(testCase.names.spherical_subsys);
            testCase.verifyEqual(sort(fieldnames(s.signals)), sort(fieldnames(g.signals)));
            for f = reshape(fieldnames(g.signals), 1, [])
                a = g.signals.(f{1});
                testCase.verifyLessThanOrEqual(max(abs(s.signals.(f{1})(:) - a(:))), ...
                    1e-9 + testCase.RigRelTol * max(abs(a(:))), f{1});
            end
            fprintf('Joint rig wall time: Gimbal %.2f s, Spherical %.2f s\n', g.wall_s, s.wall_s);
        end

        function sixdof_hip_rig_matches_bushing_hip_rig(testCase)
            slim = char(testCase.names.variants.slim);
            quat = char(testCase.names.variants.quat);
            load_system(slim); load_system(quat);
            b = gs3dx_hip_rig(slim);
            q = gs3dx_hip_rig(quat);
            % The drive must stay clear of the Bushing's Y-angle singularity.
            testCase.assertLessThan(max(abs(b.signals.HipAngularPositionY)), 80);
            testCase.verifyEqual(sort(fieldnames(q.signals)), sort(fieldnames(b.signals)));
            for f = reshape(fieldnames(b.signals), 1, [])
                a = b.signals.(f{1});
                testCase.verifyLessThanOrEqual(max(abs(q.signals.(f{1})(:) - a(:))), ...
                    1e-9 + testCase.RigRelTol * max(abs(a(:))), f{1});
            end
            fprintf('Hip rig wall time: Bushing %.2f s, 6-DOF %.2f s\n', b.wall_s, q.wall_s);
        end

        function quat_converges_to_the_slim_solution(testCase)
            % Different state variables give different truncation errors,
            % so the models agree only as the solver tolerance tightens.
            % Reference: GS3DX_Slim at RelTol 1e-7.  At each looser
            % tolerance GS3DX_Quat must be no farther from the reference
            % than GS3DX_Slim, and the Quat-Slim gap must shrink >= 10x per
            % 100x tolerance.  Drive: GS3DX_PINNED_DRIVE (same start state).
            slim = char(testCase.names.variants.slim);
            quat = char(testCase.names.variants.quat);
            ref = local_pinned_run(testCase.info, slim, '1e-7');
            gap = zeros(1, 2);
            for k = 1:2
                tol = {'1e-3', '1e-5'};
                s = local_pinned_run(testCase.info, slim, tol{k});
                q = local_pinned_run(testCase.info, quat, tol{k});
                for key = ["CHGlobalPosition", "MPGlobalPosition"]
                    testCase.verifyLessThanOrEqual(local_err(q, ref, key), local_err(s, ref, key), ...
                        sprintf('%s at RelTol %s', key, tol{k}));
                end
                gap(k) = local_err(q, s, "CHGlobalPosition");
                fprintf('RelTol %s: |Quat-Slim| %.3g m, steps Quat %d / Slim %d\n', ...
                    tol{k}, gap(k), q.n_steps, s.n_steps);
            end
            testCase.verifyLessThanOrEqual(gap(2), gap(1) / 10);
        end

        function unequal_axis_priorities_are_rejected(testCase)
            testCase.verifyError(@() local_mixed_priority_rig(testCase.names.spherical_subsys), ...
                'Simulink:Masking:Bad_Init_Commands');
        end
    end
end

function local_mixed_priority_rig(subsys)
    rig = 'gs3dx_rig_mixed_priority';
    new_system(rig);
    cleanup = onCleanup(@() close_system(rig, 0));
    blk = add_block('simulink/Ports & Subsystems/Subsystem Reference', [rig '/Joint']);
    set_param(blk, 'ReferencedSubsystem', subsys);
    set_param(blk, 'RxPositionTargetPriority', 'High', 'RyPositionTargetPriority', 'Low');
end

function run = local_pinned_run(info, mdl, rel_tol)
    [vars, blocks] = gs3dx_pinned_drive(info, mdl);
    run = gs3dx_simulate(mdl, variables = vars, block_parameters = blocks, ...
        model_parameters = struct('RelTol', rel_tol));
    assert(run.status == "success", 'gs3dx:test', '%s at RelTol %s: %s', mdl, rel_tol, run.message);
end

function e = local_err(a, b, key)
    k = find(endsWith(a.flat.names, key), 1);
    assert(~isempty(k) && a.flat.names(k) == b.flat.names(k), 'gs3dx:test', 'Signal %s not aligned', key);
    e = max(abs(a.flat.data{k}(:) - b.flat.data{k}(:)));
end

function report = local_fresh_budget(mdl)
% Count from a fresh load (see test_gs3dx_slim).
    if bdIsLoaded(mdl)
        close_system(mdl, 0);
    end
    report = gs3dx_block_budget(mdl);
end
