classdef test_simulate_golf_markers < matlab.unittest.TestCase
    methods (Test)
        function rejectsNonIncreasingClock(testCase)
            testCase.verifyError(@() simulate_golf_markers([],struct(),[],struct(), ...
                1,zeros(1,3),[0;0]), 'simulate_golf_markers:clock');
        end
        function followsMeasuredBodyRotation(testCase)
            shared = fileparts(fileparts(mfilename('fullpath')));
            engine = fileparts(fileparts(shared));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                fullfile(engine,'src','model')));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                fullfile(engine,'src','functions'),'IncludingSubfolders',true));
            load_system('GolfSwing3D_Kinetic');
            testCase.addTeardown(@() bdclose('GolfSwing3D_Kinetic'));
            [ks,schema] = build_golf_kinematics();
            addTargetVariables(ks,schema.q_ids);
            addOutputVariables(ks,schema.frame_ids);
            addOutputVariables(ks,schema.rotation_ids);
            [~,bodies] = ismember(["LS";"RS"],string({schema.frames.name}));
            offsets = [0.02 0.01 0;0 -0.01 0.03];
            opts = capture_fit_sim_options(0.05);
            opts.sample_rate = 360;
            opts.joint_names = schema.coordinate_names';
            opts.retain_raw_output = true;
            opts.verbosity = "Silent";
            time = [0;0.007;0.05];
            theta = zeros(189,1);
            theta(7*find(schema.coordinate_names=="LFInput")) = 0.2;
            [actual,replay] = simulate_golf_markers(theta,opts,ks,schema, ...
                bodies,offsets,time);
            testCase.verifySize(actual,[3 2 3]);
            for m = 1:2
                names = ["LSLogs" "RSLogs"];
                logs = replay.raw_output.CombinedSignalBus.(names(m));
                nativeTime = logs.GlobalPosition.Time(:);
                origin = resample_logged_signal(logs.GlobalPosition,nativeTime,3,[]);
                rotations = resample_logged_signal(logs.Rotation_Transform,nativeTime,9,[]);
                projected = origin;
                for k = 1:numel(nativeTime)
                    projected(k,:) = origin(k,:) + offsets(m,:)*reshape(rotations(k,:),3,3)';
                end
                expected = resample_logged_signal(projected,time,3,nativeTime);
                testCase.verifyEqual(squeeze(actual(:,m,:)),expected,'AbsTol',1e-8);
            end
        end
    end
end
