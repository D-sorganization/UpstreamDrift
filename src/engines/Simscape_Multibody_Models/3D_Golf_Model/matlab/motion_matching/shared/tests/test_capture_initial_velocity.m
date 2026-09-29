classdef test_capture_initial_velocity < matlab.unittest.TestCase
    methods(Test)
        function honorsIndependentRatesAndRestoresPriorities(testCase)
            load_system('GolfSwing3D_Kinetic');
            testCase.addTeardown(@()close_system('GolfSwing3D_Kinetic',0));
            shared=fileparts(which('build_golf_kinematics'));
            schema=jsondecode(fileread(fullfile(shared,'golf_kinematic_schema.json')));
            repo=shared;
            while ~isfile(fullfile(repo,'pyproject.toml'))
                parent=fileparts(repo);testCase.assertNotEqual(parent,repo);repo=parent;
            end
            seed=jsondecode(fileread(fullfile(repo,'docs/development/simscape_tour_matching/native_evidence/initial_velocity_seed_unqualified_r2025b.json')));
            names=string({schema.coordinates.name});
            torso=fileparts(schema.coordinates(names=="TorsoInput").block_path);
            wrist=fileparts(schema.coordinates(names=="LWInputY").block_path);
            previousTorso=get_param(torso,'RzVelocityTargetPriority');previousWrist=get_param(wrist,'RyVelocityTargetPriority');
            torsoJoint=schema.coordinates(names=="TorsoInput").block_path;wristJoint=schema.coordinates(names=="LWInputY").block_path;
            previousTorsoSpecify=get_param(torsoJoint,'VelocityTargetSpecify');previousWristSpecify=get_param(wristJoint,'RyVelocityTargetSpecify');
            opts=capture_fit_sim_options(.02);
            opts.verbosity='Silent';opts.retain_raw_output=true;opts.joint_names=string(seed.coordinate_names)';
            opts.input_overrides.UpperArmLength=seed.geometry_in(1);opts.input_overrides.LowerArmLength=seed.geometry_in(2);
            for j=1:numel(seed.q)
                name=opts.joint_names(j);position=seed.q(j);velocity=seed.qd(j);
                if ~startsWith(name,'Translation');position=rad2deg(position);velocity=rad2deg(velocity);end
                opts.input_overrides.(replace(name,'Input','StartPosition'))=position;
                opts.input_overrides.(replace(name,'Input','StartVelocity'))=velocity;
            end
            priority_cleanup=configure_capture_velocity_targets();
            for restart=[false true true false]
                opts.fast_restart=restart;
                replay=simulate_with_coefficients(zeros(189,1),opts);
                testCase.verifyEqual(replay.q(1,:),seed.q','AbsTol',1e-8);
                testCase.verifyEqual(replay.qd(1,:),seed.qd','AbsTol',1e-8);
                testCase.verifyEqual(get_param(torso,'RzVelocityTargetPriority'),'High');
                testCase.verifyEqual(get_param(wrist,'RyVelocityTargetPriority'),'High');
            end
            clear priority_cleanup;
            testCase.verifyEqual(get_param(torso,'RzVelocityTargetPriority'),previousTorso);
            testCase.verifyEqual(get_param(wrist,'RyVelocityTargetPriority'),previousWrist);
            testCase.verifyEqual(get_param(torsoJoint,'VelocityTargetSpecify'),previousTorsoSpecify);
            testCase.verifyEqual(get_param(wristJoint,'RyVelocityTargetSpecify'),previousWristSpecify);
            testCase.verifyEqual(get_param('GolfSwing3D_Kinetic','FastRestart'),'off');
        end
        function restoresAfterFailure(testCase)
            load_system('GolfSwing3D_Kinetic');
            testCase.addTeardown(@()close_system('GolfSwing3D_Kinetic',0));
            shared=fileparts(which('build_golf_kinematics'));
            schema=jsondecode(fileread(fullfile(shared,'golf_kinematic_schema.json')));
            names=string({schema.coordinates.name});
            joint=schema.coordinates(names=="TorsoInput").block_path;mask=get_param(joint,'Parent');
            before=get_param(mask,'RzVelocityTargetPriority');specify=get_param(joint,'VelocityTargetSpecify');
            testCase.verifyError(@()testCase.failInsideSession(),'capture_velocity_test:expected');
            testCase.verifyEqual(get_param(mask,'RzVelocityTargetPriority'),before);
            testCase.verifyEqual(get_param(joint,'VelocityTargetSpecify'),specify);
        end
    end
    methods(Access=private)
        function failInsideSession(~)
            cleanup=configure_capture_velocity_targets(); %#ok<NASGU>
            error('capture_velocity_test:expected','Exercise failure cleanup.');
        end
    end
end
