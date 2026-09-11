classdef test_fit_golf_pose_seed < matlab.unittest.TestCase
    methods(Test)
        function translatedNativePose(testCase)
            [ks,schema,q]=testCase.nativeFixture();
            addTargetVariables(ks,schema.q_ids); addOutputVariables(ks,schema.frame_ids);
            [values,status,targets]=solve(ks,q);
            testCase.assertEqual(status,1); testCase.assertTrue(all(targets));
            points=reshape(values,3,[])'+[0.02,0,0];
            result=fit_golf_pose_seed(ks,schema,q,string({schema.frames.name})',points);
            testCase.verifyEqual(result.solver_flag,1);
            testCase.verifyTrue(all(result.target_flags));
            testCase.verifyLessThan(result.proxy_euclidean_rms_m,1e-4);
            testCase.verifyGreaterThan(norm(result.q-q),0.01);
        end
        function rotatedAttachedMarkers(testCase)
            [ks,schema,q]=testCase.nativeFixture();
            target=q; target(schema.coordinate_names=="HipInputZ")= ...
                target(schema.coordinate_names=="HipInputZ")+0.06;
            addTargetVariables(ks,schema.q_ids);
            addOutputVariables(ks,schema.frame_ids);
            addOutputVariables(ks,schema.rotation_ids);
            [values,status,targets]=solve(ks,target);
            testCase.assertEqual(status,1); testCase.assertTrue(all(targets));
            n=numel(schema.frame_ids); origins=reshape(values(1:n),3,[])';
            rotations=intrinsic_xyz_to_rotm(reshape(values(n+1:end),3,[])');
            % Three noncollinear attachments per frame constrain orientation.
            bodies=repelem((1:numel(schema.frames))',3);
            offsets=repmat([0.11,0,0;0,0.09,0;0,0,0.07],numel(schema.frames),1);
            points=project_body_markers(origins,rotations,bodies,offsets);
            names=string({schema.frames.name})';
            result=fit_golf_pose_seed(ks,schema,q,names(bodies),points,offsets);
            testCase.verifyLessThan(result.proxy_euclidean_rms_m,1e-4);
            testCase.verifyEqual(result.marker_positions_m-points,result.proxy_errors_m,'AbsTol',1e-12);
            testCase.verifyEqual(result.qualification,'fixed-attachment-pose-seed-only');
            testCase.verifyEqual(result.solver_flag,1);
            testCase.verifyTrue(all(result.target_flags));
        end
        function rejectsAttachmentCount(testCase)
            [ks,schema,q]=testCase.nativeFixture();
            testCase.verifyError(@()fit_golf_pose_seed(ks,schema,q,"Hip",zeros(1,3),zeros(2,3)), ...
                'fit_golf_pose_seed:attachments');
        end
    end
    methods(Access=private)
        function [ks,schema,q]=nativeFixture(testCase)
            load_system('GolfSwing3D_Kinetic');
            [ks,schema]=build_golf_kinematics();
            repo=fileparts(which('build_golf_kinematics'));
            while ~isfile(fullfile(repo,'pyproject.toml'))
                parent=fileparts(repo); testCase.assertNotEqual(parent,repo); repo=parent;
            end
            fixture=jsondecode(fileread(fullfile(repo,'docs/development/simscape_tour_matching/native_evidence/reproduction/cold_candidate_input.json')));
            [found,order]=ismember(schema.coordinate_names,string(fixture.seed.coordinate_names));
            testCase.assertTrue(all(found)); q=fixture.seed.q(order);
        end
    end
end

