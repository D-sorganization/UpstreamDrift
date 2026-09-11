classdef test_golf_marker_velocity_map < matlab.unittest.TestCase
    methods(Test)
        function agreesWithNativePoseDifferencing(testCase)
            load_system('GolfSwing3D_Kinetic');[ks,schema]=build_golf_kinematics();
            repo=fileparts(which('build_golf_kinematics'));
            while ~isfile(fullfile(repo,'pyproject.toml'))
                parent=fileparts(repo);testCase.assertNotEqual(parent,repo);repo=parent;
            end
            fixture=jsondecode(fileread(fullfile(repo,'docs/development/simscape_tour_matching/native_evidence/reproduction/cold_candidate_input.json')));
            [found,order]=ismember(schema.coordinate_names,string(fixture.seed.coordinate_names));testCase.assertTrue(all(found));q=fixture.seed.q(order);
            [~,bodies]=ismember(["LS";"RS";"Clubhead"],string({schema.frames.name}));
            offsets=[.02 .01 -.03;-.01 .03 .02;.1 -.02 .04];
            result=golf_marker_velocity_map(ks,schema,q,bodies,offsets);
            testCase.verifySize(result.marker_jacobian,[9 21]);
            independent=ismember(schema.coordinate_names,result.independent_names);
            testCase.verifyEqual(result.joint_velocity_map(independent,:),eye(21),'AbsTol',1e-8);
            clearTargetVariables(ks);clearOutputVariables(ks);clearInitialGuessVariables(ks);
            addTargetVariables(ks,schema.q_ids(independent));addInitialGuessVariables(ks,schema.q_ids(~independent));
            addOutputVariables(ks,schema.frame_ids);addOutputVariables(ks,schema.rotation_ids);
            column=find(result.independent_names=="HipInputX");h=1e-5;
            samples=zeros(9,2);
            for k=1:2
                x=q(independent);x(column)=x(column)+(2*k-3)*h;
                [values,flag,targets]=solve(ks,x,q(~independent));testCase.assertEqual(flag,1);testCase.assertTrue(all(targets));
                n=numel(schema.frames);origins=reshape(values(1:3*n),3,[])';
                rotations=intrinsic_xyz_to_rotm(reshape(values(3*n+1:end),3,[])');
                markers=project_body_markers(origins,rotations,bodies,offsets);samples(:,k)=reshape(markers',[],1);
            end
            testCase.verifyEqual(result.marker_jacobian(:,column),(samples(:,2)-samples(:,1))/(2*h),'AbsTol',1e-4);
        end
    end
end
