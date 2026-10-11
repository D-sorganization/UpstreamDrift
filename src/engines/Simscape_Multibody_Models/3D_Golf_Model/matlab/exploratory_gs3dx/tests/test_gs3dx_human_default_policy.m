classdef test_gs3dx_human_default_policy < matlab.unittest.TestCase
% Native Human-default API seam; optional local capture fixture stays private.
    properties
        jc struct
        mdl char
    end
    methods (TestClassSetup)
        function setup(testCase)
            root=fileparts(fileparts(mfilename('fullpath')));
            addpath(root); addpath(fullfile(root,'tools')); gs3dx_setup();
            names=gs3dx_names(); testCase.mdl=char(names.variants.human);
            testCase.assumeTrue(~isempty(which([testCase.mdl '.slx'])), ...
                'Native Human model is required for this integration test.');
            fixture=getenv('GS3DX_POLICY_CAPTURE');
            if ~isempty(fixture)
                testCase.assumeTrue(isfile(fixture),'Configured local capture fixture is unavailable.');
            end
            % Parser errors remain errors; a malformed capture is not a skip.
            cap=gs3dx_capture_markers(fixture);
            testCase.jc=gs3dx_capture_joint_centres(cap);
            names=[string(gs3dx_names().variants.fit),string(testCase.mdl)];
            for name=names
                mdl=char(name);
                if ~bdIsLoaded(mdl) && ~isempty(which([mdl '.slx']))
                    load_system(mdl);
                    testCase.addTeardown(@() close_system(mdl,0));
                end
            end
        end
    end
    methods (Test)
        function omitted_model_matches_explicit_human(testCase)
            actual=gs3dx_whole_body_ik(testCase.jc,frames=1,calibration_frames=1);
            expected=gs3dx_whole_body_ik(testCase.jc,model=testCase.mdl, ...
                frames=1,calibration_frames=1);
            testCase.verifyEqual(actual.model,testCase.mdl);
            testCase.verifyTrue(isequaln(actual,expected), ...
                'Omitted-model and explicit Human outputs must match fully.');
            testCase.verifyEqual(actual.status,1);
            [keys,ids]=gs3dx_joint_keys(actual.model);
            [found,at]=ismember(string(actual.joint_ids),ids);
            testCase.assertTrue(all(found));
            keys=keys(at);
            testCase.verifyTrue(any(contains(keys,'Neck Joint|Rx.q')));
            testCase.verifyTrue(any(contains(keys,'Neck Joint|Ry.q')));
            testCase.verifyTrue(any(contains(keys,'L Midfoot Joint|Rz.q')));
            testCase.verifyTrue(any(contains(keys,'R Midfoot Joint|Rz.q')));
        end
        function keyed_seed_uses_actual_native_units(testCase)
            first=gs3dx_whole_body_ik(testCase.jc,model=testCase.mdl, ...
                frames=1,calibration_frames=1);
            ks=simscape.multibody.KinematicsSolver(testCase.mdl);
            jp=ks.jointPositionVariables;
            [keys,ids]=gs3dx_joint_keys(testCase.mdl,jp);
            [found,at]=ismember(string(first.joint_ids),ids);
            testCase.assertTrue(all(found));
            seed=struct('joint_keys',keys(at),'joint',first.joint(:,1), ...
                'units',string(jp.Unit(at)),'status',first.status(1));
            delete(ks);
            actual=gs3dx_whole_body_ik(testCase.jc,frames=1, ...
                calibration_frames=1,offsets=first.offsets,initial_pose=seed);
            expected=gs3dx_whole_body_ik(testCase.jc,model=testCase.mdl, ...
                frames=1,calibration_frames=1,offsets=first.offsets,initial_pose=seed);
            testCase.verifyEqual(actual.model,testCase.mdl);
            testCase.verifyEqual(actual.seed_source,'initial_pose');
            testCase.verifyTrue(isequaln(actual,expected));
            testCase.verifyEqual(actual.status,1);
        end
    end
end
