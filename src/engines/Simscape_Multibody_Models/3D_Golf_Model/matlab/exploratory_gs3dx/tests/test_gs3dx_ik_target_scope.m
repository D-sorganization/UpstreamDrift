classdef test_gs3dx_ik_target_scope < matlab.unittest.TestCase
    methods (TestClassSetup)
        function setupPath(tc)
            here = fileparts(mfilename('fullpath'));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(here));
            root = fileparts(here);
            tools_dir = fullfile(root, 'tools');
            if isfolder(tools_dir)
                tc.applyFixture(matlab.unittest.fixtures.PathFixture(tools_dir));
            end
        end
    end

    methods (Test)
        function testAutoScopeFullBody(tc)
            jp = makeFullBodyJP();
            scope = gs3dx_ik_target_scope(jp);
            tc.verifyEqual(scope.name, "whole_body");
            tc.verifyEqual(scope.target_indices, 1:14);

            % Explicit auto string and char
            scope_str = gs3dx_ik_target_scope(jp, "auto");
            tc.verifyEqual(scope_str.name, "whole_body");
            tc.verifyEqual(scope_str.target_indices, 1:14);

            scope_char = gs3dx_ik_target_scope(jp, 'auto');
            tc.verifyEqual(scope_char.name, "whole_body");
            tc.verifyEqual(scope_char.target_indices, 1:14);
        end

        function testAutoScopeUpperBody(tc)
            jp = makeUpperBodyJP();
            scope = gs3dx_ik_target_scope(jp);
            tc.verifyEqual(scope.name, "upper_body");
            tc.verifyEqual(scope.target_indices, [1, 8:14]);

            scope_auto = gs3dx_ik_target_scope(jp, "auto");
            tc.verifyEqual(scope_auto.name, "upper_body");
            tc.verifyEqual(scope_auto.target_indices, [1, 8:14]);
        end

        function testExplicitWholeBody(tc)
            jp_full = makeFullBodyJP();
            scope = gs3dx_ik_target_scope(jp_full, "whole_body");
            tc.verifyEqual(scope.name, "whole_body");
            tc.verifyEqual(scope.target_indices, 1:14);

            % Requesting whole_body on upper-only model must error
            jp_upper = makeUpperBodyJP();
            tc.verifyError(@() gs3dx_ik_target_scope(jp_upper, "whole_body"), 'gs3dx:ik:scope');
        end

        function testExplicitUpperBody(tc)
            % Permitted on upper-only model
            jp_upper = makeUpperBodyJP();
            scope_upper = gs3dx_ik_target_scope(jp_upper, "upper_body");
            tc.verifyEqual(scope_upper.name, "upper_body");
            tc.verifyEqual(scope_upper.target_indices, [1, 8:14]);

            % Permitted on full model
            jp_full = makeFullBodyJP();
            scope_full = gs3dx_ik_target_scope(jp_full, "upper_body");
            tc.verifyEqual(scope_full.name, "upper_body");
            tc.verifyEqual(scope_full.target_indices, [1, 8:14]);
        end

        function testPartialLowerBodyErrors(tc)
            % 3 out of 6 roles present
            paths = [ ...
                "Model/Hips and Torso Inputs/Torso Kinetically Driven/Joint"; ...
                "Model/Lower Body/Left Hip Joint/Kinetically Driven"; ...
                "Model/Lower Body/Right Hip Joint/Kinetically Driven"; ...
                "Model/Lower Body/Left Knee/Kinetically Driven" ...
            ];
            jp_partial = table(paths, 'VariableNames', {'BlockPath'});

            tc.verifyError(@() gs3dx_ik_target_scope(jp_partial, "auto"), 'gs3dx:ik:scope');
            tc.verifyError(@() gs3dx_ik_target_scope(jp_partial, "whole_body"), 'gs3dx:ik:scope');
            tc.verifyError(@() gs3dx_ik_target_scope(jp_partial, "upper_body"), 'gs3dx:ik:scope');
        end

        function testDuplicateAmbiguousTopologyErrors(tc)
            % Two distinct block paths matching 'Left Hip Joint'
            paths = [ ...
                "Model/Lower Body/Left Hip Joint/BranchA"; ...
                "Model/Lower Body/Left Hip Joint/BranchB"; ...
                "Model/Lower Body/Right Hip Joint/Joint"; ...
                "Model/Lower Body/Left Knee/Joint"; ...
                "Model/Lower Body/Right Knee/Joint"; ...
                "Model/Lower Body/Left Ankle/Joint"; ...
                "Model/Lower Body/Right Ankle/Joint" ...
            ];
            jp_ambiguous = table(paths, 'VariableNames', {'BlockPath'});

            tc.verifyError(@() gs3dx_ik_target_scope(jp_ambiguous, "auto"), 'gs3dx:ik:scope');
            tc.verifyError(@() gs3dx_ik_target_scope(jp_ambiguous, "whole_body"), 'gs3dx:ik:scope');
            tc.verifyError(@() gs3dx_ik_target_scope(jp_ambiguous, "upper_body"), 'gs3dx:ik:scope');
        end

        function testRepeatedPathsDoNotFailIfIdenticalLeaf(tc)
            % Multiple entries for the same joint DOF (identical BlockPath) should succeed
            paths = [ ...
                "Model/Lower Body/Left Hip Joint/Kinetically Driven"; ...
                "Model/Lower Body/Left Hip Joint/Kinetically Driven"; ...
                "Model/Lower Body/Right Hip Joint/Kinetically Driven"; ...
                "Model/Lower Body/Left Knee/Kinetically Driven"; ...
                "Model/Lower Body/Right Knee/Kinetically Driven"; ...
                "Model/Lower Body/Left Ankle/Kinetically Driven"; ...
                "Model/Lower Body/Right Ankle/Kinetically Driven" ...
            ];
            jp_multidof = table(paths, 'VariableNames', {'BlockPath'});
            scope = gs3dx_ik_target_scope(jp_multidof, "auto");
            tc.verifyEqual(scope.name, "whole_body");
            tc.verifyEqual(scope.target_indices, 1:14);
        end

        function testInvalidSchemaErrors(tc)
            % Not a table
            tc.verifyError(@() gs3dx_ik_target_scope("not_a_table"), 'gs3dx:ik:scope');
            tc.verifyError(@() gs3dx_ik_target_scope(struct('a', 1)), 'gs3dx:ik:scope');

            % Table missing BlockPath
            t_bad = table([1; 2], 'VariableNames', {'ID'});
            tc.verifyError(@() gs3dx_ik_target_scope(t_bad), 'gs3dx:ik:scope');
        end

        function testInvalidSelectionErrors(tc)
            jp = makeFullBodyJP();
            tc.verifyError(@() gs3dx_ik_target_scope(jp, "invalid_scope"), 'gs3dx:ik:scope');
            tc.verifyError(@() gs3dx_ik_target_scope(jp, 123), 'gs3dx:ik:scope');
            tc.verifyError(@() gs3dx_ik_target_scope(jp, ["auto", "upper_body"]), 'gs3dx:ik:scope');
        end
    end
end

function jp = makeFullBodyJP()
    BlockPath = [ ...
        "GS3DX_FullBody/Hips and Torso Inputs/Torso Kinetically Driven/Joint"; ...
        "GS3DX_FullBody/Left Shoulder Joint/Gimbal Joint/Kinetically Driven"; ...
        "GS3DX_FullBody/Right Shoulder Joint/Gimbal Joint/Kinetically Driven"; ...
        "GS3DX_FullBody/Left Elbow Joint/Revolute Joint/Kinetically Driven"; ...
        "GS3DX_FullBody/Right Elbow Joint/Revolute Joint/Kinetically Driven"; ...
        "GS3DX_FullBody/Left Wrist and Hand/Universal Joint/Kinetically Driven"; ...
        "GS3DX_FullBody/Right Wrist and Hand/Universal Joint/Kinetically Driven"; ...
        "GS3DX_FullBody/Lower Body/Left Hip Joint/Kinetically Driven"; ...
        "GS3DX_FullBody/Lower Body/Right Hip Joint/Kinetically Driven"; ...
        "GS3DX_FullBody/Lower Body/Left Knee Joint/Kinetically Driven Revolute"; ...
        "GS3DX_FullBody/Lower Body/Right Knee Joint/Kinetically Driven Revolute"; ...
        "GS3DX_FullBody/Lower Body/Left Ankle Joint/Kinetically Driven Universal Joint"; ...
        "GS3DX_FullBody/Lower Body/Right Ankle Joint/Kinetically Driven Universal Joint" ...
    ];
    jp = table(BlockPath);
end

function jp = makeUpperBodyJP()
    BlockPath = [ ...
        "GS3DX_Baseline/Hips and Torso Inputs/Torso Kinetically Driven/Joint"; ...
        "GS3DX_Baseline/Left Shoulder Joint/Gimbal Joint/Kinetically Driven"; ...
        "GS3DX_Baseline/Right Shoulder Joint/Gimbal Joint/Kinetically Driven"; ...
        "GS3DX_Baseline/Left Elbow Joint/Revolute Joint/Kinetically Driven"; ...
        "GS3DX_Baseline/Right Elbow Joint/Revolute Joint/Kinetically Driven"; ...
        "GS3DX_Baseline/Left Wrist and Hand/Universal Joint/Kinetically Driven"; ...
        "GS3DX_Baseline/Right Wrist and Hand/Universal Joint/Kinetically Driven" ...
    ];
    jp = table(BlockPath);
end
