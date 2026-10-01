classdef test_gs3dx_slim < matlab.unittest.TestCase
%TEST_GS3DX_SLIM  GS3DX_Slim: direct joint torque drive (#10954).
%
%   Each torque axis of the kinetically driven joints is driven by its
%   Simulink-PS converter straight into the joint's InputTorque port instead
%   of Ideal Torque Source -> Rotational Multibody Interface (+ Reference).
%   The slim model must keep every logged signal and match the original run
%   on the impact regression drive (GS3DX_DRIVE).

    properties
        info struct
        names struct
    end

    properties (Constant)
        % Removed blocks per torque axis: source, interface, reference.
        BlocksPerAxis = 3
        Axes = struct('Gimbal', 3, 'Revolute', 1, 'Universal', 2)
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info  = gs3dx_setup();
            testCase.names = gs3dx_names();
            slim = char(testCase.names.variants.slim);
            if ~isfile(fullfile(testCase.info.models_dir, [slim '.slx']))
                gs3dx_build_slim(testCase.info);
            end
        end
    end

    methods (Test)
        function slim_subsystems_have_no_one_d_drive_chain(testCase)
            for r = cellstr(testCase.names.roles)
                mdl = testCase.names.slim_subsys(r{1});
                load_system(mdl);
                for ref = ["fl_lib/Mechanical/Mechanical Sources/Ideal Torque Source", ...
                           "fl_lib/Mechanical/Multibody Interfaces/Rotational Multibody Interface", ...
                           "fl_lib/Mechanical/Rotational Elements/Mechanical Rotational Reference"]
                    testCase.verifyEmpty(find_system(mdl, 'ReferenceBlock', char(ref)), ...
                        sprintf('%s still contains %s', mdl, ref));
                end
            end
        end

        function slim_subsystems_save_exactly_three_blocks_per_axis(testCase)
            for r = cellstr(testCase.names.roles)
                before = gs3dx_block_budget(testCase.names.clone_subsys(r{1}));
                after  = gs3dx_block_budget(testCase.names.slim_subsys(r{1}));
                testCase.verifyEqual(before.nonvirtual_total - after.nonvirtual_total, ...
                    testCase.BlocksPerAxis * testCase.Axes.(r{1}), r{1});
            end
        end

        function slim_model_references_only_slim_subsystems(testCase)
            slim = char(testCase.names.variants.slim);
            load_system(slim);
            refs = get_param(find_system(slim, 'LookUnderMasks', 'all', ...
                'MatchFilter', @Simulink.match.allVariants, 'BlockType', 'SubSystem'), ...
                'ReferencedSubsystem');
            refs = refs(~cellfun(@isempty, refs));
            testCase.verifyNotEmpty(refs);
            testCase.verifyTrue(all(ismember(refs, values(testCase.names.slim_subsys))), ...
                strjoin(unique(refs), ', '));
        end

        function slim_model_saves_sixty_three_blocks(testCase)
            base = local_fresh_budget(char(testCase.names.variants.baseline));
            slim = local_fresh_budget(char(testCase.names.variants.slim));
            testCase.verifyEqual(base.nonvirtual_total - slim.nonvirtual_total, 63);
        end

        function builder_refuses_to_overwrite(testCase)
            testCase.verifyError(@() gs3dx_build_slim(testCase.info), 'gs3dx:slim');
        end
    end

    methods (Test, TestTags = {'Simulation'})
        function slim_reproduces_original(testCase)
            % Impact drive only: the persisted drive amplifies rounding noise
            % (docs/SENSITIVITY_FINDINGS.md), so no restructured model can
            % match it, while the unchanged clone does bit for bit.
            slim = char(testCase.names.variants.slim);
            S = load(gs3dx_baseline_file(testCase.info, "impact"));
            run = gs3dx_simulate(slim, variables = gs3dx_drive(testCase.info, "impact", slim));
            testCase.assertEqual(run.status, "success", run.message);
            cmp = gs3dx_compare(S.baseline, run);
            testCase.verifyEmpty(cmp.missing);
            testCase.verifyTrue(cmp.pass, strjoin(cmp.table.name(~cmp.table.pass), ', '));
            fprintf('GS3DX_Slim: %d steps (baseline %d), %.1f s wall\n', ...
                run.n_steps, S.baseline.n_steps, run.wall_s);
        end
    end
end

function report = local_fresh_budget(mdl)
% Count from a fresh load: after a simulation in the same session the
% count of the compiled model differs (+101 blocks seen on GS3DX_Slim).
    if bdIsLoaded(mdl)
        close_system(mdl, 0);
    end
    report = gs3dx_block_budget(mdl);
end
