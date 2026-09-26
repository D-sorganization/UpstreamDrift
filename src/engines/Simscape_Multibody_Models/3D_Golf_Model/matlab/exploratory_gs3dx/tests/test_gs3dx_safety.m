classdef test_gs3dx_safety < matlab.unittest.TestCase
%TEST_GS3DX_SAFETY  Guards for epic #10950: the originals stay untouched and names never collide.
%
%   Run:  results = runtests('test_gs3dx_safety')   (from exploratory_gs3dx/tests)

    properties
        info struct
    end

    methods (TestClassSetup)
        function setup(testCase)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root);
            testCase.info = gs3dx_setup();
        end
    end

    methods (Test)
        function originals_match_git_head(testCase)
            % The originals must be byte-identical to the committed blobs.
            m = gs3dx_original_manifest(testCase.info);
            for k = 1:numel(m)
                [status, out] = system(sprintf('git hash-object "%s"', m(k).file));
                testCase.assumeEqual(status, 0, 'git not available');
                [~, rel] = fileparts(m(k).file);
                [s2, head] = system(sprintf('git -C "%s" rev-parse HEAD:./%s.slx', ...
                    testCase.info.original_model_dir, rel));
                testCase.assumeEqual(s2, 0, 'cannot resolve HEAD blob');
                testCase.verifyEqual(strtrim(out), strtrim(head), ...
                    sprintf('Original %s differs from git HEAD', m(k).file));
            end
        end

        function no_shadowing(testCase)
            testCase.verifyWarningFree(@() gs3dx_assert_no_shadowing(testCase.info));
        end

        function every_model_file_is_prefixed(testCase)
            names = gs3dx_names();
            files = dir(fullfile(testCase.info.models_dir, '*.slx'));
            for k = 1:numel(files)
                testCase.verifyTrue(startsWith(files(k).name, names.prefix), files(k).name);
            end
        end

        function save_guard_rejects_unprefixed_names(testCase)
            testCase.verifyError(@() gs3dx_save_model('NotPrefixed', testCase.info), ...
                'gs3dx:unsafeSave');
        end

        function clones_never_reference_originals(testCase)
            names = gs3dx_names();
            originals = string(values(names.original_subsys));
            files = dir(fullfile(testCase.info.models_dir, 'GS3DX_*.slx'));
            for k = 1:numel(files)
                [~, mdl] = fileparts(files(k).name);
                load_system(mdl);
                c = onCleanup(@() close_system(mdl, 0));
                refs = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
                    'MatchFilter', @Simulink.match.allVariants, 'BlockType', 'SubSystem');
                for b = 1:numel(refs)
                    r = string(get_param(refs{b}, 'ReferencedSubsystem'));
                    testCase.verifyFalse(any(r == originals), ...
                        sprintf('%s references original %s', refs{b}, r));
                end
                clear c
            end
        end
    end
end
