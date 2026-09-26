classdef test_gs3dx_harness < matlab.unittest.TestCase
%TEST_GS3DX_HARNESS  Regression-harness tests (#10952).
%
%   Unit tests exercise gs3dx_flatten_bus and gs3dx_compare on synthetic
%   timeseries.  The 'Simulation' tagged test runs GS3DX_Baseline and
%   compares it with the recorded run of the hand-built original.
%
%   Run:  runtests('tests')                         (everything, ~2 min)
%         runtests('tests', 'ExcludeTag', 'Simulation')   (fast subset)

    properties
        info struct
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
        end
    end

    methods (Test)
        function flatten_walks_nested_bus_onto_grid(testCase)
            t = (0:0.1:1).';
            bus.A.pos = timeseries([t, 2*t], t);
            bus.B     = timeseries(t.^2, t);
            flat = gs3dx_flatten_bus(bus, (0:0.5:1).');
            testCase.verifyEqual(flat.names, ["A/pos", "B"]);
            testCase.verifyEqual(flat.data{1}, [0 0; 0.5 1; 1 2], 'AbsTol', 1e-12);
            testCase.verifyEqual(flat.data{2}, [0; 0.25; 1], 'AbsTol', 1e-12);
        end

        function flatten_skips_scalar_time_leaves(testCase)
            bus.const = timeseries(5, 0);
            bus.sig   = timeseries((0:2).', (0:2).');
            flat = gs3dx_flatten_bus(bus, (0:2).');
            testCase.verifyEqual(flat.names, "sig");
        end

        function flatten_reshapes_channel_major_data(testCase)
            t = (0:2).';
            ts = timeseries(reshape([t.'; 10*t.'], 2, 1, 3), t);   % (2,1,N) layout
            flat = gs3dx_flatten_bus(struct('R', ts), t);
            testCase.verifyEqual(flat.data{1}, [t, 10*t], 'AbsTol', 1e-12);
        end

        function flatten_keeps_last_sample_at_repeated_time(testCase)
            ts = timeseries([0; 1; 3; 4], [0; 1; 1; 2]);           % zero-crossing repeat
            flat = gs3dx_flatten_bus(struct('x', ts), [0; 1; 2]);
            testCase.verifyEqual(flat.data{1}, [0; 3; 4], 'AbsTol', 1e-12);
        end

        function compare_identical_runs_pass(testCase)
            ref = local_flat(["q/CHGlobalPosition", "q/MPGlobalPosition", "q/x"]);
            cmp = gs3dx_compare(ref, ref);
            testCase.verifyTrue(cmp.pass);
            testCase.verifyTrue(cmp.key_pass);
            testCase.verifyEmpty(cmp.missing);
        end

        function compare_flags_perturbation_beyond_tolerance(testCase)
            ref = local_flat(["q/CHGlobalPosition", "q/MPGlobalPosition", "q/x"]);
            test = ref;
            test.data{3} = test.data{3} + 0.01;                    % range 1, rtol 1e-3
            cmp = gs3dx_compare(ref, test);
            testCase.verifyTrue(cmp.key_pass);
            testCase.verifyFalse(cmp.pass);
            testCase.verifyEqual(cmp.table.name(~cmp.table.pass), "q/x");
        end

        function compare_accepts_perturbation_within_tolerance(testCase)
            ref = local_flat(["q/CHGlobalPosition", "q/MPGlobalPosition"]);
            test = ref;
            test.data{1} = test.data{1} + 5e-4;
            testCase.verifyTrue(gs3dx_compare(ref, test).pass);
        end

        function compare_fails_when_key_signal_missing(testCase)
            ref  = local_flat(["q/CHGlobalPosition", "q/MPGlobalPosition"]);
            test = local_flat("q/CHGlobalPosition");
            cmp = gs3dx_compare(ref, test);
            testCase.verifyFalse(cmp.key_pass);
            testCase.verifyEqual(cmp.missing, "q/MPGlobalPosition");
        end

        function compare_rejects_different_time_grids(testCase)
            ref = local_flat("q/CHGlobalPosition");
            test = ref;
            test.time = test.time + 1;
            testCase.verifyError(@() gs3dx_compare(ref, test), 'gs3dx:compare');
        end

        function baseline_is_stored_in_double_precision(testCase)
            b = local_load_baseline(testCase.info);
            testCase.verifyTrue(all(cellfun(@(d) isa(d, 'double'), b.flat.data)));
            manifest = gs3dx_original_manifest(testCase.info);
            testCase.verifyEqual(b.source_sha256, manifest(1).sha256);
        end
    end

    methods (Test, TestTags = {'Simulation'})
        function clone_reproduces_original(testCase)
            names = gs3dx_names();
            ref = local_load_baseline(testCase.info);
            run = gs3dx_simulate(char(names.variants.baseline));
            testCase.assertEqual(run.status, "success", run.message);
            cmp = gs3dx_compare(ref, run);
            testCase.verifyEmpty(cmp.missing);
            testCase.verifyEmpty(cmp.extra);
            testCase.verifyTrue(cmp.pass, ...
                strjoin(cmp.table.name(~cmp.table.pass), ', '));
            testCase.verifyEqual(run.n_steps, ref.n_steps);
        end
    end
end

function flat = local_flat(names)
    t = linspace(0, 1, 11).';
    flat = struct('time', t, 'names', names, 'data', {cell(1, numel(names))});
    for k = 1:numel(names)
        flat.data{k} = t;                                          % range exactly 1
    end
end

function b = local_load_baseline(info)
    names = gs3dx_names();
    S = load(fullfile(info.baselines_dir, sprintf('original_%s_0p3S.mat', names.original_model)));
    b = S.baseline;
end
