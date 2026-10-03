classdef test_gs3dx_inertia < matlab.unittest.TestCase
%TEST_GS3DX_INERTIA  Inertia audit and anthropometric segment inertia tests (#10979).
%
%   Verifies:
%     1. de Leva radii of gyration table values and (0,1) fraction bounds.
%     2. gs3dx_segment_inertia against hand-computed cases and input guards.
%     3. gs3dx_inertia_audit reproduces the analytic uniform-cylinder formula
%        for a known cylinder solid (Lower Body/L Thigh) to 1e-12 relative.
%     4. gs3dx_inertia_audit total mass equals documented sensed mass
%        (80.393 kg) to 1 g.
%     5. Parallel-axis theorem lumping for multi-piece segments (forearm halves
%        and UpperTorsoBase + UpperTorsoTop).
%     6. Segment audit table structure, required segments, and positive finite ratios.

    properties
        info struct
        mdl char
    end

    properties (Constant)
        % Equipment mass: clubhead 0.25 kg, shaft 0.0802566 kg, grip 0.036667 kg,
        % hand standoffs 0.020 kg, contact spheres 0.006 kg = 0.3929236 kg.
        EquipmentMass = 0.25 + 0.0802566 + 0.036667 + 0.020 + 0.006
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
            testCase.mdl = char(gs3dx_names().variants.fit_balance);
            load_system(testCase.mdl);
            testCase.addTeardown(@() close_system(testCase.mdl, 0));
        end
    end

    methods (Test)
        function test_gyration_table(testCase)
            % Verify de Leva (1996) Table 4 male values and fractions in (0,1)
            a = gs3dx_anthropometry(80);
            testCase.assertTrue(isfield(a, 'gyration'), 'gyration field present');
            testCase.assertTrue(isfield(a, 'length'), 'length field present');

            expected_gyration = struct( ...
                'head',         [0.362 0.376 0.312], ...
                'trunk',        [0.372 0.347 0.191], ...
                'upper_trunk',  [0.505 0.320 0.465], ...
                'middle_trunk', [0.482 0.383 0.468], ...
                'lower_trunk',  [0.615 0.551 0.587], ...
                'upper_arm',    [0.285 0.269 0.158], ...
                'forearm',      [0.276 0.265 0.121], ...
                'hand',         [0.628 0.513 0.401], ...
                'thigh',        [0.329 0.329 0.149], ...
                'shank',        [0.255 0.249 0.103], ...
                'foot',         [0.257 0.245 0.124]);

            for fn = fieldnames(expected_gyration).'
                k = fn{1};
                testCase.assertTrue(isfield(a.gyration, k), sprintf('Key %s in gyration', k));
                val = a.gyration.(k);
                exp_val = expected_gyration.(k);
                testCase.verifyEqual(val, exp_val, 'RelTol', 1e-12, sprintf('%s gyration values', k));
                testCase.verifyTrue(all(val > 0) && all(val < 1), sprintf('%s all fractions in (0,1)', k));
            end

            % Check reference lengths in meters
            expected_length = struct( ...
                'head', 0.2033, 'trunk', 0.5319, 'upper_arm', 0.2817, 'forearm', 0.2689, ...
                'hand', 0.0862, 'thigh', 0.4222, 'shank', 0.4395, 'foot', 0.2581);
            for fn = fieldnames(expected_length).'
                k = fn{1};
                testCase.assertTrue(isfield(a.length, k), sprintf('Key %s in length', k));
                testCase.verifyEqual(a.length.(k), expected_length.(k), 'RelTol', 1e-12, sprintf('%s length', k));
            end
        end

        function test_segment_inertia_calculation(testCase)
            % Test gs3dx_segment_inertia on hand-computed cases
            % Case 1: mass=2.0 kg, len=0.5 m, com_frac=0.4, gyration=[0.3 0.2 0.1]
            % c = 0.4 * 0.5 = 0.2 m
            % I = 2.0 * ([0.3 0.2 0.1] * 0.5).^2 = 2.0 * [0.0225 0.0100 0.0025] = [0.045 0.020 0.005] kg*m^2
            [m1, c1, I1] = gs3dx_segment_inertia(2.0, 0.5, 0.4, [0.3 0.2 0.1]);
            testCase.verifyEqual(m1, 2.0, 'AbsTol', 1e-15);
            testCase.verifyEqual(c1, 0.2, 'AbsTol', 1e-15);
            testCase.verifyEqual(I1, [0.045 0.020 0.005], 'AbsTol', 1e-15);

            % Case 2: mass=10 kg, len=1.0 m, com_frac=0.5, gyration=[0.5 0.25 0.1]
            % c = 0.5 m, I = 10 * [0.25 0.0625 0.01] = [2.5 0.625 0.1]
            [m2, c2, I2] = gs3dx_segment_inertia(10, 1.0, 0.5, [0.5 0.25 0.1]);
            testCase.verifyEqual(m2, 10, 'AbsTol', 1e-15);
            testCase.verifyEqual(c2, 0.5, 'AbsTol', 1e-15);
            testCase.verifyEqual(I2, [2.5 0.625 0.1], 'AbsTol', 1e-15);

            % Verify input validation error handling
            testCase.verifyError(@() gs3dx_segment_inertia(-1, 0.5, 0.4, [0.3 0.2 0.1]), 'MATLAB:validators:mustBePositive');
            testCase.verifyError(@() gs3dx_segment_inertia(2.0, -0.5, 0.4, [0.3 0.2 0.1]), 'MATLAB:validators:mustBePositive');
            testCase.verifyError(@() gs3dx_segment_inertia(2.0, 0.5, 1.5, [0.3 0.2 0.1]), 'MATLAB:validators:mustBeInRange');
            testCase.verifyError(@() gs3dx_segment_inertia(2.0, 0.5, 0.4, [1.2 0.2 0.1]), 'MATLAB:validators:mustBeLessThan');
        end

        function test_audit_cylinder_analytic_reproduction(testCase)
            % Reproduces the analytic uniform-cylinder formula for Lower Body/L Thigh to 1e-12 relative
            audit = gs3dx_inertia_audit(testCase.mdl);
            testCase.assertTrue(istable(audit.solids), 'audit.solids is a table');

            idx = find(audit.solids.block == "Lower Body/L Thigh", 1);
            testCase.assertNotEmpty(idx, 'Lower Body/L Thigh found in audit.solids');
            r = audit.solids(idx, :);

            ws = get_param(testCase.mdl, 'ModelWorkspace');
            M = ws.getVariable('ThighMass');
            R = ws.getVariable('ThighRadius');
            L = ws.getVariable('ThighLength');

            % Measured 2026-09-28: M = 11.328 kg, R = 0.07 m, L = 0.460137 m
            % Analytic moments about cylinder center:
            % Ixx = Iyy = M/12 * (3*R^2 + L^2), Izz = 1/2 * M * R^2
            I_analytic = [M/12 * (3*R^2 + L^2), M/12 * (3*R^2 + L^2), 0.5 * M * R^2];

            testCase.verifyEqual(r.mass, M, 'RelTol', 1e-12, 'Thigh mass');
            testCase.verifyEqual(r.moments, I_analytic, 'RelTol', 1e-12, 'Thigh cylinder moments');
        end

        function test_audit_total_mass_matches_sensed_mass(testCase)
            % Audit total mass for GS3DX_FitBalance equals documented 80.393 kg to 1 g
            % Measured 2026-09-28: audit.total_mass = 80.3929236 kg
            audit = gs3dx_inertia_audit(testCase.mdl);
            total_mass = audit.total_mass;

            % ANTHROPOMETRY.md documents 80.393 kg (80 kg body + 0.393 kg equipment)
            testCase.verifyEqual(total_mass, 80.393, 'AbsTol', 1e-3, ...
                sprintf('Total mass %.6f kg equals documented 80.393 kg to 1 g', total_mass));

            % Check exact match to 80 + EquipmentMass (80.3929236 kg)
            expected_total = 80 + testCase.EquipmentMass;
            testCase.verifyEqual(total_mass, expected_total, 'RelTol', 1e-9, ...
                'Audit total mass matches sum of body mass and equipment');
        end

        function test_parallel_axis_lumping(testCase)
            % Forearm lumped moments match a single uniform cylinder of length FitLowerArmLength
            % Measured 2026-09-28: L = 0.279918 m, M = 1.296 kg, R = 0.04445 m
            audit = gs3dx_inertia_audit(testCase.mdl);
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            M_fa = ws.getVariable('GolferForearmMass');
            L_fa = ws.getVariable('FitLowerArmLength') * 0.0254;
            R_fa = 1.75 * 0.0254;

            I_fa_analytic = [M_fa/12 * (3*R_fa^2 + L_fa^2), M_fa/12 * (3*R_fa^2 + L_fa^2), 0.5 * M_fa * R_fa^2];

            idx_fa_L = find(audit.segments.segment == "forearm_L", 1);
            testCase.assertNotEmpty(idx_fa_L, 'forearm_L in segments table');
            testCase.verifyEqual(audit.segments.I_model(idx_fa_L, :), I_fa_analytic, 'RelTol', 1e-12, ...
                'Lumped forearm moments match continuous cylinder analytic formula');

            % Upper torso lumped moments match continuous cylinder of length FitUpperTorsoLength
            M_ut = ws.getVariable('GolferUpperTrunkMass');
            L_ut = ws.getVariable('FitUpperTorsoLength') * 0.0254;
            R_ut = 6.0 * 0.0254;

            I_ut_analytic = [M_ut/12 * (3*R_ut^2 + L_ut^2), M_ut/12 * (3*R_ut^2 + L_ut^2), 0.5 * M_ut * R_ut^2];

            idx_ut = find(audit.segments.segment == "upper_trunk", 1);
            testCase.assertNotEmpty(idx_ut, 'upper_trunk in segments table');
            testCase.verifyEqual(audit.segments.I_model(idx_ut, :), I_ut_analytic, 'RelTol', 1e-12, ...
                'Lumped upper torso moments match continuous cylinder analytic formula');
        end

        function test_segment_table_structure_and_ratios(testCase)
            % Segment table contains all required segments with positive finite ratios
            audit = gs3dx_inertia_audit(testCase.mdl);
            testCase.assertTrue(istable(audit.segments), 'audit.segments is a table');

            required_segments = ["head", "upper_arm_L", "upper_arm_R", "forearm_L", "forearm_R", ...
                                 "hand_L", "hand_R", "thigh_L", "thigh_R", "shank_L", "shank_R", ...
                                 "foot_L", "foot_R", "lower_trunk", "upper_trunk", "trunk"];

            for seg = required_segments
                idx = find(audit.segments.segment == seg, 1);
                testCase.assertNotEmpty(idx, sprintf('Segment %s present in audit.segments', seg));
                r = audit.segments(idx, :);
                testCase.verifyGreaterThan(r.mass_model, 0, sprintf('%s positive mass', seg));
                testCase.verifyGreaterThan(r.length, 0, sprintf('%s positive length', seg));
                testCase.verifyTrue(all(r.I_model > 0), sprintf('%s positive model moments', seg));
                testCase.verifyTrue(all(r.I_de_leva > 0), sprintf('%s positive de Leva moments', seg));
                testCase.verifyGreaterThan(r.ratio_transverse, 0, sprintf('%s positive transverse ratio', seg));
                testCase.verifyGreaterThan(r.ratio_longitudinal, 0, sprintf('%s positive longitudinal ratio', seg));
                testCase.verifyTrue(isfinite(r.ratio_transverse), sprintf('%s finite transverse ratio', seg));
                testCase.verifyTrue(isfinite(r.ratio_longitudinal), sprintf('%s finite longitudinal ratio', seg));
            end
        end

        function foot_ratios_use_the_brick_long_axis(testCase)
            % The foot brick's long axis is its largest dimension (x); the
            % longitudinal ratio compares the moment about it with de Leva's.
            audit = gs3dx_inertia_audit(testCase.mdl);
            r = audit.segments(audit.segments.segment == "foot_L", :);
            testCase.verifyEqual(r.ratio_longitudinal, r.I_model(1) / r.I_de_leva(3), 'RelTol', 1e-12);
            testCase.verifyEqual(r.ratio_transverse, mean(r.I_model(2:3)) / mean(r.I_de_leva(1:2)), 'RelTol', 1e-12);
            testCase.verifyFalse(any(ismember(["neck" "shoulder_L" "shoulder_R"], audit.segments.segment)), ...
                'segments without a de Leva counterpart are not compared');
        end
    end
end
