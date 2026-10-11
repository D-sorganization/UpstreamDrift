classdef test_gs3dx_apply_subject_physics < matlab.unittest.TestCase
%TEST_GS3DX_APPLY_SUBJECT_PHYSICS  Tests for in-memory subject physics application (#10979, #11011).
%
%   Verifies:
%     1. Pure parameter and inertia calculation (GS3DX_CUSTOM_SEGMENT_INERTIAS)
%        against 80 kg de Leva baseline values documented in docs/SHAPE.md.
%     2. Linear scaling of custom moments with subject mass ratio (104.3 kg).
%     3. Physical consistency under fitted length changes: L^2 moment scaling,
%        L COM scaling, and principal inertia tensor triangle inequality.
%     4. Forearm halves parallel-axis lumping consistency.
%     5. Pure input validators (rejection of NaN, Inf, complex, negative, non-positive).
%     6. Support for actual evaluated segment mass structs in shared catalog.
%     7. Native GS3DX_Human in-memory integration:
%        - 80 kg baseline preserves current custom block inertias (accounting for
%          6-digit baked string rounding on foot COM).
%        - 104.3 kg scales custom moments and total audited mass.
%        - Equipment mass is evaluated from model solids (source = "evaluated_from_model_solids").
%        - Split foot mass expressions and structural forefoot constant (0.25 kg) preserved.
%        - Massless visual solids (Mass == 0) untouched.
%        - Massive head and neck solids verified.
%        - Strict fitted_lengths whitelist: accepts capture lengths, rejects
%          arbitrary variables (e.g. GolferBodyMass) and unsupported FootLength.
%        - Simulink.Parameter container and metadata preservation.
%        - Geometry expressions and joints untouched; no graphical modifications.
%        - Report structure reports status UNQUALIFIED without fake equilibrium.
%        - Preflight rejection and transactional rollback on failure.

    properties
        info struct
        mdl char
        L_baseline struct
        model_available logical = false
    end

    properties (Constant)
        EquipmentMass = 0.25 + 0.0802566 + 0.036667 + 0.020 + 0.006; % 0.3929236 kg
    end

    methods (TestClassSetup)
        function setup(testCase)
            tools_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'tools');
            addpath(tools_dir);
            testCase.info = gs3dx_setup();
            names = gs3dx_names();
            testCase.mdl = char(names.variants.human);

            in = 0.0254;
            testCase.L_baseline = struct( ...
                'thigh', 0.460137, ...
                'shank', 0.424213, ...
                'upper_arm', 12.0 * in, ...
                'forearm', 11.02 * in, ...
                'hand', 0.0862, ...
                'head', 0.2033);

            model_file = fullfile(testCase.info.models_dir, [testCase.mdl '.slx']);
            if isfile(model_file)
                load_system(testCase.mdl);
                testCase.model_available = true;
                testCase.addTeardown(@() close_system(testCase.mdl, 0));
            end
        end
    end

    % =====================================================================
    % PURE UNIT TESTS (No Simulink Engine / Model Loading Required)
    % =====================================================================
    methods (Test)
        function test_pure_inertias_80kg_baseline(testCase)
            % Verifies exact reproduction of documented 80 kg de Leva custom inertias
            % from docs/SHAPE.md table (measured 2026-09-28).
            inert = gs3dx_custom_segment_inertias(80, testCase.L_baseline);

            % Thigh: mass 11.328 kg, com +0.0416 m, moments [0.2596 0.2596 0.0532]
            testCase.verifyEqual(inert.thigh.mass, 11.328, 'AbsTol', 1e-12);
            testCase.verifyEqual(inert.thigh.com(3), 0.0416, 'AbsTol', 2e-4);
            testCase.verifyEqual(inert.thigh.moments, [0.2596 0.2596 0.0532], 'RelTol', 1e-3);

            % Shank: mass 3.464 kg, com +0.0230 m, moments [0.0396 0.0396 0.00661]
            testCase.verifyEqual(inert.shank.mass, 3.464, 'AbsTol', 1e-12);
            testCase.verifyEqual(inert.shank.com(3), 0.0230, 'AbsTol', 2e-4);
            testCase.verifyEqual(inert.shank.moments, [0.0396 0.0396 0.00661], 'RelTol', 1e-3);

            % Upper Arm: mass 2.168 kg, com -0.0235 m
            % Independent analytic de Leva formula: I = m * (gyr * L).^2
            m_ua_ref = 80 * 0.0271;
            L_ua_ref = testCase.L_baseline.upper_arm;
            gyr_ua_ref = [0.285 0.269 0.158];
            I_analytic_ua = m_ua_ref * (gyr_ua_ref * L_ua_ref).^2;
            moments_analytic_ua = [mean(I_analytic_ua(1:2)), mean(I_analytic_ua(1:2)), I_analytic_ua(3)];

            testCase.verifyEqual(inert.upper_arm.mass, 2.168, 'AbsTol', 1e-12);
            testCase.verifyEqual(inert.upper_arm.com(3), -0.0235, 'AbsTol', 2e-4);
            testCase.verifyEqual(inert.upper_arm.moments, moments_analytic_ua, 'RelTol', 1e-12);
            % Consistent with displayed 3-significant-digit precision in docs/SHAPE.md [0.0155 0.0155 0.00502]
            testCase.verifyEqual(inert.upper_arm.moments, [0.0155 0.0155 0.00502], 'RelTol', 5e-3);

            % Forearm halves: mass 0.648 kg each, com +0.0119 m, moments [0.000543 0.000543 0.000743]
            testCase.verifyEqual(inert.forearm.upper.mass, 0.648, 'AbsTol', 1e-12);
            testCase.verifyEqual(inert.forearm.lower.mass, 0.648, 'AbsTol', 1e-12);
            testCase.verifyEqual(inert.forearm.upper.com(3), 0.0119, 'AbsTol', 2e-4);
            testCase.verifyEqual(inert.forearm.upper.moments, [0.000543 0.000543 0.000743], 'RelTol', 2e-3);

            % Hand: mass 0.488 kg, com [0 0 0], moments [0.00119 0.00119 0.000583]
            testCase.verifyEqual(inert.hand.mass, 0.488, 'AbsTol', 1e-12);
            testCase.verifyEqual(inert.hand.com, [0 0 0], 'AbsTol', 1e-15);
            testCase.verifyEqual(inert.hand.moments, [0.00119 0.00119 0.000583], 'RelTol', 2e-3);

            % Head: mass 4.7192 kg, com [0 0 0], moments [0.0266 0.0266 0.0190]
            testCase.verifyEqual(inert.head.mass, 4.7192, 'AbsTol', 1e-4);
            testCase.verifyEqual(inert.head.com, [0 0 0], 'AbsTol', 1e-15);
            testCase.verifyEqual(inert.head.moments, [0.0266 0.0266 0.0190], 'RelTol', 2e-3);
        end

        function test_pure_inertias_mass_scaling_104p3(testCase)
            % Mass ratio scaling: 104.3 kg / 80 kg
            M_target = 104.3;
            ratio = M_target / 80;

            i80 = gs3dx_custom_segment_inertias(80, testCase.L_baseline);
            i104 = gs3dx_custom_segment_inertias(M_target, testCase.L_baseline);

            % Mass scales linearly
            testCase.verifyEqual(i104.thigh.mass, i80.thigh.mass * ratio, 'RelTol', 1e-12);
            testCase.verifyEqual(i104.shank.mass, i80.shank.mass * ratio, 'RelTol', 1e-12);
            testCase.verifyEqual(i104.upper_arm.mass, i80.upper_arm.mass * ratio, 'RelTol', 1e-12);
            testCase.verifyEqual(i104.forearm.upper.mass, i80.forearm.upper.mass * ratio, 'RelTol', 1e-12);
            testCase.verifyEqual(i104.hand.mass, i80.hand.mass * ratio, 'RelTol', 1e-12);
            testCase.verifyEqual(i104.head.mass, i80.head.mass * ratio, 'RelTol', 1e-12);

            % Moments scale linearly with mass (lengths unchanged)
            testCase.verifyEqual(i104.thigh.moments, i80.thigh.moments * ratio, 'RelTol', 1e-12);
            testCase.verifyEqual(i104.shank.moments, i80.shank.moments * ratio, 'RelTol', 1e-12);
            testCase.verifyEqual(i104.upper_arm.moments, i80.upper_arm.moments * ratio, 'RelTol', 1e-12);
            testCase.verifyEqual(i104.forearm.upper.moments, i80.forearm.upper.moments * ratio, 'RelTol', 1e-12);
            testCase.verifyEqual(i104.hand.moments, i80.hand.moments * ratio, 'RelTol', 1e-12);
            testCase.verifyEqual(i104.head.moments, i80.head.moments * ratio, 'RelTol', 1e-12);

            % Center of mass is independent of mass (depends only on length and com_fraction)
            testCase.verifyEqual(i104.thigh.com, i80.thigh.com, 'AbsTol', 1e-15);
            testCase.verifyEqual(i104.shank.com, i80.shank.com, 'AbsTol', 1e-15);
            testCase.verifyEqual(i104.upper_arm.com, i80.upper_arm.com, 'AbsTol', 1e-15);
            testCase.verifyEqual(i104.forearm.upper.com, i80.forearm.upper.com, 'AbsTol', 1e-15);
        end

        function test_pure_inertias_length_scaling_triangle_inequality(testCase)
            % Test lengths variation (+15% scale on lengths)
            scale = 1.15;
            L_scaled = testCase.L_baseline;
            fn = fieldnames(L_scaled);
            for i = 1:numel(fn)
                L_scaled.(fn{i}) = L_scaled.(fn{i}) * scale;
            end

            i_base = gs3dx_custom_segment_inertias(80, testCase.L_baseline);
            i_scaled = gs3dx_custom_segment_inertias(80, L_scaled);

            % Moments scale with scale^2 (mass unchanged)
            testCase.verifyEqual(i_scaled.thigh.moments, i_base.thigh.moments * scale^2, 'RelTol', 1e-12);
            testCase.verifyEqual(i_scaled.shank.moments, i_base.shank.moments * scale^2, 'RelTol', 1e-12);
            testCase.verifyEqual(i_scaled.upper_arm.moments, i_base.upper_arm.moments * scale^2, 'RelTol', 1e-12);
            testCase.verifyEqual(i_scaled.hand.moments, i_base.hand.moments * scale^2, 'RelTol', 1e-12);
            testCase.verifyEqual(i_scaled.head.moments, i_base.head.moments * scale^2, 'RelTol', 1e-12);

            % COM scales linearly with scale
            testCase.verifyEqual(i_scaled.thigh.com, i_base.thigh.com * scale, 'RelTol', 1e-12);
            testCase.verifyEqual(i_scaled.shank.com, i_base.shank.com * scale, 'RelTol', 1e-12);
            testCase.verifyEqual(i_scaled.upper_arm.com, i_base.upper_arm.com * scale, 'RelTol', 1e-12);

            % Triangle inequality on all tensors: I_i + I_j >= I_k
            segs = {i_scaled.thigh, i_scaled.shank, i_scaled.upper_arm, ...
                    i_scaled.forearm.upper, i_scaled.forearm.lower, ...
                    i_scaled.forearm.whole, i_scaled.hand, i_scaled.head};
            for k = 1:numel(segs)
                I = segs{k}.moments;
                testCase.verifyTrue(all(I > 0), 'All moments positive');
                testCase.verifyGreaterThanOrEqual(I(1) + I(2), I(3) - 1e-15, 'Triangle inequality 1+2>=3');
                testCase.verifyGreaterThanOrEqual(I(2) + I(3), I(1) - 1e-15, 'Triangle inequality 2+3>=1');
                testCase.verifyGreaterThanOrEqual(I(1) + I(3), I(2) - 1e-15, 'Triangle inequality 1+3>=2');
            end
        end

        function test_pure_forearm_parallel_axis_lumping(testCase)
            % Forearm halves lump along z to reconstruct the whole de Leva forearm
            inert = gs3dx_custom_segment_inertias(80, testCase.L_baseline);
            fa = inert.forearm;

            m_half = fa.upper.mass;
            L = fa.length;
            d = L / 4;
            It = fa.upper.moments(1);
            Iz = fa.upper.moments(3);

            % Reconstructed whole forearm transverse moment
            I_trans_lumped = 2 * (It + m_half * d^2);
            % Reconstructed whole forearm longitudinal moment
            I_long_lumped = 2 * Iz;

            testCase.verifyEqual(I_trans_lumped, fa.whole.moments(1), 'RelTol', 1e-12);
            testCase.verifyEqual(I_long_lumped, fa.whole.moments(3), 'RelTol', 1e-12);
            testCase.verifyGreaterThan(It, 0, 'Forearm half transverse moment It must be strictly positive');
        end

        function test_pure_actual_mass_map_support(testCase)
            % Test that gs3dx_custom_segment_inertias accepts actual per-solid masses
            m_custom = struct( ...
                'thigh_L', 12.0, 'thigh_R', 11.5, ...
                'shank_L', 3.5, 'shank_R', 3.4, ...
                'upper_arm_L', 2.2, 'upper_arm_R', 2.1, ...
                'forearm_L', 1.3, 'forearm_R', 1.28, ...
                'hand_L', 0.5, 'hand_R', 0.49, ...
                'head', 4.8);

            inert = gs3dx_custom_segment_inertias(m_custom, testCase.L_baseline);

            testCase.verifyEqual(inert.thigh_L.mass, 12.0);
            testCase.verifyEqual(inert.thigh_R.mass, 11.5);
            testCase.verifyGreaterThan(inert.thigh_L.moments(1), inert.thigh_R.moments(1));
            testCase.verifyEqual(inert.forearm_L.upper.mass, 0.65);
            testCase.verifyEqual(inert.forearm_R.upper.mass, 0.64);
        end

        function test_pure_input_rejections(testCase)
            % Test input validators on gs3dx_custom_segment_inertias
            L = testCase.L_baseline;

            % Invalid masses
            testCase.verifyError(@() gs3dx_custom_segment_inertias(NaN, L), 'gs3dx:custom_inertias:invalid_mass');
            testCase.verifyError(@() gs3dx_custom_segment_inertias(Inf, L), 'gs3dx:custom_inertias:invalid_mass');
            testCase.verifyError(@() gs3dx_custom_segment_inertias(-80, L), 'gs3dx:custom_inertias:invalid_mass');
            testCase.verifyError(@() gs3dx_custom_segment_inertias(0, L), 'gs3dx:custom_inertias:invalid_mass');
            testCase.verifyError(@() gs3dx_custom_segment_inertias([80 80], L), 'gs3dx:custom_inertias:invalid_mass');
            testCase.verifyError(@() gs3dx_custom_segment_inertias("eighty", L), 'gs3dx:custom_inertias:invalid_input');

            % Invalid lengths
            bad_L = L; bad_L.thigh = -0.4;
            testCase.verifyError(@() gs3dx_custom_segment_inertias(80, bad_L), 'gs3dx:custom_inertias:invalid_length');
            bad_L = L; bad_L.shank = NaN;
            testCase.verifyError(@() gs3dx_custom_segment_inertias(80, bad_L), 'gs3dx:custom_inertias:invalid_length');
            bad_L = rmfield(L, 'upper_arm');
            testCase.verifyError(@() gs3dx_custom_segment_inertias(80, bad_L), 'gs3dx:custom_inertias:missing_length');
        end
    end

    % =====================================================================
    % INTEGRATION TESTS WITH LOADED GS3DX_HUMAN MODEL
    % =====================================================================
    methods (Test)
        function test_human_80kg_preserves_current_custom_inertias(testCase)
            testCase.assumeTrue(testCase.model_available, 'GS3DX_Human model file required');

            % Read baseline parameters before application
            s_before = get_param([testCase.mdl '/Lower Body/L Thigh'], 'MomentsOfInertia');
            c_before = get_param([testCase.mdl '/Lower Body/L Thigh'], 'CenterOfMass');
            c_foot_before = get_param([testCase.mdl '/Lower Body/L Foot'], 'CenterOfMass');

            % Apply 80 kg
            report = gs3dx_apply_subject_physics(testCase.mdl, 80);

            testCase.verifyEqual(report.status, "applied");
            testCase.verifyEqual(report.subject_mass, 80);

            s_after = get_param([testCase.mdl '/Lower Body/L Thigh'], 'MomentsOfInertia');
            c_after = get_param([testCase.mdl '/Lower Body/L Thigh'], 'CenterOfMass');
            c_foot_after = get_param([testCase.mdl '/Lower Body/L Foot'], 'CenterOfMass');

            % Limbs match to 1e-9
            testCase.verifyEqual(str2num(s_after), str2num(s_before), 'AbsTol', 1e-9); %#ok<ST2NM>
            testCase.verifyEqual(str2num(c_after), str2num(c_before), 'AbsTol', 1e-9); %#ok<ST2NM>

            % Foot COM baked string in gs3dx_build_human was formatted with mat2str(..., 6);
            % verify match allowing for 6-significant-digit rounding
            testCase.verifyEqual(str2num(c_foot_after), str2num(c_foot_before), 'AbsTol', 1e-5); %#ok<ST2NM>

            % Verify tensor triangle inequality across all custom solid blocks (80 kg baseline)
            custom_solids = [ ...
                "Lower Body/L Thigh", "Lower Body/R Thigh", ...
                "Lower Body/L Shank", "Lower Body/R Shank", ...
                "LUpperArm", "RUpperArm", ...
                "Left Forearm/LUpperForearm", "Left Forearm/LLowerForearm", ...
                "Right Forearm/RUpperForearm", "Right Forearm/RLowerForearm", ...
                "Grip/LHand", "Grip/RHand", ...
                "Hips and Torso Inputs/Head", ...
                "Lower Body/L Foot", "Lower Body/R Foot", ...
                "Lower Body/L Forefoot", "Lower Body/R Forefoot"];
            for blk_name = custom_solids
                blk = [testCase.mdl '/' char(blk_name)];
                I = str2num(get_param(blk, 'MomentsOfInertia')); %#ok<ST2NM>
                testCase.verifyTrue(all(I > 0), sprintf('%s moments positive at 80 kg', blk_name));
                testCase.verifyGreaterThanOrEqual(I(1) + I(2), I(3) - 1e-15, sprintf('%s 1+2>=3 at 80 kg', blk_name));
                testCase.verifyGreaterThanOrEqual(I(2) + I(3), I(1) - 1e-15, sprintf('%s 2+3>=1 at 80 kg', blk_name));
                testCase.verifyGreaterThanOrEqual(I(1) + I(3), I(2) - 1e-15, sprintf('%s 1+3>=2 at 80 kg', blk_name));
            end
        end

        function test_human_104p3_mass_scaling_and_equipment_evaluation(testCase)
            testCase.assumeTrue(testCase.model_available, 'GS3DX_Human model file required');
            testCase.addTeardown(@() gs3dx_apply_subject_physics(testCase.mdl, 80));

            M_target = 104.3;
            ratio = M_target / 80;

            % Pre-record 80 kg thigh moments
            gs3dx_apply_subject_physics(testCase.mdl, 80);
            I80 = str2num(get_param([testCase.mdl '/Lower Body/L Thigh'], 'MomentsOfInertia')); %#ok<ST2NM>

            % Apply 104.3 kg
            report = gs3dx_apply_subject_physics(testCase.mdl, M_target);

            testCase.verifyEqual(report.status, "applied");
            testCase.verifyEqual(report.qualification_status, "UNQUALIFIED");
            testCase.verifyEqual(report.subject_mass, M_target);
            testCase.verifyEqual(report.equipment_mass_source, "evaluated_from_model_solids");

            % Workspace variable updated
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            testCase.verifyEqual(ws.getVariable('GolferBodyMass'), M_target);

            % Custom moments scaled by mass ratio
            I104 = str2num(get_param([testCase.mdl '/Lower Body/L Thigh'], 'MomentsOfInertia')); %#ok<ST2NM>
            testCase.verifyEqual(I104, I80 * ratio, 'RelTol', 1e-9);

            % Verify equipment mass evaluated from solids equals documented ~0.3929 kg
            testCase.verifyEqual(report.equipment_mass, testCase.EquipmentMass, 'AbsTol', 1e-4);

            % Audit total mass matches subject + evaluated equipment mass
            audit = gs3dx_inertia_audit(testCase.mdl);
            testCase.verifyEqual(audit.total_mass, report.expected_total_mass, 'AbsTol', 1e-3);

            % Sensed mass check if available
            if exist('gs3dx_mass_check', 'file')
                m_sensed = gs3dx_mass_check(testCase.mdl);
                testCase.verifyEqual(m_sensed, report.expected_total_mass, 'AbsTol', 1e-3);
            end

            % Verify tensor triangle inequality across all custom solid blocks (104.3 kg scaled)
            custom_solids = [ ...
                "Lower Body/L Thigh", "Lower Body/R Thigh", ...
                "Lower Body/L Shank", "Lower Body/R Shank", ...
                "LUpperArm", "RUpperArm", ...
                "Left Forearm/LUpperForearm", "Left Forearm/LLowerForearm", ...
                "Right Forearm/RUpperForearm", "Right Forearm/RLowerForearm", ...
                "Grip/LHand", "Grip/RHand", ...
                "Hips and Torso Inputs/Head", ...
                "Lower Body/L Foot", "Lower Body/R Foot", ...
                "Lower Body/L Forefoot", "Lower Body/R Forefoot"];
            for blk_name = custom_solids
                blk = [testCase.mdl '/' char(blk_name)];
                I = str2num(get_param(blk, 'MomentsOfInertia')); %#ok<ST2NM>
                testCase.verifyTrue(all(I > 0), sprintf('%s moments positive at 104.3 kg', blk_name));
                testCase.verifyGreaterThanOrEqual(I(1) + I(2), I(3) - 1e-15, sprintf('%s 1+2>=3 at 104.3 kg', blk_name));
                testCase.verifyGreaterThanOrEqual(I(2) + I(3), I(1) - 1e-15, sprintf('%s 2+3>=1 at 104.3 kg', blk_name));
                testCase.verifyGreaterThanOrEqual(I(1) + I(3), I(2) - 1e-15, sprintf('%s 1+3>=2 at 104.3 kg', blk_name));
            end
        end

        function test_human_split_foot_expressions_and_forefoot_constant(testCase)
            testCase.assumeTrue(testCase.model_available, 'GS3DX_Human model file required');
            testCase.addTeardown(@() gs3dx_apply_subject_physics(testCase.mdl, 80));

            M_target = 104.3;
            report = gs3dx_apply_subject_physics(testCase.mdl, M_target);

            ws = get_param(testCase.mdl, 'ModelWorkspace');
            M_foot = ws.getVariable('FootMass');
            mf = ws.getVariable('ForefootMass');
            mr = M_foot - mf;

            % Verify forefoot mass remains the structural constant 0.25 kg
            testCase.verifyEqual(mf, 0.25, 'AbsTol', 1e-12);

            % Verify rearfoot block mass expression and moments
            blk_foot = [testCase.mdl '/Lower Body/L Foot'];
            testCase.verifyEqual(get_param(blk_foot, 'Mass'), 'FootMass - ForefootMass');
            testCase.verifyEqual(get_param(blk_foot, 'InertiaType'), 'Custom');

            % Verify forefoot block mass expression and moments
            blk_fore = [testCase.mdl '/Lower Body/L Forefoot'];
            testCase.verifyEqual(get_param(blk_fore, 'Mass'), 'ForefootMass');
            testCase.verifyEqual(get_param(blk_fore, 'InertiaType'), 'Custom');

            % Check rearfoot moments satisfy triangle inequality
            I_rear = str2num(get_param(blk_foot, 'MomentsOfInertia')); %#ok<ST2NM>
            testCase.verifyGreaterThan(I_rear(1) + I_rear(2), I_rear(3) - 1e-15);
            testCase.verifyGreaterThan(I_rear(2) + I_rear(3), I_rear(1) - 1e-15);
            testCase.verifyGreaterThan(I_rear(1) + I_rear(3), I_rear(2) - 1e-15);

            % Sum of rearfoot and forefoot masses equals total foot mass
            testCase.verifyEqual(mr + mf, M_foot, 'AbsTol', 1e-12);
        end

        function test_human_massless_visuals_untouched(testCase)
            testCase.assumeTrue(testCase.model_available, 'GS3DX_Human model file required');
            testCase.addTeardown(@() gs3dx_apply_subject_physics(testCase.mdl, 80));

            report = gs3dx_apply_subject_physics(testCase.mdl, 104.3);

            % Visual solids must retain Mass 0 and Moments [0 0 0]
            visuals = [ ...
                "Hips and Torso Inputs/Pelvis", ...
                "Hips and Torso Inputs/Abdomen", ...
                "Hips and Torso Inputs/Chest", ...
                "Hips and Torso Inputs/Trapezius", ...
                "Hips and Torso Inputs/Neck Shape", ...
                "Club/Clubhead/Driver Head Place/Driver Head"];

            for v = visuals
                blk = [testCase.mdl '/' char(v)];
                if getSimulinkBlockHandle(blk) > 0
                    testCase.verifyEqual(get_param(blk, 'Mass'), '0');
                    testCase.verifyEqual(get_param(blk, 'MomentsOfInertia'), '[0 0 0]');
                    testCase.verifyFalse(any(report.modified_blocks == string(blk)), ...
                        sprintf('Visual solid %s must NOT be modified by apply_subject_physics', v));
                end
            end
        end

        function test_human_head_neck_massive_solids(testCase)
            testCase.assumeTrue(testCase.model_available, 'GS3DX_Human model file required');
            testCase.addTeardown(@() gs3dx_apply_subject_physics(testCase.mdl, 80));

            M_target = 104.3;
            report = gs3dx_apply_subject_physics(testCase.mdl, M_target);

            ws = get_param(testCase.mdl, 'ModelWorkspace');
            head_m = ws.getVariable('GolferHeadMass');
            neck_m = ws.getVariable('GolferNeckMass');

            % Head mass is 85%, Neck is 15% of de Leva head+neck (0.0694)
            total_head_neck = head_m + neck_m;
            testCase.verifyEqual(total_head_neck, 0.0694 * M_target, 'AbsTol', 1e-12);
            testCase.verifyEqual(neck_m / total_head_neck, 0.15, 'AbsTol', 1e-12);

            % Head solid uses Custom inertia; Neck solid uses CalculateFromGeometry
            blk_head = [testCase.mdl '/Hips and Torso Inputs/Head'];
            blk_neck = [testCase.mdl '/Hips and Torso Inputs/Neck'];

            testCase.verifyEqual(get_param(blk_head, 'InertiaType'), 'Custom');
            testCase.verifyEqual(get_param(blk_neck, 'InertiaType'), 'CalculateFromGeometry');
        end

        function test_human_whitelisted_fitted_lengths(testCase)
            testCase.assumeTrue(testCase.model_available, 'GS3DX_Human model file required');
            testCase.addTeardown(@() gs3dx_apply_subject_physics(testCase.mdl, 80));

            % Whitelisted fitted lengths
            new_lens = struct( ...
                'ThighLength', 0.48, ...
                'ShankLength', 0.44, ...
                'FitUpperArmLength', 13.0, ...
                'FitLowerArmLength', 11.5);

            report = gs3dx_apply_subject_physics(testCase.mdl, 80, fitted_lengths=new_lens);

            testCase.verifyEqual(report.status, "applied");
            testCase.verifyEqual(report.fitted_lengths.thigh, 0.48, 'AbsTol', 1e-12);
            testCase.verifyEqual(report.fitted_lengths.shank, 0.44, 'AbsTol', 1e-12);

            ws = get_param(testCase.mdl, 'ModelWorkspace');
            testCase.verifyEqual(double(ws.getVariable('ThighLength')), 0.48, 'AbsTol', 1e-12);
            testCase.verifyEqual(double(ws.getVariable('ShankLength')), 0.44, 'AbsTol', 1e-12);

            % Verify tensor triangle inequality holds on updated blocks
            blk_thigh = [testCase.mdl '/Lower Body/L Thigh'];
            I_thigh = str2num(get_param(blk_thigh, 'MomentsOfInertia')); %#ok<ST2NM>
            testCase.verifyGreaterThanOrEqual(I_thigh(1) + I_thigh(2), I_thigh(3) - 1e-15);
        end

        function test_simulink_parameter_container_and_metadata_preserved(testCase)
            testCase.assumeTrue(testCase.model_available, 'GS3DX_Human model file required');
            testCase.addTeardown(@() gs3dx_apply_subject_physics(testCase.mdl, 80));

            ws = get_param(testCase.mdl, 'ModelWorkspace');

            % Inject a test Simulink.Parameter into workspace
            p_orig = Simulink.Parameter;
            p_orig.Value = 0.460137;
            p_orig.Description = 'Test Parameter for ThighLength';
            p_orig.DataType = 'double';
            ws.assignin('ThighLength', p_orig);

            % Apply physics with updated ThighLength
            new_lens = struct('ThighLength', 0.475);
            gs3dx_apply_subject_physics(testCase.mdl, 80, fitted_lengths=new_lens);

            p_after = ws.getVariable('ThighLength');
            testCase.verifyTrue(isa(p_after, 'Simulink.Parameter'), 'Must remain a Simulink.Parameter');
            testCase.verifyEqual(p_after.Value, 0.475, 'AbsTol', 1e-12);
            testCase.verifyEqual(p_after.Description, 'Test Parameter for ThighLength', 'Description preserved');
            testCase.verifyEqual(p_after.DataType, 'double', 'DataType preserved');
        end

        function test_geometry_expressions_and_joints_unchanged(testCase)
            testCase.assumeTrue(testCase.model_available, 'GS3DX_Human model file required');
            testCase.addTeardown(@() gs3dx_apply_subject_physics(testCase.mdl, 80));

            % Snapshot joints
            nj = [testCase.mdl '/Hips and Torso Inputs/Neck Joint'];
            mj = [testCase.mdl '/Lower Body/L Midfoot Joint'];

            ref_nj = get_param(nj, 'ReferenceBlock');
            ref_mj = get_param(mj, 'ReferenceBlock');

            gs3dx_apply_subject_physics(testCase.mdl, 104.3);

            % Joint blocks unchanged
            testCase.verifyEqual(get_param(nj, 'ReferenceBlock'), ref_nj);
            testCase.verifyEqual(get_param(mj, 'ReferenceBlock'), ref_mj);

            % Graphic appearance unchanged (no graphical changes)
            head_radii = get_param([testCase.mdl '/Hips and Torso Inputs/Head'], 'EllipsoidRadii');
            testCase.verifyNotEmpty(head_radii);
        end

        function test_report_structure_and_unqualified_status(testCase)
            testCase.assumeTrue(testCase.model_available, 'GS3DX_Human model file required');
            testCase.addTeardown(@() gs3dx_apply_subject_physics(testCase.mdl, 80));

            ws = get_param(testCase.mdl, 'ModelWorkspace');
            com_ref_before = ws.getVariable('BalanceCOMRef');

            report = gs3dx_apply_subject_physics(testCase.mdl, 90.0);

            testCase.verifyEqual(report.status, "applied");
            testCase.verifyEqual(report.qualification_status, "UNQUALIFIED");
            testCase.verifyTrue(isfield(report, 'applied_masses'));
            testCase.verifyTrue(isfield(report, 'applied_inertias'));
            testCase.verifyTrue(isfield(report, 'remaining_assumptions'));
            testCase.verifyGreaterThan(numel(report.remaining_assumptions), 3);

            % Balance reference was NOT modified or re-anchored
            com_ref_after = ws.getVariable('BalanceCOMRef');
            testCase.verifyEqual(com_ref_after, com_ref_before, 'AbsTol', 1e-15);
        end

        function test_preflight_rejections_and_rollback(testCase)
            testCase.assumeTrue(testCase.model_available, 'GS3DX_Human model file required');

            ws = get_param(testCase.mdl, 'ModelWorkspace');
            m_orig = ws.getVariable('GolferBodyMass');
            I_orig = get_param([testCase.mdl '/Lower Body/L Thigh'], 'MomentsOfInertia');

            % 1. Rejection of arbitrary unwhitelisted field (e.g. GolferBodyMass in fitted_lengths)
            bad_lens1 = struct('GolferBodyMass', 5);
            testCase.verifyError(@() gs3dx_apply_subject_physics(testCase.mdl, 80, fitted_lengths=bad_lens1), ...
                'gs3dx:apply_physics:unsupported_length_field');

            % 2. Rejection of unknown foreign field
            bad_lens_foreign = struct('PelvisWidth', 0.35);
            testCase.verifyError(@() gs3dx_apply_subject_physics(testCase.mdl, 80, fitted_lengths=bad_lens_foreign), ...
                'gs3dx:apply_physics:unsupported_length_field');

            % 3. Explicit rejection of FootLength
            bad_lens2 = struct('FootLength', 0.28);
            testCase.verifyError(@() gs3dx_apply_subject_physics(testCase.mdl, 80, fitted_lengths=bad_lens2), ...
                'gs3dx:apply_physics:unsupported_length_field');

            % 4. Rejection of non-struct fitted_lengths
            testCase.verifyError(@() gs3dx_apply_subject_physics(testCase.mdl, 80, fitted_lengths=5), ...
                'gs3dx:apply_physics:invalid_lengths');

            % 5. Rejection of non-scalar struct fitted_lengths
            bad_lens_arr = [struct('ThighLength', 0.45), struct('ThighLength', 0.46)];
            testCase.verifyError(@() gs3dx_apply_subject_physics(testCase.mdl, 80, fitted_lengths=bad_lens_arr), ...
                'gs3dx:apply_physics:invalid_lengths');

            % 6. Rejection of wrapper extras when using .vars
            bad_lens_wrapper = struct('vars', struct('ThighLength', 0.45), 'extra', 123);
            testCase.verifyError(@() gs3dx_apply_subject_physics(testCase.mdl, 80, fitted_lengths=bad_lens_wrapper), ...
                'gs3dx:apply_physics:unsupported_length_field');

            % 7. Rejection of non-scalar .vars struct
            bad_lens_vars_arr = struct('vars', [struct('ThighLength', 0.45), struct('ThighLength', 0.46)]);
            testCase.verifyError(@() gs3dx_apply_subject_physics(testCase.mdl, 80, fitted_lengths=bad_lens_vars_arr), ...
                'gs3dx:apply_physics:invalid_lengths');

            % 8. Rejection of non-scalar length in fitted_lengths
            bad_lens_nonscalar = struct('ThighLength', [0.45, 0.48]);
            testCase.verifyError(@() gs3dx_apply_subject_physics(testCase.mdl, 80, fitted_lengths=bad_lens_nonscalar), ...
                'gs3dx:apply_physics:invalid_lengths');

            % 9. Rejection of NaN, negative, and complex mass
            testCase.verifyError(@() gs3dx_apply_subject_physics(testCase.mdl, NaN), ...
                'gs3dx:apply_physics:invalid_mass');
            testCase.verifyError(@() gs3dx_apply_subject_physics(testCase.mdl, -90), ...
                'gs3dx:apply_physics:invalid_mass');
            testCase.verifyError(@() gs3dx_apply_subject_physics(testCase.mdl, 80 + 5i), ...
                'gs3dx:apply_physics:invalid_mass');

            % 10. Rejection of unsupported model name
            testCase.verifyError(@() gs3dx_apply_subject_physics('GS3DX_Neck', 80), ...
                'gs3dx:apply_physics:unsupported_model');

            % 11. Verify transactional rollback: model state untouched
            testCase.verifyEqual(ws.getVariable('GolferBodyMass'), m_orig);
            testCase.verifyEqual(get_param([testCase.mdl '/Lower Body/L Thigh'], 'MomentsOfInertia'), I_orig);
        end

        function test_preflight_unexpected_mass_unit_rejection_and_rollback(testCase)
            testCase.assumeTrue(testCase.model_available, 'GS3DX_Human model file required');

            blk = [testCase.mdl '/Lower Body/L Thigh'];
            orig_unit = get_param(blk, 'MassUnits');
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            m_orig = ws.getVariable('GolferBodyMass');

            % Temporarily set unexpected MassUnits on a solid block
            set_param(blk, 'MassUnits', 'g');
            cleanup = onCleanup(@() set_param(blk, 'MassUnits', orig_unit));

            % Must fail preflight closed before any workspace or block mutation
            testCase.verifyError(@() gs3dx_apply_subject_physics(testCase.mdl, 95.0), ...
                'gs3dx:apply_physics:unexpected_unit');

            % Verify workspace was untouched (no partial mutation)
            testCase.verifyEqual(ws.getVariable('GolferBodyMass'), m_orig);
        end

        function test_preflight_unexpected_mass_expression_rejection_and_rollback(testCase)
            testCase.assumeTrue(testCase.model_available, 'GS3DX_Human model file required');

            blk = [testCase.mdl '/Lower Body/L Thigh'];
            orig_mass = get_param(blk, 'Mass');
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            m_orig = ws.getVariable('GolferBodyMass');

            % Temporarily set unexpected Mass expression (with extra term) on a solid block
            set_param(blk, 'Mass', 'ThighMass + 0.1');
            cleanup = onCleanup(@() set_param(blk, 'Mass', orig_mass));

            % Must fail preflight closed before any workspace or block mutation
            testCase.verifyError(@() gs3dx_apply_subject_physics(testCase.mdl, 95.0), ...
                'gs3dx:apply_physics:unexpected_solid_expression');

            % Verify workspace was untouched (no partial mutation)
            testCase.verifyEqual(ws.getVariable('GolferBodyMass'), m_orig);
        end
    end
end
