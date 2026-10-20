classdef test_gs3dx_grip_pair_wrench < matlab.unittest.TestCase
%TEST_GS3DX_GRIP_PAIR_WRENCH  Constitutive golden-vector tests for gs3dx_grip_contact_kernel (#13334).
%
%   Verifies that the generated %#codegen MATLAB kernel
%   (tools/generated/gs3dx_grip_contact_kernel.m) matches the Python
%   pair_contact / topology_contact golden vectors in
%   tests/fixtures/gs3dx_grip_pair_golden.json to <= 1e-10 relative and
%   <= 1e-9 N (or N*m / m / m/s) absolute tolerance across all styles
%   (overlap, interlock, ten_finger), randomized 3D states, and edge cases
%   (separated, just_touching, near_collinear_capsules, stick, slip,
%   zero_relative_velocity, coincident_axis, rebounding_clamp).

    properties (Constant)
        REL_TOL = 1e-10
        ABS_TOL = 1e-9
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.fatalAssertEqual(string(version('-release')), "2025b", ...
                'MATLAB R2025b is required for HRG-GS-M12 (#13334).');
            here = fileparts(mfilename('fullpath'));
            gs3dx_root = fileparts(here);
            tools_dir = fullfile(gs3dx_root, 'tools');
            gen_dir = fullfile(tools_dir, 'generated');
            addpath(here);
            addpath(tools_dir);
            addpath(gen_dir);
        end
    end

    methods (Test)
        function testGoldenVectorsParity(testCase)
            here = fileparts(mfilename('fullpath'));
            fixture_path = fullfile(here, 'fixtures', 'gs3dx_grip_pair_golden.json');
            testCase.fatalAssertTrue(isfile(fixture_path), ...
                sprintf('Missing golden fixture: %s', fixture_path));
            payload = jsondecode(fileread(fixture_path));
            testCase.verifyEqual(string(payload.schema_version), ...
                "simscape-grip-pair-golden/v1");

            cases = payload.cases;
            testCase.verifyGreaterThanOrEqual(numel(cases), 36);

            categories_seen = strings(0, 1);
            styles_seen = strings(0, 1);

            for idx = 1:numel(cases)
                c = cases(idx);
                cid = char(c.case_id);
                categories_seen(end + 1, 1) = string(c.category); %#ok<AGROW>
                styles_seen(end + 1, 1) = string(c.style); %#ok<AGROW>

                inp = c.inputs;
                exp = c.expected;

                p_club = reshape(double(inp.p_club_m), [3, 1]);
                R_club = reshape(double(inp.r_club), [3, 3]);
                v_club = reshape(double(inp.v_club_m_s), [3, 1]);
                omega_club = reshape(double(inp.omega_club_rad_s), [3, 1]);

                p_L = reshape(double(inp.p_hand_l_m), [3, 1]);
                R_L = reshape(double(inp.r_hand_l), [3, 3]);
                v_L = reshape(double(inp.v_hand_l_m_s), [3, 1]);
                omega_L = reshape(double(inp.omega_hand_l_rad_s), [3, 1]);

                p_R = reshape(double(inp.p_hand_r_m), [3, 1]);
                R_R = reshape(double(inp.r_hand_r), [3, 3]);
                v_R = reshape(double(inp.v_hand_r_m_s), [3, 1]);
                omega_R = reshape(double(inp.omega_hand_r_rad_s), [3, 1]);

                style_id = double(c.style_id);

                [F_club_w, M_club_w, F_hand_L_w, M_hand_L_w, F_hand_R_w, M_hand_R_w, ...
                    pair_normal_n, pair_tangent_n, pair_spin_nm, pair_penetration_m, ...
                    pair_slip_speed_m_s, pair_signed_dist_m, pair_force_club_w, ...
                    pair_moment_club_w, pair_contact_point_w] = ...
                    gs3dx_grip_contact_kernel( ...
                        p_club, R_club, v_club, omega_club, ...
                        p_L, R_L, v_L, omega_L, ...
                        p_R, R_R, v_R, omega_R, ...
                        style_id);

                local_assert_close(testCase, F_club_w, reshape(double(exp.f_club_w_n), [3, 1]), ...
                    [cid ': f_club_w_n']);
                local_assert_close(testCase, M_club_w, reshape(double(exp.m_club_w_nm), [3, 1]), ...
                    [cid ': m_club_w_nm']);
                local_assert_close(testCase, F_hand_L_w, reshape(double(exp.f_hand_l_w_n), [3, 1]), ...
                    [cid ': f_hand_l_w_n']);
                local_assert_close(testCase, M_hand_L_w, reshape(double(exp.m_hand_l_w_nm), [3, 1]), ...
                    [cid ': m_hand_l_w_nm']);
                local_assert_close(testCase, F_hand_R_w, reshape(double(exp.f_hand_r_w_n), [3, 1]), ...
                    [cid ': f_hand_r_w_n']);
                local_assert_close(testCase, M_hand_R_w, reshape(double(exp.m_hand_r_w_nm), [3, 1]), ...
                    [cid ': m_hand_r_w_nm']);

                local_assert_close(testCase, pair_normal_n, reshape(double(exp.pair_normal_n), [12, 1]), ...
                    [cid ': pair_normal_n']);
                local_assert_close(testCase, pair_tangent_n, reshape(double(exp.pair_tangent_n), [12, 1]), ...
                    [cid ': pair_tangent_n']);
                local_assert_close(testCase, pair_spin_nm, reshape(double(exp.pair_spin_nm), [12, 1]), ...
                    [cid ': pair_spin_nm']);
                local_assert_close(testCase, pair_penetration_m, reshape(double(exp.pair_penetration_m), [12, 1]), ...
                    [cid ': pair_penetration_m']);
                local_assert_close(testCase, pair_slip_speed_m_s, reshape(double(exp.pair_slip_speed_m_s), [12, 1]), ...
                    [cid ': pair_slip_speed_m_s']);
                local_assert_close(testCase, pair_signed_dist_m, reshape(double(exp.pair_signed_dist_m), [12, 1]), ...
                    [cid ': pair_signed_dist_m']);
                local_assert_close(testCase, pair_force_club_w, double(exp.pair_force_club_w_n), ...
                    [cid ': pair_force_club_w_n']);
                local_assert_close(testCase, pair_moment_club_w, double(exp.pair_moment_club_w_nm), ...
                    [cid ': pair_moment_club_w_nm']);
                local_assert_close(testCase, pair_contact_point_w, double(exp.pair_contact_point_w_m), ...
                    [cid ': pair_contact_point_w_m']);

                % Action-reaction invariance on net force
                local_assert_close(testCase, F_club_w + F_hand_L_w + F_hand_R_w, zeros(3, 1), ...
                    [cid ': action_reaction']);
            end

            req_cats = ["separated", "just_touching", "near_collinear_capsules", ...
                "stick", "slip", "zero_relative_velocity", "coincident_axis", ...
                "rebounding_clamp", "randomized"];
            for rc = req_cats
                testCase.verifyTrue(any(categories_seen == rc), ...
                    sprintf('Missing required edge-case category: %s', rc));
            end
            for st = ["overlap", "interlock", "ten_finger"]
                testCase.verifyTrue(any(styles_seen == st), ...
                    sprintf('Missing style: %s', st));
            end
        end
    end
end

function local_assert_close(testCase, actual, expected, label)
    diff_abs = abs(actual - expected);
    bound = max(test_gs3dx_grip_pair_wrench.ABS_TOL, ...
        test_gs3dx_grip_pair_wrench.REL_TOL * abs(expected));
    testCase.verifyLessThanOrEqual(diff_abs, bound, label);
end
