classdef test_control_names < matlab.unittest.TestCase
%TEST_CONTROL_NAMES Verify the +control_names registry (issue #4042).
%
%   Run from the matlab/ root:
%     >> addpath(genpath('motion_matching/shared'));
%     >> runtests('motion_matching.shared.tests.test_control_names')

    methods(Test)
        function test_torque_to_polynomial_base_nonempty(tc)
            m = control_names.get_torque_to_polynomial_base();
            tc.verifyGreaterThanOrEqual(m.Count, 18);
        end

        function test_coefficient_letters_ABCDEFG(tc)
            letters = control_names.get_coefficient_letters();
            tc.verifyEqual(letters, {'A','B','C','D','E','F','G'});
        end

        function test_all_coefficient_names_unique(tc)
            names = control_names.all_coefficient_names();
            tc.verifyEqual(numel(names), numel(unique(names)));
        end

        function test_all_coefficient_names_length(tc)
            names = control_names.all_coefficient_names();
            m = control_names.get_torque_to_polynomial_base();
            n_unique_bases = numel(unique(values(m)));
            tc.verifyEqual(numel(names), n_unique_bases * 7);
        end

        function test_matches_python_fixture(tc)
            here = fileparts(mfilename('fullpath'));
            fixturePath = fullfile(here, '..', '..', '..', '..', '..', '..', '..', ...
                'tests', 'fixtures', 'control_names_matlab.json');
            tc.assumeTrue(isfile(fixturePath), ...
                'fixture missing; run tools/regen_control_names_fixture.m');
            txt = fileread(fixturePath);
            fixture = jsondecode(txt);
            names = control_names.all_coefficient_names();
            tc.verifyEqual(names(:), cellstr(fixture.all_coefficient_names));
            tc.verifyEqual(control_names.manifest_sha256(), ...
                char(fixture.manifest_sha256));
        end
    end
end
