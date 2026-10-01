classdef test_gs3dx_golfer < matlab.unittest.TestCase
%TEST_GS3DX_GOLFER  GS3DX_Golfer: typical (de Leva) segment masses (#11011).
%
%   The table sums to the body mass, the legs and upper body use the same
%   table, every upper-body solid takes its mass from a Golfer* variable,
%   the edit adds no blocks, the simulated whole-body mass is the body mass
%   plus the equipment, and the lighter golfer still stands from rest.

    properties
        info struct
        mdl char
    end

    properties (Constant)
        % Club head 0.25 kg, shaft 0.0803 kg, grip 10 + 10 + 16.667 g,
        % hand standoffs 2 x 10 g, six 1 g contact spheres: not body mass.
        EquipmentMass = 0.25 + 0.0802566 + 0.036667 + 0.020 + 0.006
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
            testCase.mdl = char(gs3dx_names().variants.golfer);
            if ~isfile(fullfile(testCase.info.models_dir, [testCase.mdl '.slx']))
                gs3dx_build_golfer(testCase.info);
            end
            load_system(testCase.mdl);
            testCase.addTeardown(@() close_system(testCase.mdl, 0));
        end
    end

    methods (Test)
        function table_sums_to_body_mass(testCase)
            for M = [60 80 100]
                a = gs3dx_anthropometry(M);
                v = a.vars;
                l = a.legs;
                total = v.GolferHeadMass + v.GolferNeckMass + v.GolferLowerTrunkMass + v.GolferUpperTrunkMass ...
                    + 2 * (v.GolferShoulderMass + v.GolferUpperArmMass + v.GolferForearmMass + v.GolferHandMass ...
                    + l.ThighMass + l.ShankMass + l.FootMass);
                testCase.verifyEqual(total, M, 'RelTol', 1e-12);
                testCase.verifyEqual(v.GolferLowerTrunkMass + v.GolferUpperTrunkMass + 2 * v.GolferShoulderMass, ...
                    a.fraction.trunk * M, 'RelTol', 1e-12, 'trunk');
            end
        end

        function legs_use_the_same_table(testCase)
            p = gs3dx_leg_table().params;
            l = gs3dx_anthropometry(p.LegBodyMass).legs;
            for f = fieldnames(l).'
                testCase.verifyEqual(p.(f{1}), l.(f{1}), f{1});
            end
        end

        function model_workspace_holds_the_table(testCase)
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            a = gs3dx_anthropometry(ws.getVariable('GolferBodyMass'));
            expect = a.vars;
            for f = fieldnames(a.legs).'
                expect.(f{1}) = a.legs.(f{1});
            end
            for f = fieldnames(expect).'
                testCase.verifyEqual(ws.getVariable(f{1}), expect.(f{1}), 'RelTol', 1e-12, f{1});
            end
        end

        function upper_body_masses_come_from_the_table(testCase)
            % Every massive solid outside the club, grip parts and legs is
            % mass-based in kg on a Golfer* variable: no literal, no lbm.
            solids = find_system(testCase.mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
                'Regexp', 'on', 'ReferenceBlock', 'sm_lib/Body Elements/.* Solid');
            rel = extractAfter(solids, [testCase.mdl '/']);
            body = ~startsWith(rel, {'Club/', 'Lower Body/'}) & ...
                ~(startsWith(rel, 'Grip/') & ~endsWith(rel, {'/LHand', '/RHand'}));
            checked = 0;
            for k = reshape(find(body), 1, [])
                blk = solids{k};
                if strcmp(get_param(blk, 'BasedOnType'), 'Density') && str2double(get_param(blk, 'Density')) == 0
                    continue;   % massless joint and marker solids
                end
                if strcmp(get_param(blk, 'BasedOnType'), 'Mass') && str2double(get_param(blk, 'Mass')) == 0
                    continue;
                end
                testCase.verifyEqual(get_param(blk, 'BasedOnType'), 'Mass', rel{k});
                testCase.verifyEqual(get_param(blk, 'MassUnits'), 'kg', rel{k});
                testCase.verifySubstring(get_param(blk, 'Mass'), 'Golfer', rel{k});
                checked = checked + 1;
            end
            testCase.verifyEqual(checked, 15, 'head, neck, 3 trunk, 2 shoulder, 2 upper arm, 4 forearm, 2 hand');
        end

        function mass_edit_adds_no_blocks(testCase)
            src = char(gs3dx_names().variants.contact);
            load_system(src);
            testCase.addTeardown(@() close_system(src, 0));
            count = @(m) numel(find_system(m, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
                'LookInsideSubsystemReference', 'on', 'Virtual', 'off'));
            testCase.verifyEqual(count(testCase.mdl), count(src));
        end

        function golfer_weighs_the_body_mass_and_stands_from_rest(testCase)
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            M = ws.getVariable('GolferBodyMass');
            c = gs3dx_contact_check(testCase.info, rest=true, model=testCase.mdl);
            testCase.verifyEqual(c.status, "success");
            testCase.verifyEqual(c.mass, M + testCase.EquipmentMass, 'AbsTol', 1e-3, ...
                'inertia-sensed whole-body mass');
            testCase.verifyLessThanOrEqual(c.newton.max, c.newton.bound, ...
                sprintf('Newton residual %.3g N*s over bound %.3g', c.newton.max, c.newton.bound));
            for P = 'LR'
                testCase.verifyLessThan(c.feet.(P).slip, 5e-3, [P ' foot slides']);
                testCase.verifyLessThan(c.feet.(P).lift, 1e-3, [P ' foot lifts']);
            end
            testCase.verifyGreaterThan(c.support(2), 0.9, 'the ground carries the body weight');
        end
    end
end
