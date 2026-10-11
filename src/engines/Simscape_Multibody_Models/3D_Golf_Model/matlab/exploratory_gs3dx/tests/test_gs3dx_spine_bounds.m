classdef test_gs3dx_spine_bounds < matlab.unittest.TestCase
    methods (TestClassSetup)
        function tools(tc)
            root = fileparts(fileparts(mfilename('fullpath')));
            if isfolder(fullfile(root, 'tools'))
                tc.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root, 'tools')));
            end
        end
    end

    methods (Test)
        function disabledWhenBothZero(tc)
            [jp, layout, ref] = fixture();
            b = gs3dx_spine_bounds(jp, layout, ref, 0, 0);
            tc.verifyFalse(b.active);
            tc.verifyEmpty(b.indices);
            tc.verifyEmpty(b.keys);
            tc.verifyEmpty(b.reference);
            tc.verifyEmpty(b.lower);
            tc.verifyEmpty(b.upper);

            % Disabled even with empty inputs
            b2 = gs3dx_spine_bounds(table(), struct([]), [], 0, 0);
            tc.verifyFalse(b2.active);
            tc.verifyEmpty(b2.lower);
        end

        function numericEndpointsAndBounds(tc)
            [jp, layout, ref] = fixture();
            bend = 15;
            twist = 25;
            b = gs3dx_spine_bounds(jp, layout, ref, bend, twist);

            tc.verifyTrue(b.active);
            tc.verifyEqual(b.indices, [2; 3; 6]);
            tc.verifyEqual(b.keys, ["j77.Rx"; "j77.Ry"; "j12.Rz"]);
            tc.verifyEqual(b.reference, ref(:));

            % Check resolved bound values
            b_rad = deg2rad(bend);
            t_rad = deg2rad(twist);
            tc.verifyEqual(b.lower(2), ref(2) - b_rad, 'AbsTol', 1e-12);
            tc.verifyEqual(b.upper(2), ref(2) + b_rad, 'AbsTol', 1e-12);
            tc.verifyEqual(b.lower(3), ref(3) - b_rad, 'AbsTol', 1e-12);
            tc.verifyEqual(b.upper(3), ref(3) + b_rad, 'AbsTol', 1e-12);
            tc.verifyEqual(b.lower(6), ref(6) - t_rad, 'AbsTol', 1e-12);
            tc.verifyEqual(b.upper(6), ref(6) + t_rad, 'AbsTol', 1e-12);

            % Other coordinates must remain +/-Inf
            unbounded = [1, 4, 5];
            tc.verifyTrue(all(b.lower(unbounded) == -inf));
            tc.verifyTrue(all(b.upper(unbounded) == inf));
        end

        function selectiveZeroLimit(tc)
            [jp, layout, ref] = fixture();
            % Bend 0, twist > 0: bend unconstrained, twist constrained
            b1 = gs3dx_spine_bounds(jp, layout, ref, 0, 20);
            tc.verifyTrue(b1.active);
            tc.verifyTrue(isinf(b1.lower(2)) && isinf(b1.upper(2)));
            tc.verifyTrue(isinf(b1.lower(3)) && isinf(b1.upper(3)));
            tc.verifyEqual(b1.lower(6), ref(6) - deg2rad(20), 'AbsTol', 1e-12);
            tc.verifyEqual(b1.upper(6), ref(6) + deg2rad(20), 'AbsTol', 1e-12);

            % Bend > 0, twist 0: bend constrained, twist unconstrained
            b2 = gs3dx_spine_bounds(jp, layout, ref, 10, 0);
            tc.verifyTrue(b2.active);
            tc.verifyEqual(b2.lower(2), ref(2) - deg2rad(10), 'AbsTol', 1e-12);
            tc.verifyEqual(b2.upper(2), ref(2) + deg2rad(10), 'AbsTol', 1e-12);
            tc.verifyTrue(isinf(b2.lower(6)) && isinf(b2.upper(6)));
        end

        function invalidLimitsFail(tc)
            [jp, layout, ref] = fixture();
            invalid_vals = {-1, NaN, Inf, -Inf, 1i, [5, 5], "10"};
            for v = invalid_vals
                tc.verifyError(@() gs3dx_spine_bounds(jp, layout, ref, v{1}, 10), 'gs3dx:ik:spine');
                tc.verifyError(@() gs3dx_spine_bounds(jp, layout, ref, 10, v{1}), 'gs3dx:ik:spine');
            end
        end

        function referenceValidationFailures(tc)
            [jp, layout, ref] = fixture();
            % Wrong length
            tc.verifyError(@() gs3dx_spine_bounds(jp, layout, ref(1:end-1), 10, 10), 'gs3dx:ik:spine');
            tc.verifyError(@() gs3dx_spine_bounds(jp, layout, [ref; 0], 10, 10), 'gs3dx:ik:spine');
            % Nonfinite reference
            ref_nan = ref; ref_nan(2) = NaN;
            tc.verifyError(@() gs3dx_spine_bounds(jp, layout, ref_nan, 10, 10), 'gs3dx:ik:spine');
            ref_inf = ref; ref_inf(6) = Inf;
            tc.verifyError(@() gs3dx_spine_bounds(jp, layout, ref_inf, 10, 10), 'gs3dx:ik:spine');
            % Complex reference
            ref_c = ref; ref_c(1) = 1i;
            tc.verifyError(@() gs3dx_spine_bounds(jp, layout, ref_c, 10, 10), 'gs3dx:ik:spine');
        end

        function missingOrDuplicateAnatomyFails(tc)
            [jp, layout, ref] = fixture();
            % Missing Torso Rz
            jp_missing = jp(1:3, :);
            tc.verifyError(@() gs3dx_spine_bounds(jp_missing, layout, ref, 10, 10), 'gs3dx:ik:spine');

            % Duplicate Spine Tilt Rx
            jp_dup = [jp; jp(1, :)];
            tc.verifyError(@() gs3dx_spine_bounds(jp_dup, layout, ref, 10, 10), 'gs3dx:ik:spine');
        end

        function nonDegreeUnitFails(tc)
            [jp, layout, ref] = fixture();
            jp.Unit(1) = "rad";
            tc.verifyError(@() gs3dx_spine_bounds(jp, layout, ref, 10, 10), 'gs3dx:ik:spine');

            [jp2, layout2, ref2] = fixture();
            jp2.Unit(5) = "m";
            tc.verifyError(@() gs3dx_spine_bounds(jp2, layout2, ref2, 10, 10), 'gs3dx:ik:spine');
        end

        function multiDofLayoutFails(tc)
            [jp, layout, ref] = fixture();
            % Make j77.Rx layout multi-DOF (n > 1)
            layout(2).n = 2;
            ref_expanded = [ref; 0];
            tc.verifyError(@() gs3dx_spine_bounds(jp, layout, ref_expanded, 10, 10), 'gs3dx:ik:spine');
        end

        function malformedLayoutOrTableFails(tc)
            [jp, layout, ref] = fixture();
            % Table missing column
            tc.verifyError(@() gs3dx_spine_bounds(jp(:, {'ID', 'BlockPath'}), layout, ref, 10, 10), 'gs3dx:ik:spine');
            % Layout missing coordinate key
            layout_missing = layout(1:4);
            ref_trunc = ref(1:4);
            tc.verifyError(@() gs3dx_spine_bounds(jp, layout_missing, ref_trunc, 10, 10), 'gs3dx:ik:spine');
        end
    end
end

function [jp, layout, ref] = fixture()
    % Synthetic renumbered IDs and prefix-sum layout
    BlockPath = [
        "Rig/Subsystem/Spine Tilt Joint/P";
        "Rig/Subsystem/Spine Tilt Joint/P";
        "Rig/Subsystem/Spine Tilt Joint/P";
        "Rig/Torso Kinetically Driven Joint/U";
        "Rig/Torso Kinetically Driven Joint/U"
    ];
    ID = [
        "j77.Rx.q";
        "j77.Ry.q";
        "j77.Rz.q";
        "j12.Ry.q";
        "j12.Rz.q"
    ];
    Unit = repmat("deg", 5, 1);
    jp = table(BlockPath, ID, Unit);

    % Independent layout ordering with synthetic prefix offsets
    % Total coordinates = 1 + 1 + 1 + 1 + 1 + 1 = 6
    layout = struct( ...
        'key', {"j100.Px", "j77.Rx", "j77.Ry", "j77.Rz", "j12.Ry", "j12.Rz"}, ...
        'n',   {1,         1,        1,        1,        1,        1} ...
    );
    ref = [0.1; 0.2; -0.15; 0.05; -0.3; 0.45];
end
