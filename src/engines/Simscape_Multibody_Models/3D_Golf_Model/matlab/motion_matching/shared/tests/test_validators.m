classdef test_validators < matlab.unittest.TestCase
%TEST_VALIDATORS  Unit tests for the shared/+validators package.

    methods (Test)
        function test_mustHaveFields_passes_when_present(testCase)
            s = struct("a", 1, "b", 2, "c", 3);
            % Should not error.
            validators.mustHaveFields(s, ["a", "b"]);
            testCase.verifyTrue(true);
        end

        function test_mustHaveFields_errors_listing_missing(testCase)
            s = struct("a", 1);
            try
                validators.mustHaveFields(s, ["a", "b", "c"]);
                testCase.verifyFail("expected validator:missingField error");
            catch err
                testCase.verifyEqual(err.identifier, "validator:missingField");
                testCase.verifySubstring(err.message, "b");
                testCase.verifySubstring(err.message, "c");
            end
        end

        function test_mustBeFiniteVector_accepts_finite(testCase)
            validators.mustBeFiniteVector([1; 2; 3.5]);
            validators.mustBeFiniteVector([0, -1, 2]);
            testCase.verifyTrue(true);
        end

        function test_mustBeFiniteVector_rejects_nan(testCase)
            testCase.verifyError( ...
                @() validators.mustBeFiniteVector([1; NaN; 3]), ...
                "validator:notFinite");
        end

        function test_mustBeFiniteVector_rejects_inf(testCase)
            testCase.verifyError( ...
                @() validators.mustBeFiniteVector([1, Inf, 2]), ...
                "validator:notFinite");
        end

        function test_mustBeFiniteVector_rejects_matrix(testCase)
            testCase.verifyError( ...
                @() validators.mustBeFiniteVector(ones(3, 2)), ...
                "validator:notVector");
        end

        function test_mustBeUnitQuaternionRows_tolerance(testCase)
            % Exact unit quaternions pass.
            Q = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
            validators.mustBeUnitQuaternionRows(Q);
            % Within default 1e-6 tolerance: pass.
            Qok = Q;
            Qok(1, 1) = 1 + 5e-7;
            validators.mustBeUnitQuaternionRows(Qok);
            % Outside tolerance: error.
            Qbad = Q;
            Qbad(1, 1) = 1.01;
            testCase.verifyError( ...
                @() validators.mustBeUnitQuaternionRows(Qbad), ...
                "validator:notUnitNorm");
        end

        function test_mustBeUnitQuaternionRows_rejects_wrong_shape(testCase)
            testCase.verifyError( ...
                @() validators.mustBeUnitQuaternionRows(ones(3, 3)), ...
                "validator:badShape");
        end

        function test_mustBeMonotonicTime_rejects_non_monotonic(testCase)
            validators.mustBeMonotonicTime(linspace(0, 1, 10));
            testCase.verifyError( ...
                @() validators.mustBeMonotonicTime([0, 0.1, 0.1, 0.2]), ...
                "validator:notMonotonic");
            testCase.verifyError( ...
                @() validators.mustBeMonotonicTime([0, 0.2, 0.1]), ...
                "validator:notMonotonic");
        end

        function test_mustBeMonotonicTime_rejects_nan(testCase)
            testCase.verifyError( ...
                @() validators.mustBeMonotonicTime([0, NaN, 1]), ...
                "validator:notFinite");
        end
    end
end
