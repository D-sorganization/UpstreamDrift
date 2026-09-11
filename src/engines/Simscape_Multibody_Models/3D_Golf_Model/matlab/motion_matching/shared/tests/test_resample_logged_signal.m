classdef test_resample_logged_signal < matlab.unittest.TestCase
    methods (Test)
        function usesPhysicalTimeEvenForEqualRowCounts(testCase)
            signal = timeseries([0; 1; 10], [0; 0.1; 1]);
            actual = resample_logged_signal(signal, [0; 0.5; 1], 1, []);
            testCase.verifyEqual(actual, [0; 5; 10], 'AbsTol', 1e-12);
        end

        function numericArraysRequireMatchingSolverClock(testCase)
            actual = resample_logged_signal([0; 1; 10], [0; 0.5; 1], 1, [0; 0.1; 1]);
            testCase.verifyEqual(actual, [0; 5; 10], 'AbsTol', 1e-12);
            testCase.verifyError(@() resample_logged_signal([0; 1; 10], ...
                [0; 0.5; 1], 1, []), 'resample_logged_signal:missingClock');
        end

        function neverExtrapolatesEarlyStoppedMotion(testCase)
            actual = resample_logged_signal(timeseries([0; 1], [0; 0.1]), ...
                [0; 0.05; 0.1; 1], 1, []);
            testCase.verifyEqual(actual(1:3), [0; 0.5; 1], 'AbsTol', 1e-12);
            testCase.verifyTrue(isnan(actual(4)));
        end

        function singleSampleDoesNotInventConstantMotion(testCase)
            actual = resample_logged_signal(timeseries([1 2 3], 0), [0; 1], 3, []);
            testCase.verifyEqual(actual(1,:), [1 2 3]);
            testCase.verifyTrue(all(isnan(actual(2,:))));
        end

        function supportsStructureWithTime(testCase)
            signal = struct('time', [0; 0.1; 1], ...
                'signals', struct('values', [0 0 0; 1 2 3; 10 20 30]));
            actual = resample_logged_signal(signal, [0; 0.5; 1], 3, []);
            testCase.verifyEqual(actual(2,:), [5 10 15], 'AbsTol', 1e-12);
        end

        function rejectsAmbiguousClockAndWrongDimensions(testCase)
            testCase.verifyError(@() resample_logged_signal([1; 2; 3], ...
                [0; 1], 1, [0; 0; 1]), 'resample_logged_signal:badClock');
            testCase.verifyError(@() resample_logged_signal(ones(2,4), ...
                [0; 1], 3, [0; 1]), 'resample_logged_signal:badShape');
        end

        function ownClockOverridesSolverClock(testCase)
            signal = timeseries([0; 10], [0; 1]);
            actual = resample_logged_signal(signal, [0; 0.5; 1], 1, [0; 0.1]);
            testCase.verifyEqual(actual, [0; 5; 10], 'AbsTol', 1e-12);
        end

        function supportsTimeLastVectorTimeseries(testCase)
            signal = timeseries(reshape([0 0 0 10 20 30], [3 1 2]), [0; 1]);
            actual = resample_logged_signal(signal, [0; 0.5; 1], 3, []);
            testCase.verifyEqual(actual(2,:), [5 10 15], 'AbsTol', 1e-12);
        end
    end
end
