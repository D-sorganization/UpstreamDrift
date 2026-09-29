classdef test_intrinsic_xyz_to_rotm < matlab.unittest.TestCase
    methods (Test)
        function mapsIntrinsicAxesInOrder(testCase)
            actual = intrinsic_xyz_to_rotm([pi/2 0 0; pi/2 pi/2 0]);
            testCase.verifyEqual(actual(:,:,1), [1 0 0;0 0 -1;0 1 0], 'AbsTol', 1e-14);
            testCase.verifyEqual(actual(:,:,2), [0 0 1;1 0 0;0 1 0], 'AbsTol', 1e-14);
        end
        function rejectsMissingOrientation(testCase)
            testCase.verifyError(@() intrinsic_xyz_to_rotm([NaN 0 0]), ...
                'MATLAB:validators:mustBeFinite');
        end
    end
end
