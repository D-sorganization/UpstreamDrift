classdef test_gs3dx_capture_com_reference < matlab.unittest.TestCase
%TEST_GS3DX_CAPTURE_COM_REFERENCE  Resampling and alignment of the capture centre of mass (#10979).

    methods (TestClassSetup)
        function addPaths(~)
            addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'tools'));
        end
    end

    methods (Test)
        function starts_at_com0_and_keeps_the_motion(testCase)
            k.t = 0.5 + (0:100) / 100;
            k.com = [sin(k.t); cos(k.t); k.t .^ 2];
            t = 0:0.0137:0.9;
            com0 = [1; -2; 0.9];
            com = gs3dx_capture_com_reference(k, t, com0);
            testCase.verifySize(com, [3 numel(t)]);
            testCase.verifyEqual(com(:, 1), com0, 'AbsTol', 1e-15);
            expected = [sin(t + 0.5); cos(t + 0.5); (t + 0.5) .^ 2];
            expected = expected - expected(:, 1) + com0;
            % linear interpolation of 10 ms samples: error <= h^2 max|f''| / 8 = 2.5e-5, twice after the shift to com0
            testCase.verifyEqual(com, expected, 'AbsTol', 6e-5);
        end

        function rejects_times_outside_the_capture(testCase)
            k.t = 0:0.01:1;
            k.com = zeros(3, numel(k.t));
            testCase.verifyError(@() gs3dx_capture_com_reference(k, 0:0.1:1.2, zeros(3, 1)), 'gs3dx:capturecom');
            k.com = zeros(2, numel(k.t));
            testCase.verifyError(@() gs3dx_capture_com_reference(k, 0:0.1:1, zeros(3, 1)), 'gs3dx:capturecom');
        end
    end
end
