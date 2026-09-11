classdef test_select_capture_marker_frame < matlab.unittest.TestCase
    methods(Test)
        function usesValidityAndPreservesRequestedOrder(testCase)
            capture=struct('labels',{{'missing','origin','moving'}}, ...
                'time_s',[0;1],'points_world_m',zeros(2,3,3), ...
                'valid',logical([0 1 1;1 1 1]));
            capture.points_world_m(1,3,:)=[1 2 3];
            [points,indices]=select_capture_marker_frame(capture,1,["moving";"missing";"origin"]);
            testCase.verifyEqual(points,[1 2 3;0 0 0]);
            testCase.verifyEqual(indices,[1;3]);
        end
        function rejectsInvalidSelections(testCase)
            capture=struct('labels',{{'a'}},'time_s',0, ...
                'points_world_m',zeros(1,1,3),'valid',false);
            testCase.verifyError(@()select_capture_marker_frame(capture,1,"a"), ...
                'select_capture_marker_frame:noObservations');
            testCase.verifyError(@()select_capture_marker_frame(capture,1,"unknown"), ...
                'select_capture_marker_frame:labels');
            testCase.verifyError(@()select_capture_marker_frame(capture,2,"a"), ...
                'select_capture_marker_frame:range');
            capture.valid=true;capture.points_world_m(1,1,1)=NaN;
            testCase.verifyError(@()select_capture_marker_frame(capture,1,"a"), ...
                'select_capture_marker_frame:nonfinite');
        end
    end
end
