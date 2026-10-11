classdef test_gs3dx_render_output_options < matlab.unittest.TestCase
    methods (TestClassSetup)
        function tools(tc)
            root=fileparts(fileparts(mfilename('fullpath')));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root,'tools')));
        end
    end
    methods (Test)
        function rejectsUnsupportedDimensionsBeforeModelLoad(tc)
            tc.verifyError(@() gs3dx_render('model_must_not_load',resolution=[1919 1080]),'gs3dx:render');
            tc.verifyError(@() gs3dx_render('model_must_not_load',resolution=[32 32]),'gs3dx:render');
        end
        function rejectsInvalidQualityBeforeModelLoad(tc)
            tc.verifyError(@() gs3dx_render('model_must_not_load',video_quality=101),'gs3dx:render');
        end
    end
end
