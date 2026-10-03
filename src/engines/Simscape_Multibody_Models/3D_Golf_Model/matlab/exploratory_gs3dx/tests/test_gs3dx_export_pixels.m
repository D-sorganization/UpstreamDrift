classdef test_gs3dx_export_pixels < matlab.unittest.TestCase
    methods (TestClassSetup)
        function tools(tc)
            root=fileparts(fileparts(mfilename('fullpath')));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root,'tools')));
        end
    end
    methods (Test)
        function preservesRoundShapeFromPortraitCanvas(tc)
            check_circle(tc,[400 900]);
        end
        function preservesRoundShapeFromLandscapeCanvas(tc)
            check_circle(tc,[640 480]);
        end
    end
end
function check_circle(tc,resolution)
    fig=figure('Visible','off','Color','w','Units','pixels','Position',[100 100 resolution]);
    cleanup=onCleanup(@() close(fig));
    ax=axes('Parent',fig);rectangle(ax,'Position',[-.5 -.5 1 1],'Curvature',[1 1],'FaceColor','k','EdgeColor','none');
    axis(ax,'equal');xlim(ax,[-1 1]);ylim(ax,[-1 1]);axis(ax,'off');
    file=[tempname '.png'];file_cleanup=onCleanup(@() delete(file));
    pixels=gs3dx_export_pixels(fig,file,[1920 1080]);
    tc.verifySize(pixels,[1080 1920 3]);
    [rows,cols]=find(all(pixels<20,3));
    tc.verifyGreaterThan(numel(rows),100);
    tc.verifyLessThanOrEqual(abs((max(rows)-min(rows))-(max(cols)-min(cols))),2);
end
