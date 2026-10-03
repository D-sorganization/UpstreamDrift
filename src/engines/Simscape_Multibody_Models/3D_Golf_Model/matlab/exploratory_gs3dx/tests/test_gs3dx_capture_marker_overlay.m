classdef test_gs3dx_capture_marker_overlay < matlab.unittest.TestCase
    methods (TestClassSetup)
        function tools(tc)
            root=fileparts(fileparts(mfilename('fullpath')));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root,'tools')));
        end
    end
    methods (Test)
        function measuredChannelsAndAddressFrame(tc)
            cap=fixture();
            [p,info]=gs3dx_capture_marker_overlay(cap,[3 1]);
            % Independent rotation and fixed address waist origin oracle.
            expected=reshape(cap.target_frame.'*(reshape(cap.points(:,:, [3 1]),3,[])-[1;2;3]),3,5,2);
            expected(:,5,1)=NaN;
            tc.verifyEqual(p,expected,'AbsTol',1e-12);
            tc.verifyEqual(info.labels,cap.labels);
            tc.verifyEqual(info.channel_count,5);
            tc.verifyEqual(info.visible_counts,[4 5]);
            tc.verifyTrue(all(isnan(p(:,5,1))));
        end
        function invalidIndicesAndResidualMask(tc)
            cap=fixture();
            tc.verifyError(@() gs3dx_capture_marker_overlay(cap,[0 1]),'gs3dx:marker_overlay');
            tc.verifyError(@() gs3dx_capture_marker_overlay(cap,[1.5]),'gs3dx:marker_overlay');
            cap.missing_mask(1,2,1)=true;
            [p,info]=gs3dx_capture_marker_overlay(cap,1);
            tc.verifyTrue(all(isnan(p(:,2,1))));
            tc.verifyEqual(info.visible_counts,4);
        end
    end
end
function cap=fixture()
    cap.labels=["WaistLeft","WaistRight","WaistLBack","WaistRBack","Extra"];
    cap.points=repmat([1;2;3],1,5,3);
    cap.points(:,5,:)=reshape([2 3 4;4 5 6;6 7 8],3,1,3);
    cap.target_frame=[0 -1 0;1 0 0;0 0 1];
    cap.missing_mask=false(1,5,3);cap.missing_mask(1,5,3)=true;
    pts=cap.points;labels=cap.labels;
    cap.marker=@(name) reshape(pts(:,labels==name,:),3,[]);
end
