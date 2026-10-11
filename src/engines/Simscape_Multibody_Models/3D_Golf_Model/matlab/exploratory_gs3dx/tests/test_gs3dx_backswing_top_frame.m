classdef test_gs3dx_backswing_top_frame < matlab.unittest.TestCase
    methods (TestClassSetup)
        function tools(tc)
            root=fileparts(fileparts(mfilename('fullpath')));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root,'tools')));
        end
    end
    methods (Test)
        function knownProxyAndGapMask(tc)
            jc=fixture();
            tc.verifyEqual(gs3dx_backswing_top_frame(jc),3);
            jc.gap.pelvis(3)=true;
            tc.verifyEqual(gs3dx_backswing_top_frame(jc,true),2);
            jc.gap.pelvis(:)=true;
            tc.verifyError(@() gs3dx_backswing_top_frame(jc,true),'gs3dx:ik:scapula');
        end
        function missingDataNeverInventsStrictEvent(tc)
            jc=struct('impact_frame',10);
            tc.verifyEqual(gs3dx_backswing_top_frame(jc),7);
            tc.verifyError(@() gs3dx_backswing_top_frame(jc,true),'gs3dx:ik:scapula');
            jc=fixture();jc.impact_frame=NaN;
            tc.verifyError(@() gs3dx_backswing_top_frame(jc,true),'gs3dx:ik:scapula');
        end
    end
end
function jc=fixture()
    yaw=[0,-.2,-.4,-.1,0];R=zeros(3,3,5);
    for i=1:5
        a=yaw(i);R(:,:,i)=[cos(a),-sin(a),0;sin(a),cos(a),0;0,0,1];
    end
    jc=struct('impact_frame',5,'pelvis_R',R,'gap',struct('pelvis',false(1,5)));
end
