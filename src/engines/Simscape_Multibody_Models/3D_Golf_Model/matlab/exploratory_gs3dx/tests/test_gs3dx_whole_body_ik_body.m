classdef test_gs3dx_whole_body_ik_body < matlab.unittest.TestCase
    methods (TestClassSetup)
        function tools(tc)
            root=fileparts(fileparts(mfilename('fullpath')));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root,'tools')));
        end
    end
    methods (Test)
        function missingRawBackPointsFailsBeforeModelSetup(tc)
            tc.verifyError(@() gs3dx_whole_body_ik(fixture(),model='must_not_load', ...
                scapula_protraction_deg=0,back_marker_weight=.5),'gs3dx:ik:body');
        end
        function wrongRawBackShapeFailsBeforeModelSetup(tc)
            jc=fixture();jc.back_marker_points=zeros(3,2,5);
            tc.verifyError(@() gs3dx_whole_body_ik(jc,model='must_not_load', ...
                scapula_protraction_deg=0,back_marker_weight=.5),'gs3dx:ik:body');
        end
        function nonfiniteTorsoOffsetsFailBeforeModelSetup(tc)
            jc=fixture();jc.back_marker_points=zeros(3,3,5);off=zeros(3);off(1)=NaN;
            tc.verifyError(@() gs3dx_whole_body_ik(jc,model='must_not_load', ...
                scapula_protraction_deg=0,back_marker_weight=.5,back_marker_offsets=off),'gs3dx:ik:body');
        end
        function spineRequiresMeasuredPhaseBeforeModelSetup(tc)
            tc.verifyError(@() gs3dx_whole_body_ik(fixture(),model='must_not_load', ...
                scapula_protraction_deg=0,spine_bend_excursion_deg=15),'gs3dx:ik:scapula');
        end
    end
end
function jc=fixture()
    jc=struct('pelvis',zeros(3,5),'t',0:.01:.04,'gap',struct());
    names=gs3dx_ik_target_names();
    for name=string(names(:)).'
        jc.(name)=zeros(3,5);jc.gap.(name)=false(1,5);
    end
end

