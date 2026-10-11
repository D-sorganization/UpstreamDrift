classdef test_gs3dx_scapula_phase < matlab.unittest.TestCase
    methods (TestClassSetup)
        function tools(tc)
            root=fileparts(fileparts(mfilename('fullpath')));
            if isfolder(fullfile(root,'tools'))
                tc.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root,'tools')));
            end
        end
    end
    methods (Test)
        function addressBackswingAndRelease(tc)
            b=struct('active',true,'lower',[5;-25]*pi/180,'upper',[25;-5]*pi/180, ...
                'address_lower',[5;-10]*pi/180,'address_upper',[10;-5]*pi/180);
            [lo,hi,pull]=gs3dx_scapula_phase(b,1,100,200);
            tc.verifyEqual(lo,b.address_lower);tc.verifyEqual(hi,b.address_upper);tc.verifyEqual(pull,1);
            [lo,hi,pull]=gs3dx_scapula_phase(b,100,100,200);
            tc.verifyEqual(lo,b.lower);tc.verifyEqual(hi,b.upper);tc.verifyEqual(pull,1);
            [lo,hi,pull]=gs3dx_scapula_phase(b,150,100,200);
            tc.verifyEqual(pull,.5,'AbsTol',1e-12);
            tc.verifyTrue(all(lo<b.lower));tc.verifyTrue(all(hi>b.upper));
            [lo,hi,pull]=gs3dx_scapula_phase(b,200,100,200);
            tc.verifyEmpty(lo);tc.verifyEmpty(hi);tc.verifyEqual(pull,0);
        end
        function badPhaseOrDisabled(tc)
            b=struct('active',false);
            [lo,hi,pull]=gs3dx_scapula_phase(b,1,0,0);
            tc.verifyEmpty(lo);tc.verifyEmpty(hi);tc.verifyEqual(pull,0);
            b.active=true;
            tc.verifyError(@() gs3dx_scapula_phase(b,1,200,100),'gs3dx:ik:scapula');
        end
    end
end
