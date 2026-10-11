classdef test_gs3dx_scapula_bounds < matlab.unittest.TestCase
    methods (TestClassSetup)
        function tools(tc)
            root=fileparts(fileparts(mfilename('fullpath')));
            if isfolder(fullfile(root,'tools'))
                tc.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root,'tools')));
            end
        end
    end
    methods (Test)
        function mirroredAndRenumbered(tc)
            [jp,layout]=fixture();
            b=gs3dx_scapula_bounds(jp,layout,7.5);
            tc.verifyEqual(b.indices,[1;3]);
            tc.verifyEqual(rad2deg(b.address_lower([1,3])),[5;-10],'AbsTol',1e-12);
            tc.verifyEqual(rad2deg(b.address_upper([1,3])),[10;-5],'AbsTol',1e-12);
            tc.verifyEqual(rad2deg(b.lower([1,3])),[5;-25],'AbsTol',1e-12);
            tc.verifyEqual(rad2deg(b.upper([1,3])),[25;-5],'AbsTol',1e-12);
            tc.verifyTrue(all(isinf(b.lower([2,4]))));
            tc.verifyEqual(rad2deg(b.target([1,3])),[7.5;-7.5],'AbsTol',1e-12);
        end
        function disabledPreservesUnboundedSolver(tc)
            b=gs3dx_scapula_bounds(table(),struct([]),0);
            tc.verifyFalse(b.active);
        end
        function badInputFailsClosed(tc)
            [jp,layout]=fixture();
            for value={NaN,Inf,-1,4.9,10.1,1i}
                tc.verifyError(@() gs3dx_scapula_bounds(jp,layout,value{1}),'gs3dx:ik:scapula');
            end
            jp.Unit(3)="m";
            tc.verifyError(@() gs3dx_scapula_bounds(jp,layout,7.5),'gs3dx:ik:scapula');
        end
        function missingOrDuplicateAnatomyFails(tc)
            [jp,layout]=fixture();
            tc.verifyError(@() gs3dx_scapula_bounds(jp(1:2,:),layout,7.5),'gs3dx:ik:scapula');
            tc.verifyError(@() gs3dx_scapula_bounds([jp;jp(1,:)],layout,7.5),'gs3dx:ik:scapula');
        end
        function endpointsAndSchema(tc)
            [jp,layout]=fixture();
            for value=[5,10]
                b=gs3dx_scapula_bounds(jp,layout,value);
                tc.verifyEqual(rad2deg(b.target(b.indices)),[value;-value],'AbsTol',1e-12);
            end
            tc.verifyError(@() gs3dx_scapula_bounds(jp(:,1:2),layout,7.5),'gs3dx:ik:scapula');
            tc.verifyError(@() gs3dx_scapula_bounds(jp,layout(2:end),7.5),'gs3dx:ik:scapula');
            layout(1).n=3;
            tc.verifyError(@() gs3dx_scapula_bounds(jp,layout,7.5),'gs3dx:ik:scapula');
        end
    end
end
function [jp,layout]=fixture()
    BlockPath=["X/Left Scapula Joint/U";"X/Left Scapula Joint/U";"X/Right Scapula Joint/U";"X/Right Scapula Joint/U"];
    ID=["j91.Rx.q";"j91.Ry.q";"j4.Rx.q";"j4.Ry.q"];
    Unit=repmat("deg",4,1);jp=table(BlockPath,ID,Unit);
    layout=struct('key',{"j91.Rx","j91.Ry","j4.Rx","j4.Ry"},'n',{1,1,1,1});
end
