classdef test_gs3dx_leg_ik_contract < matlab.unittest.TestCase
    % True pose acceptance, including the former 180-degree false success.
    methods (TestClassSetup)
        function setup(~)
            gs3dx_setup();
        end
    end
    methods (Test)
        function opposite_orientation_never_returns_false_success(t)
            g=local_geom();q0=[5;-20;3;-40;2;-18];pp=[0;0;1];
            [R,p]=gs3dx_leg_fk(g,eye(3),pp,q0);target=R*diag([1,-1,-1]);
            try
                q=gs3dx_leg_ik(g,eye(3),pp,target,p,q0);
            catch err
                t.verifyEqual(err.identifier,'gs3dx:ik');return;
            end
            [actual_R,actual_p]=gs3dx_leg_fk(g,eye(3),pp,q);
            t.verifyEqual(actual_R,target,'AbsTol',1e-9);
            t.verifyEqual(actual_p,p,'AbsTol',1e-9);
        end
        function reachable_target_and_continuous_trajectory(t)
            g=local_geom();q0=[0;-15;0;-30;0;-15];
            [R,p]=gs3dx_leg_fk(g,eye(3),[0;0;1],q0);
            pelvis_R=repmat(eye(3),1,1,5);
            pelvis_p=[0;0;1]+[linspace(0,.02,5);zeros(1,5);linspace(0,-.02,5)];
            [q,res]=gs3dx_leg_ik(g,pelvis_R,pelvis_p,R,p,q0+[3;-4;2;5;-2;3]);
            t.verifySize(q,[6,5]);t.verifyLessThan(max(res),1e-9);
            for k=1:5
                [actual_R,actual_p]=gs3dx_leg_fk(g,pelvis_R(:,:,k),pelvis_p(:,k),q(:,k));
                t.verifyEqual(actual_R,R,'AbsTol',1e-9);
                t.verifyEqual(actual_p,p,'AbsTol',1e-9);
            end
            t.verifyLessThan(q(4,:),0);
        end
        function rejects_reflections_and_nonorthogonal_rotations(t)
            g=local_geom();
            for R={diag([1,1,-1]),eye(3)+.1*ones(3)}
                t.verifyError(@() gs3dx_leg_ik(g,eye(3),[0;0;1],R{1},zeros(3,1),zeros(6,1)), ...
                    'gs3dx:orientation_residual:invalid_rotation');
                t.verifyError(@() gs3dx_leg_ik(g,R{1},[0;0;1],eye(3),zeros(3,1),zeros(6,1)), ...
                    'gs3dx:orientation_residual:invalid_rotation');
            end
        end
        function rejects_nonfinite_geometry(t)
            g=local_geom();g.mount_R(1,1)=NaN;
            t.verifyError(@() gs3dx_leg_ik(g,eye(3),[0;0;1],eye(3),zeros(3,1),zeros(6,1)), ...
                'gs3dx:ik:invalid_geometry');
        end
        function rejects_vector_and_missing_geometry(t)
            g=local_geom();g.thigh=[.4,.5];
            t.verifyError(@() gs3dx_leg_ik(g,eye(3),[0;0;1],eye(3),zeros(3,1),zeros(6,1)), ...
                'gs3dx:ik:invalid_geometry');
            g=rmfield(local_geom(),'shank');
            t.verifyError(@() gs3dx_leg_ik(g,eye(3),[0;0;1],eye(3),zeros(3,1),zeros(6,1)), ...
                'gs3dx:ik:invalid_geometry');
        end
        function rejects_mismatched_pelvis_sequence(t)
            t.verifyError(@() gs3dx_leg_ik(local_geom(),repmat(eye(3),1,1,2),[0;0;1], ...
                eye(3),zeros(3,1),zeros(6,1)),'gs3dx:ik:invalid_dimensions');
        end
    end
end
function g=local_geom()
    g=struct('mount_R',eye(3),'mount_p',[0;.09;-.1],'thigh',.45,'shank',.43);
end
