function [q, residual] = gs3dx_leg_ik(geom, pelvis_R, pelvis_p, foot_R, foot_p, q0)
%GS3DX_LEG_IK  Leg joint angles placing the ankle follower frame on a target.
%
%   Angles [hip X Y Z, knee, ankle X Y] are degrees. Pelvis and target
%   positions are metres; all rotations are proper SO(3). Each trajectory
%   pose starts from the preceding solution, with Q0 seeding the first.
%   Damped Gauss-Newton uses a 12x6 finite-difference Jacobian: three
%   translation components and nine normalized chordal rotation components
%   from GS3DX_ORIENTATION_RESIDUAL. The mixed residual combines metres and
%   dimensionless chordal error; its rotational norm is 2*sin(theta/2),
%   approximately theta in radians near zero. It stays nonzero at pi.
%   Final acceptance requires position <1e-9 m and orientation <1e-9 rad,
%   evaluated through the equivalent chordal bound to avoid acos roundoff
%   near identity. Unconverged targets raise gs3dx:ik. This local numerical
%   solver does not establish anatomical limits or native contact support.
    arguments
        geom (1,1) struct
        pelvis_R (3,3,:) double {mustBeReal,mustBeFinite}
        pelvis_p (3,:) double {mustBeReal,mustBeFinite}
        foot_R (3,3) double {mustBeReal,mustBeFinite}
        foot_p (3,1) double {mustBeReal,mustBeFinite}
        q0 (6,1) double {mustBeReal,mustBeFinite}
    end
    n=size(pelvis_R,3);
    assert(n>=1 && size(pelvis_p,2)==n,'gs3dx:ik:invalid_dimensions', ...
        'Pelvis rotations and positions must have the same nonzero length');
    required={'mount_R','mount_p','thigh','shank'};
    for field=required
        assert(isfield(geom,field{1}),'gs3dx:ik:invalid_geometry', ...
            'Missing geometry field %s',field{1});
        value=geom.(field{1});
        assert(isnumeric(value) && isreal(value) && all(isfinite(value),'all'), ...
            'gs3dx:ik:invalid_geometry','Geometry field %s must be finite real numeric',field{1});
    end
    assert(isequal(size(geom.mount_R),[3,3]) && isequal(size(geom.mount_p),[3,1]) && ...
        isscalar(geom.thigh) && isscalar(geom.shank) && geom.thigh>0 && geom.shank>0, ...
        'gs3dx:ik:invalid_geometry','Geometry requires 3x3 mount_R, 3x1 mount_p and positive scalar lengths');
    % Reuse the shared SO(3) contract before any iterative computation.
    gs3dx_orientation_residual(pelvis_R,pelvis_R,[],1);
    gs3dx_orientation_residual(geom.mount_R,foot_R,false,1);
    q=zeros(6,n);residual=zeros(1,n);x=q0;
    for k=1:n
        f=@(v) local_error(geom,pelvis_R(:,:,k),pelvis_p(:,k),v,foot_R,foot_p);
        [x,residual(k)]=local_solve(f,x);
        e=f(x);
        assert(norm(e(1:3))<1e-9 && norm(e(4:end))<2*sin(1e-9/2), ...
            'gs3dx:ik','IK did not converge at pose %d of %d (residual %.3g): target out of reach?', ...
            k,n,residual(k));
        q(:,k)=x;
    end
end
function e=local_error(geom,pR,pp,v,foot_R,foot_p)
    [R,p]=gs3dx_leg_fk(geom,pR,pp,v);
    e=[p-foot_p;gs3dx_orientation_residual(R,foot_R,false,1)];
end
function [x,r]=local_solve(f,x)
    e=f(x);r=norm(e);lambda=1e-6;
    for it=1:100
        if r<1e-12,break;end
        J=zeros(numel(e),6);h=1e-6;
        for c=1:6
            d=zeros(6,1);d(c)=h;
            J(:,c)=(f(x+d)-f(x-d))/(2*h);
        end
        step=-(J.'*J+lambda*eye(6))\(J.'*e);
        trial=x+step;et=f(trial);
        if norm(et)<r
            x=trial;e=et;r=norm(e);lambda=max(lambda/10,1e-12);
        else
            lambda=lambda*10;
        end
    end
end
