function [R, p] = gs3dx_leg_fk(geom, pelvis_R, pelvis_p, q)
%GS3DX_LEG_FK  World pose of a GS3DX foot frame (the ankle follower).
%
%   [R, P] = GS3DX_LEG_FK(GEOM, PELVIS_R, PELVIS_P, Q) follows the leg chain
%   GS3DX_BUILD_LOWER_BODY builds (#10957):
%
%     pelvis -> Hip Mount (GEOM.mount_R, GEOM.mount_p) -> hip X-Y-Z
%            -> Knee Mount ([0 0 -thigh], knee axis along -y) -> knee Rz
%            -> Ankle Mount ([0 -shank 0], back to leg axes) -> ankle X-Y
%
%   Q = [hip X Y Z, knee, ankle X Y] in degrees, the angles the joints
%   report on their SignalBus.  The hip is an intrinsic X-Y-Z rotation
%   (GS3DX_KDS_Spherical), the ankle a Universal Rx*Ry.  GEOM fields:
%   mount_R (3x3), mount_p (3x1) relative to the pelvis ('Lower Torso')
%   frame, thigh and shank lengths (m).  R (3x3) and P (3x1) are the foot
%   frame in World.

    arguments
        geom (1,1) struct
        pelvis_R (3,3) double
        pelvis_p (3,1) double
        q (6,1) double
    end
    Rm = pelvis_R * geom.mount_R;
    pm = pelvis_p + pelvis_R * geom.mount_p;
    Rh = Rm * rx(q(1)) * ry(q(2)) * rz(q(3));
    Rk = Rh * [1 0 0; 0 0 -1; 0 1 0] * rz(q(4));
    pk = pm + Rh * [0; 0; -geom.thigh];
    R = Rk * [1 0 0; 0 0 1; 0 -1 0] * rx(q(5)) * ry(q(6));
    p = pk + Rk * [0; -geom.shank; 0];
end

function R = rx(a)
    R = [1 0 0; 0 cosd(a) -sind(a); 0 sind(a) cosd(a)];
end

function R = ry(a)
    R = [cosd(a) 0 sind(a); 0 1 0; -sind(a) 0 cosd(a)];
end

function R = rz(a)
    R = [cosd(a) -sind(a) 0; sind(a) cosd(a) 0; 0 0 1];
end
