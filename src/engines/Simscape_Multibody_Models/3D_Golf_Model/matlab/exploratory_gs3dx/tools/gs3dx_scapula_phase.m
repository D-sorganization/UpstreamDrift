function [lo, hi, pull] = gs3dx_scapula_phase(b, frame, top, impact)
%GS3DX_SCAPULA_PHASE  Sustain Protraction Through Backswing, Release to Impact.
% TOP and IMPACT are explicit capture frame indices. A raised-cosine release
% widens only finite scapula bounds to +/- pi, then restores unbounded IK.
% It also releases the scapula posture target back to the legacy zero prior.
    lo=[];hi=[];pull=0;
    if ~b.active, return; end
    assert(isreal([frame,top,impact]) && all(isfinite([frame,top,impact])) && ...
        all([frame,top,impact]==fix([frame,top,impact])) && frame>=1 && top>1 && impact>top, ...
        'gs3dx:ik:scapula','Require positive frame and explicit 1 < backswing top < impact');
    if frame >= impact, return; end
    pull=1;
    if frame==1
        lo=b.address_lower;hi=b.address_upper;return;
    end
    lo=b.lower;hi=b.upper;
    if frame>top
        release=.5-.5*cos(pi*(frame-top)/(impact-top));
        bounded=isfinite(lo);lo(bounded)=(1-release)*lo(bounded)-release*pi;
        bounded=isfinite(hi);hi(bounded)=(1-release)*hi(bounded)+release*pi;
        pull=1-release;
    end
end
