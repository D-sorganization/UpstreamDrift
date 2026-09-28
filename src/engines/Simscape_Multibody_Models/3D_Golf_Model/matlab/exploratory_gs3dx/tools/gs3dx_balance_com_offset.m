function offset = gs3dx_balance_com_offset(check, t)
%GS3DX_BALANCE_COM_OFFSET  Centre of mass in the pelvis frame, from a run (#10979).
%
%   OFFSET = GS3DX_BALANCE_COM_OFFSET(CHECK, T) is the whole-body centre of
%   mass of a GS3DX_CONTACT_CHECK run in its pelvis frame,
%       R_pelvis' (com - p_pelvis)       (3 x numel(T), m)
%   on the times T, held at the run's first and last values outside it.
%   It depends only on the joint angles, so a run whose joints track the
%   capture gives the offset of the capture itself even if its body drifts:
%   GS3DX_BUILD_FIT_BALANCE carries it with the reference pelvis pose to
%   make the centre-of-mass reference.

    arguments
        check (1,1) struct
        t (1,:) double
    end
    assert(all(isfield(check, {'t', 'com', 'pelvis_p', 'pelvis_R'})), 'gs3dx:comoffset', ...
        'Precondition: CHECK needs .t, .com, .pelvis_p and .pelvis_R (GS3DX_CONTACT_CHECK)');
    n = numel(check.t);
    rel = squeeze(pagemtimes(pagetranspose(check.pelvis_R), reshape(check.com - check.pelvis_p, 3, 1, n)));
    tc = min(max(t, check.t(1)), check.t(end));
    offset = interp1(check.t(:), rel.', tc(:)).';
end
