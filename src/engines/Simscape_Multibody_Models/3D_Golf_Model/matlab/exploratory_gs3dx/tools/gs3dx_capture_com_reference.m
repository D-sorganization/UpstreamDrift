function com = gs3dx_capture_com_reference(k, t, com0)
%GS3DX_CAPTURE_COM_REFERENCE  The capture's centre of mass as a balance reference (#10979).
%
%   COM = GS3DX_CAPTURE_COM_REFERENCE(K, T, COM0) resamples the capture's
%   centre of mass K.com (GS3DX_KINEMATIC_GRF, 3 x frames, m, capture target
%   frame) onto the model reference times T (s, from the start of the
%   capture) and translates it so that COM(:, 1) = COM0, the model's centre
%   of mass at address in World.  World is the capture target frame with
%   its origin moved to the address pelvis, so a translation is the whole
%   difference; what is left of the capture is its motion.
%
%   Used as COM_REF of GS3DX_BUILD_FIT_BALANCE: the capture's centre of
%   mass moves 19 mm vertically before impact, where the model's own mass
%   distribution carried along the tracked pose moves 40 mm and demands a
%   2.6 BW support peak (docs/FIT.md).

    arguments
        k (1,1) struct
        t (1,:) double {mustBeFinite}
        com0 (3,1) double {mustBeFinite}
    end
    assert(isfield(k, 't') && isfield(k, 'com') && size(k.com, 1) == 3 && size(k.com, 2) == numel(k.t), ...
        'gs3dx:capturecom', 'K must carry .t and a 3 x numel(t) .com');
    tk = k.t(:) - k.t(1);
    assert(t(1) >= tk(1) && t(end) <= tk(end) + 1e-9, 'gs3dx:capturecom', ...
        'T (%.3f-%.3f s) lies outside the capture (%.3f-%.3f s)', t(1), t(end), tk(1), tk(end));
    com = interp1(tk, k.com.', t(:), 'linear', 'extrap').';
    com = com - com(:, 1) + com0;
end
