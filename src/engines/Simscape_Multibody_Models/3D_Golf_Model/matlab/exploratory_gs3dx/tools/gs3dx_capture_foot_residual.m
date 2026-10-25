function r = gs3dx_capture_foot_residual(k, jc, varargin)
%GS3DX_CAPTURE_FOOT_RESIDUAL  Foot-split residual on capture kinematic GRF (#11173).
%
%   R = GS3DX_CAPTURE_FOOT_RESIDUAL(K, JC) compares the closed-form foot
%   split (GS3DX_FOOT_SPLIT) to the capture's kinematic total ground reaction
%   K (GS3DX_KINEMATIC_GRF) using ankle joint centres JC
%   (GS3DX_CAPTURE_JOINT_CENTRES).  K and JC share the address target frame.
%
%   If numel(K.t) ~= size(JC.ankle_L, 2), ankle positions are resampled onto
%   K.t with interp1 (K.t is seconds from the first capture frame; JC has .t).
%
%   Optional positional MU (default 0.8) is Coulomb friction for GS3DX_FOOT_SPLIT.
%   Name-value COM_OFFSET (3x1, default [0;0;0]) is subtracted from K.com before
%   the split (K.grf is unchanged).  For K from GS3DX_KINEMATIC_GRF, use
%   TF.S.' * TF.origin with TF = GS3DX_CAPTURE_ADDRESS_TRANSFORM(CAP) so COM
%   matches the waist-origin frame used for JC ankles.
%
%   R.force_rms_bw   RMS of the pelvis force residual (N), divided by
%                    K.mass * 9.81 (body weights).
%   R.moment_rms_nm  RMS of the pelvis moment residual (N*m).
%   R.moment_rms_bw  R.moment_rms_nm / (K.mass * 9.81 * 1 m), dimensionless
%                    in metre-body-weights (the divisor uses a 1 m lever arm).
%   R.moment_vertical_rms_nm, R.moment_horizontal_rms_nm
%                    Same RMS, but the moment is split by zeroing the horizontal
%                    or vertical part of each foot force before the cross
%                    products (vertical-only forces: CoP offset; horizontal-only:
%                    lever arm from ankle height).  The two parts sum to the
%                    split moment at each frame.
%   R.impact_fz_bw   K.grf(3, K.impact_frame) / (K.mass * 9.81); the capture
%                    impact-frame vertical GRF, not a recomputed peak.

    mu = 0.8;
    com_offset = zeros(3, 1);
    args = varargin;
    if ~isempty(args) && isnumeric(args{1}) && isscalar(args{1})
        mu = args{1};
        args = args(2:end);
    end
    if mod(numel(args), 2) ~= 0
        error('gs3dx:foot_residual', 'Name-value arguments must come in pairs');
    end
    for i = 1:2:numel(args)
        name = lower(string(args{i}));
        val = args{i + 1};
        switch name
            case "mu"
                mu = val;
            case "com_offset"
                com_offset = val(:);
            otherwise
                error('gs3dx:foot_residual', 'Unknown option ''%s''', args{i});
        end
    end
    assert(numel(com_offset) == 3 && all(isfinite(com_offset)), 'gs3dx:foot_residual', ...
        'COM_OFFSET must be a finite 3x1 vector');
    assert(isscalar(mu) && mu >= 0 && isfinite(mu), 'gs3dx:foot_residual', ...
        'MU must be a non-negative finite scalar');

    assert(size(k.grf, 1) == 3 && size(k.com, 1) == 3, 'gs3dx:foot_residual', ...
        'K.grf and K.com must be 3xN');
    assert(isequal(size(k.grf), size(k.com)), 'gs3dx:foot_residual', ...
        'K.grf and K.com sizes must match');
    assert(all(isfinite(k.grf(:))) && all(isfinite(k.com(:))), 'gs3dx:foot_residual', ...
        'K.grf and K.com must be finite');
    assert(isscalar(k.mass) && k.mass > 0 && isfinite(k.mass), 'gs3dx:foot_residual', ...
        'K.mass must be a positive finite scalar');
    n = size(k.grf, 2);
    assert(isscalar(k.impact_frame) && k.impact_frame >= 1 && ...
        k.impact_frame <= n && k.impact_frame == floor(k.impact_frame), ...
        'gs3dx:foot_residual', 'K.impact_frame must be an integer index in range');

    t_k = k.t(:).';
    if numel(t_k) ~= n
        error('gs3dx:foot_residual', 'K.t length must match the number of columns in K.grf');
    end

    if numel(t_k) ~= size(jc.ankle_L, 2)
        t_jc = jc.t(:).';
        ankle_L = interp1(t_jc, jc.ankle_L.', t_k, 'linear', 'extrap').';
        ankle_R = interp1(t_jc, jc.ankle_R.', t_k, 'linear', 'extrap').';
    else
        ankle_L = jc.ankle_L;
        ankle_R = jc.ankle_R;
    end

    feet.p_L = ankle_L;
    feet.p_R = ankle_R;
    p_com = k.com - com_offset;
    s = gs3dx_foot_split(k.grf, p_com, feet, mu);

    f_L_v = s.f_L;
    f_L_v(1:2, :) = 0;
    f_R_v = s.f_R;
    f_R_v(1:2, :) = 0;
    f_L_h = s.f_L;
    f_L_h(3, :) = 0;
    f_R_h = s.f_R;
    f_R_h(3, :) = 0;

    rL = ankle_L - p_com;
    rR = ankle_R - p_com;
    moment_vertical = cross(rL, f_L_v, 1) + cross(rR, f_R_v, 1);
    moment_horizontal = cross(rL, f_L_h, 1) + cross(rR, f_R_h, 1);

    moment_sum = moment_vertical + moment_horizontal;
    err = max(abs(moment_sum - s.moment), [], 'all');
    scale = max(max(abs(s.moment(:))), 1);
    assert(err <= 1e-6 || err / scale <= 1e-6, 'gs3dx:foot_residual', ...
        'Moment vertical/horizontal split must reconstruct the foot-split moment');

    g = 9.81;
    bw = k.mass * g;
    force_norm = vecnorm(s.force);
    moment_norm = vecnorm(s.moment);

    r.force_rms_bw = sqrt(mean(force_norm .^ 2)) / bw;
    r.moment_rms_nm = sqrt(mean(moment_norm .^ 2));
    r.moment_rms_bw = r.moment_rms_nm / (bw * 1);
    r.moment_vertical_rms_nm = sqrt(mean(vecnorm(moment_vertical) .^ 2));
    r.moment_horizontal_rms_nm = sqrt(mean(vecnorm(moment_horizontal) .^ 2));
    r.impact_fz_bw = k.grf(3, k.impact_frame) / bw;
end
