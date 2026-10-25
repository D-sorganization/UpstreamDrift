function s = gs3dx_foot_split(f_required, p_com, feet, mu)
%GS3DX_FOOT_SPLIT  Split required ground reaction between feet (#11173 step 4).
%
%   S = GS3DX_FOOT_SPLIT(F_REQUIRED, P_COM, FEET, MU) distributes the total
%   external force the ground must supply (World, Z-up, Newtons) between two
%   foot contact points.  F_REQUIRED is 3xN, equal to M*(a_com - g) when the
%   motion is given (positive Z is up; quiet stance is about [0;0;+M*9.80665]).
%   P_COM is 3xN (m).  FEET has .p_L and .p_R (3xN).  MU is Coulomb friction
%   (default 0.8).  Closed form per frame; excess force and moment go to the
%   pelvis residual wrench at P_COM.  See docs/FORWARD_DYNAMICS.md idea 2.

    if nargin < 4
        mu = 0.8;
    end

    assert(size(f_required, 1) == 3, 'gs3dx:foot_split', 'F_REQUIRED must be 3xN');
    assert(isequal(size(f_required), size(p_com)), 'gs3dx:foot_split', 'F_REQUIRED and P_COM sizes must match');
    assert(isfield(feet, 'p_L') && isfield(feet, 'p_R'), 'gs3dx:foot_split', 'FEET must have p_L and p_R');
    assert(isequal(size(feet.p_L), size(f_required)), 'gs3dx:foot_split', 'FEET.p_L must match F_REQUIRED');
    assert(isequal(size(feet.p_R), size(f_required)), 'gs3dx:foot_split', 'FEET.p_R must match F_REQUIRED');
    assert(isscalar(mu) && mu >= 0 && isfinite(mu), 'gs3dx:foot_split', 'MU must be a non-negative finite scalar');
    assert(all(isfinite(f_required(:))) && all(isfinite(p_com(:))) && ...
        all(isfinite(feet.p_L(:))) && all(isfinite(feet.p_R(:))), 'gs3dx:foot_split', 'Inputs must be finite');

    n = size(f_required, 2);
    f_L = zeros(3, n);
    f_R = zeros(3, n);
    force = zeros(3, n);
    moment = zeros(3, n);
    fz_required = f_required(3, :);

    for k = 1:n
        f_req = f_required(:, k);
        pc = p_com(:, k);
        pL = feet.p_L(:, k);
        pR = feet.p_R(:, k);

        fz_total = max(f_req(3), 0);
        if fz_total <= 0
            fz_L = 0;
            fz_R = 0;
        else
            d = pR(1:2) - pL(1:2);
            dd = d.' * d;
            if dd <= 0
                fz_L = 0.5 * fz_total;
                fz_R = 0.5 * fz_total;
            else
                t = ((pc(1:2) - pL(1:2)).' * d) / dd;
                if t <= 0
                    fz_L = fz_total;
                    fz_R = 0;
                elseif t >= 1
                    fz_L = 0;
                    fz_R = fz_total;
                else
                    fz_R = t * fz_total;
                    fz_L = fz_total - fz_R;
                end
            end
        end

        fz_sum = fz_L + fz_R;
        f_req_h = f_req(1:2);
        if fz_sum > 0
            fL_h = f_req_h * (fz_L / fz_sum);
            fR_h = f_req_h * (fz_R / fz_sum);
        else
            fL_h = zeros(2, 1);
            fR_h = zeros(2, 1);
        end

        fL_h = local_cap_horizontal(fL_h, mu * fz_L);
        fR_h = local_cap_horizontal(fR_h, mu * fz_R);

        fL = [fL_h; fz_L];
        fR = [fR_h; fz_R];
        f_L(:, k) = fL;
        f_R(:, k) = fR;
        force(:, k) = f_req - fL - fR;
        moment(:, k) = cross(pL - pc, fL) + cross(pR - pc, fR);
    end

    support = zeros(1, n);
    pos_fz = fz_required > 0;
    support(pos_fz) = (f_L(3, pos_fz) + f_R(3, pos_fz)) ./ fz_required(pos_fz);

    s = struct('f_L', f_L, 'f_R', f_R, 'force', force, 'moment', moment, ...
        'support', support, 'fz_required', fz_required);

    assert(all(f_L(3, :) >= -1e-12) && all(f_R(3, :) >= -1e-12), 'gs3dx:foot_split', 'Negative vertical contact');
    assert(max(abs(f_L + f_R + force - f_required), [], 'all') <= 1e-9, 'gs3dx:foot_split', 'Force balance');
    for k = 1:n
        capL = mu * f_L(3, k) + 1e-9;
        capR = mu * f_R(3, k) + 1e-9;
        assert(norm(f_L(1:2, k)) <= capL, 'gs3dx:foot_split', 'Left foot friction');
        assert(norm(f_R(1:2, k)) <= capR, 'gs3dx:foot_split', 'Right foot friction');
    end
end

function f_h = local_cap_horizontal(f_h, cap)
    n = norm(f_h);
    if n > cap && n > 0
        f_h = f_h * (cap / n);
    end
end
