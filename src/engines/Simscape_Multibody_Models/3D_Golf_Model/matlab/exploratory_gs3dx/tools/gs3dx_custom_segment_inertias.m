function inertias = gs3dx_custom_segment_inertias(mass_spec, lengths, anthro)
%GS3DX_CUSTOM_SEGMENT_INERTIAS  De Leva (1996) custom segment inertias (#10979, #11011).
%
%   INERTIAS = GS3DX_CUSTOM_SEGMENT_INERTIAS(MASS_SPEC, LENGTHS, ANTHRO)
%   computes the mass, centre of mass (in the solid frame), and principal
%   moments of inertia for de Leva custom-inertia solids (thigh, shank,
%   upper arm, forearm halves, hand, head) based on de Leva (1996) Table 4.
%
%   Inputs:
%     MASS_SPEC  Can be:
%                  - Numeric scalar: golfer body mass (kg), evaluates GS3DX_ANTHROPOMETRY.
%                  - Struct output of GS3DX_ANTHROPOMETRY (containing .vars and .legs).
%                  - Struct of actual segment masses (kg) supporting per-side or unified
%                    keys: .thigh (or .thigh_L, .thigh_R), .shank (or .shank_L, .shank_R),
%                    .upper_arm (or .upper_arm_L, .upper_arm_R), .forearm (or .forearm_L,
%                    .forearm_R, representing whole forearm mass), .hand (or .hand_L,
%                    .hand_R), and .head.
%     LENGTHS    Struct of segment lengths (m):
%                  .thigh, .shank, .upper_arm, .forearm
%                  (optional .hand, .head; defaults to de Leva reference).
%     ANTHRO     Optional anthropometry struct when MASS_SPEC is a mass struct.
%                If omitted, GS3DX_ANTHROPOMETRY(80) supplies com & gyration fractions.
%
%   Returns struct INERTIAS containing computed segment inertias:
%     .thigh / .thigh_L / .thigh_R: .mass, .com (1x3 m), .moments (1x3 kg*m^2), .length
%     .shank / .shank_L / .shank_R: .mass, .com (1x3 m), .moments (1x3 kg*m^2), .length
%     .upper_arm / .upper_arm_L / .upper_arm_R: .mass, .com, .moments, .length
%     .forearm / .forearm_L / .forearm_R: .mass (whole), .length,
%       .upper (.mass, .com, .moments), .lower (.mass, .com, .moments),
%       .whole (.mass, .com, .moments)
%     .hand / .hand_L / .hand_R: .mass, .com ([0 0 0]), .moments, .length
%     .head: .mass, .com ([0 0 0]), .moments, .length
%     .anthro: anthropometry struct used for com and gyration fractions
%
%   Physical Invariants (DbC):
%     - All masses and principal moments are strictly positive and finite.
%     - All inertia tensors satisfy the non-negative triangle inequality:
%       I_i + I_j >= I_k for all permutations {i,j,k} = {1,2,3}.
%     - Forearm halves lump along z to reconstruct the whole de Leva forearm.
%     - Forearm half transverse moment It is strictly positive.

    arguments
        mass_spec
        lengths (1,1) struct
        anthro struct = struct()
    end

    % 1. Resolve anthropometry reference (fractions and reference lengths)
    if isempty(fieldnames(anthro))
        if isnumeric(mass_spec) && isscalar(mass_spec) && isreal(mass_spec) && isfinite(mass_spec) && mass_spec > 0
            a = gs3dx_anthropometry(double(mass_spec));
        elseif isstruct(mass_spec) && isfield(mass_spec, 'gyration') && isfield(mass_spec, 'com')
            a = mass_spec;
        else
            a = gs3dx_anthropometry(80);
        end
    else
        assert(isfield(anthro, 'gyration') && isfield(anthro, 'com') && isfield(anthro, 'length'), ...
            'gs3dx:custom_inertias:invalid_anthro', 'anthro struct must contain gyration, com, and length fields');
        a = anthro;
    end

    % 2. Resolve segment masses
    masses = struct();
    if isnumeric(mass_spec)
        assert(isreal(mass_spec) && isscalar(mass_spec) && isfinite(mass_spec) && mass_spec > 0, ...
            'gs3dx:custom_inertias:invalid_mass', 'body_mass must be a positive finite real scalar (kg)');
        m_body = double(mass_spec);
        a = gs3dx_anthropometry(m_body);
        masses.thigh = a.legs.ThighMass;
        masses.shank = a.legs.ShankMass;
        masses.upper_arm = a.vars.GolferUpperArmMass;
        masses.forearm = a.vars.GolferForearmMass;
        masses.hand = a.vars.GolferHandMass;
        masses.head = a.vars.GolferHeadMass;
    elseif isstruct(mass_spec) && isfield(mass_spec, 'vars') && isfield(mass_spec, 'legs')
        % Anthropometry struct
        masses.thigh = mass_spec.legs.ThighMass;
        masses.shank = mass_spec.legs.ShankMass;
        masses.upper_arm = mass_spec.vars.GolferUpperArmMass;
        masses.forearm = mass_spec.vars.GolferForearmMass;
        masses.hand = mass_spec.vars.GolferHandMass;
        masses.head = mass_spec.vars.GolferHeadMass;
    elseif isstruct(mass_spec)
        % Map of actual masses (preserves custom or per-side masses)
        masses = mass_spec;
    else
        error('gs3dx:custom_inertias:invalid_input', ...
            'First argument must be numeric body mass, anthropometry struct, or segment mass struct');
    end

    % 3. Validate lengths
    req_len_fields = ["thigh", "shank", "upper_arm", "forearm"];
    for f = req_len_fields
        assert(isfield(lengths, f), 'gs3dx:custom_inertias:missing_length', ...
            'lengths struct must contain field "%s" (m)', f);
        val = lengths.(f);
        assert(isnumeric(val) && isreal(val) && isscalar(val) && isfinite(val) && val > 0, ...
            'gs3dx:custom_inertias:invalid_length', ...
            'lengths.%s must be a positive finite real scalar (m)', f);
    end

    L_hand = a.length.hand;
    if isfield(lengths, 'hand') && ~isempty(lengths.hand)
        val = lengths.hand;
        assert(isnumeric(val) && isreal(val) && isscalar(val) && isfinite(val) && val > 0, ...
            'gs3dx:custom_inertias:invalid_length', 'lengths.hand must be a positive finite real scalar (m)');
        L_hand = double(val);
    end

    L_head = a.length.head;
    if isfield(lengths, 'head') && ~isempty(lengths.head)
        val = lengths.head;
        assert(isnumeric(val) && isreal(val) && isscalar(val) && isfinite(val) && val > 0, ...
            'gs3dx:custom_inertias:invalid_length', 'lengths.head must be a positive finite real scalar (m)');
        L_head = double(val);
    end

    L_thigh = double(lengths.thigh);
    L_shank = double(lengths.shank);
    L_ua = double(lengths.upper_arm);
    L_fa = double(lengths.forearm);

    inertias = struct();
    inertias.anthro = a;

    % 4. Compute simple limb segments (thigh, shank, upper arm, hand)
    % Thigh
    if isfield(masses, 'thigh')
        inertias.thigh = local_calc_limb(masses.thigh, L_thigh, a.com.thigh, a.gyration.thigh, 1, false);
        inertias.thigh_L = inertias.thigh;
        inertias.thigh_R = inertias.thigh;
    else
        if isfield(masses, 'thigh_L')
            inertias.thigh_L = local_calc_limb(masses.thigh_L, L_thigh, a.com.thigh, a.gyration.thigh, 1, false);
        end
        if isfield(masses, 'thigh_R')
            inertias.thigh_R = local_calc_limb(masses.thigh_R, L_thigh, a.com.thigh, a.gyration.thigh, 1, false);
        end
        if isfield(inertias, 'thigh_L') && ~isfield(inertias, 'thigh')
            inertias.thigh = inertias.thigh_L;
        end
    end

    % Shank
    if isfield(masses, 'shank')
        inertias.shank = local_calc_limb(masses.shank, L_shank, a.com.shank, a.gyration.shank, 1, false);
        inertias.shank_L = inertias.shank;
        inertias.shank_R = inertias.shank;
    else
        if isfield(masses, 'shank_L')
            inertias.shank_L = local_calc_limb(masses.shank_L, L_shank, a.com.shank, a.gyration.shank, 1, false);
        end
        if isfield(masses, 'shank_R')
            inertias.shank_R = local_calc_limb(masses.shank_R, L_shank, a.com.shank, a.gyration.shank, 1, false);
        end
        if isfield(inertias, 'shank_L') && ~isfield(inertias, 'shank')
            inertias.shank = inertias.shank_L;
        end
    end

    % Upper Arm
    if isfield(masses, 'upper_arm')
        inertias.upper_arm = local_calc_limb(masses.upper_arm, L_ua, a.com.upper_arm, a.gyration.upper_arm, 1, false);
        inertias.upper_arm_L = inertias.upper_arm;
        inertias.upper_arm_R = inertias.upper_arm;
    else
        if isfield(masses, 'upper_arm_L')
            inertias.upper_arm_L = local_calc_limb(masses.upper_arm_L, L_ua, a.com.upper_arm, a.gyration.upper_arm, 1, false);
        end
        if isfield(masses, 'upper_arm_R')
            inertias.upper_arm_R = local_calc_limb(masses.upper_arm_R, L_ua, a.com.upper_arm, a.gyration.upper_arm, 1, false);
        end
        if isfield(inertias, 'upper_arm_L') && ~isfield(inertias, 'upper_arm')
            inertias.upper_arm = inertias.upper_arm_L;
        end
    end

    % Forearm (Two half solids, each taking half mass; lumps to de Leva forearm)
    if isfield(masses, 'forearm')
        inertias.forearm = local_calc_forearm(masses.forearm, L_fa, a.com.forearm, a.gyration.forearm);
        inertias.forearm_L = inertias.forearm;
        inertias.forearm_R = inertias.forearm;
    else
        if isfield(masses, 'forearm_L')
            inertias.forearm_L = local_calc_forearm(masses.forearm_L, L_fa, a.com.forearm, a.gyration.forearm);
        end
        if isfield(masses, 'forearm_R')
            inertias.forearm_R = local_calc_forearm(masses.forearm_R, L_fa, a.com.forearm, a.gyration.forearm);
        end
        if isfield(inertias, 'forearm_L') && ~isfield(inertias, 'forearm')
            inertias.forearm = inertias.forearm_L;
        end
    end

    % Hand (COM fixed at solid origin [0 0 0])
    if isfield(masses, 'hand')
        inertias.hand = local_calc_limb(masses.hand, L_hand, a.com.hand, a.gyration.hand, 1, true);
        inertias.hand_L = inertias.hand;
        inertias.hand_R = inertias.hand;
    else
        if isfield(masses, 'hand_L')
            inertias.hand_L = local_calc_limb(masses.hand_L, L_hand, a.com.hand, a.gyration.hand, 1, true);
        end
        if isfield(masses, 'hand_R')
            inertias.hand_R = local_calc_limb(masses.hand_R, L_hand, a.com.hand, a.gyration.hand, 1, true);
        end
        if isfield(inertias, 'hand_L') && ~isfield(inertias, 'hand')
            inertias.hand = inertias.hand_L;
        end
    end

    % Head (COM fixed at solid origin [0 0 0])
    assert(isfield(masses, 'head'), 'gs3dx:custom_inertias:missing_head_mass', 'masses must include head');
    inertias.head = local_calc_limb(masses.head, L_head, a.com.head, a.gyration.head, 1, true);
end

% -------------------------------------------------------------------------
% Helper: Limb custom inertia calculation (pure math)
% -------------------------------------------------------------------------
function s = local_calc_limb(m, len, com_frac, gyration, sgn, zero_com)
    assert(isnumeric(m) && isreal(m) && isscalar(m) && isfinite(m) && m > 0, ...
        'gs3dx:custom_inertias:invalid_mass', 'Limb mass must be positive finite real');
    [~, c, I] = gs3dx_segment_inertia(m, len, com_frac, gyration);
    moments = [mean(I(1:2)), mean(I(1:2)), I(3)];
    if zero_com
        com = [0, 0, 0];
    else
        com = [0, 0, sgn * (len / 2 - c)];
    end
    local_assert_tensor_triangle(moments);
    s = struct('mass', m, 'com', com, 'moments', moments, 'length', len);
end

% -------------------------------------------------------------------------
% Helper: Forearm halves calculation (parallel-axis theorem along z)
% -------------------------------------------------------------------------
function fa = local_calc_forearm(m_whole, len, com_frac, gyration)
    assert(isnumeric(m_whole) && isreal(m_whole) && isscalar(m_whole) && isfinite(m_whole) && m_whole > 0, ...
        'gs3dx:custom_inertias:invalid_mass', 'Forearm mass must be positive finite real');
    [~, c, I] = gs3dx_segment_inertia(m_whole, len, com_frac, gyration);
    m_half = m_whole / 2;
    It = mean(I(1:2)) / 2 - m_half * (len / 4)^2;
    assert(It > 0, 'gs3dx:custom_inertias:forearm_transverse', ...
        'Forearm halves cannot carry de Leva transverse moment (It = %.6e <= 0)', It);
    com_half = [0, 0, len / 2 - c];
    mom_half = [It, It, I(3) / 2];
    local_assert_tensor_triangle(mom_half);

    fa = struct( ...
        'mass', m_whole, ...
        'length', len, ...
        'upper', struct('mass', m_half, 'com', com_half, 'moments', mom_half), ...
        'lower', struct('mass', m_half, 'com', com_half, 'moments', mom_half), ...
        'whole', struct('mass', m_whole, 'com', [0, 0, len / 2 - c], ...
                        'moments', [mean(I(1:2)), mean(I(1:2)), I(3)]));
end

% -------------------------------------------------------------------------
% Helper: Assert tensor triangle inequality
% -------------------------------------------------------------------------
function local_assert_tensor_triangle(I_vec)
    assert(all(I_vec > 0) && all(isfinite(I_vec)), ...
        'gs3dx:custom_inertias:invalid_tensor', 'Moments must be positive finite');
    assert(I_vec(1) + I_vec(2) >= I_vec(3) - 1e-15 && ...
           I_vec(2) + I_vec(3) >= I_vec(1) - 1e-15 && ...
           I_vec(1) + I_vec(3) >= I_vec(2) - 1e-15, ...
           'gs3dx:custom_inertias:triangle_inequality', ...
           'Inertia tensor violates triangle inequality: [%.6e, %.6e, %.6e]', I_vec(1), I_vec(2), I_vec(3));
end
