"""MATLAB source templates for the generated Simscape grip contact kernel (#13334).

Pure text consumed by :func:`matlab_kernel.render_matlab_kernel_source`.  The
loop body mirrors :func:`pair_contact.sphere_capsule` for sphere rows and the
line-contact quadrature of :func:`pair_contact.capsule_capsule` (#13308,
#13508) for capsule rows: ``n`` equally spaced node spheres along the hand
capsule, each weighted ``1 / n``, combined as in
:func:`pair_contact.combine_line_nodes`.  ``@MAXP@`` stands for
``MAX_PHYSICAL_PAIRS``.
"""

from __future__ import annotations

MATLAB_LOOP_BODY = """    % Club-side grip frames in World
    dp_L = R_club * p_club_L_local;
    p_b_L = p_club + dp_L;
    R_b_L = R_club * R_club_L_local;
    v_b_L = v_club + local_cross3(omega_club, dp_L);

    dp_R = R_club * p_club_R_local;
    p_b_R = p_club + dp_R;
    R_b_R = R_club * R_club_R_local;
    v_b_R = v_club + local_cross3(omega_club, dp_R);

    % Fixed-size output preallocation
    F_hand_L_w = zeros(3, 1);
    M_hand_L_w = zeros(3, 1);
    F_hand_R_w = zeros(3, 1);
    M_hand_R_w = zeros(3, 1);
    F_side_L = zeros(3, 1);
    M_side_L = zeros(3, 1);
    F_side_R = zeros(3, 1);
    M_side_R = zeros(3, 1);

    pair_normal_n = zeros(@MAXP@, 1);
    pair_tangent_n = zeros(@MAXP@, 1);
    pair_spin_nm = zeros(@MAXP@, 1);
    pair_penetration_m = zeros(@MAXP@, 1);
    pair_slip_speed_m_s = zeros(@MAXP@, 1);
    pair_signed_dist_m = zeros(@MAXP@, 1);
    pair_force_club_w = zeros(@MAXP@, 3);
    pair_moment_club_w = zeros(@MAXP@, 3);
    pair_contact_point_w = zeros(@MAXP@, 3);

    for k = 1:@MAXP@
        if pair_table(k, 1) <= 0.0
            continue;
        end
        side_code = pair_table(k, 2);
        a0 = [pair_table(k, 4); pair_table(k, 5); pair_table(k, 6)];
        a1 = [pair_table(k, 7); pair_table(k, 8); pair_table(k, 9)];
        r_a = pair_table(k, 10);
        b0 = [pair_table(k, 11); pair_table(k, 12); pair_table(k, 13)];
        b1 = [pair_table(k, 14); pair_table(k, 15); pair_table(k, 16)];
        r_b = pair_table(k, 17);
        n_nodes = pair_table(k, 18);

        if side_code == 1.0
            p_a = p_hand_L;
            R_a = R_hand_L;
            v_a0 = v_hand_L;
            omega_a = omega_hand_L;
            p_b = p_b_L;
            R_b = R_b_L;
            v_b0 = v_b_L;
        else
            p_a = p_hand_R;
            R_a = R_hand_R;
            v_a0 = v_hand_R;
            omega_a = omega_hand_R;
            p_b = p_b_R;
            R_b = R_b_R;
            v_b0 = v_b_R;
        end
        b0_w = p_b + R_b * b0;
        b1_w = p_b + R_b * b1;

        % Sphere rows have one node at a0; capsule rows integrate n_nodes
        % equally weighted node spheres along the hand capsule (#13308).
        w = 1.0 / n_nodes;
        f_n_mag = 0.0;
        f_t_mag = 0.0;
        spin_mag = 0.0;
        penetration = 0.0;
        signed_dist = Inf;
        force_a = zeros(3, 1);
        moment_a = zeros(3, 1);
        force_b = zeros(3, 1);
        moment_b_side = zeros(3, 1);
        cp_load = zeros(3, 1);
        cp_mean = zeros(3, 1);
        slip_load = 0.0;
        slip_mean = 0.0;
        for j = 1:n_nodes
            if n_nodes > 1.0
                node = a0 + ((j - 1) / (n_nodes - 1)) * (a1 - a0);
            else
                node = a0;
            end
            [fn, ft, sp, slp, pen, sd, fa, ma, fb, mb, cpt] = local_sphere_capsule( ...
                p_a + R_a * node, r_a, b0_w, b1_w, r_b, ...
                p_a, v_a0, omega_a, p_b, v_b0, omega_club, ...
                k_n, c_n, mu_s, mu_d, mu_v, v_tr, r_patch, omega_tr);
            f_n_mag = f_n_mag + w * fn;
            f_t_mag = f_t_mag + w * ft;
            spin_mag = spin_mag + w * sp;
            penetration = max(penetration, pen);
            signed_dist = min(signed_dist, sd);
            force_a = force_a + w * fa;
            moment_a = moment_a + w * ma;
            force_b = force_b + w * fb;
            moment_b_side = moment_b_side + w * mb;
            cp_load = cp_load + (w * fn) * cpt;
            cp_mean = cp_mean + w * cpt;
            slip_load = slip_load + (w * fn) * slp;
            slip_mean = slip_mean + w * slp;
        end
        if f_n_mag > 0.0
            contact_pt = cp_load / f_n_mag;
            slip_speed = slip_load / f_n_mag;
        else
            contact_pt = cp_mean;
            slip_speed = slip_mean;
        end

        if side_code == 1.0
            F_hand_L_w = F_hand_L_w + force_a;
            M_hand_L_w = M_hand_L_w + moment_a;
            F_side_L = F_side_L + force_b;
            M_side_L = M_side_L + moment_b_side;
        else
            F_hand_R_w = F_hand_R_w + force_a;
            M_hand_R_w = M_hand_R_w + moment_a;
            F_side_R = F_side_R + force_b;
            M_side_R = M_side_R + moment_b_side;
        end

        pair_normal_n(k) = f_n_mag;
        pair_tangent_n(k) = f_t_mag;
        pair_spin_nm(k) = spin_mag;
        pair_penetration_m(k) = penetration;
        pair_slip_speed_m_s(k) = slip_speed;
        pair_signed_dist_m(k) = signed_dist;
        pair_force_club_w(k, :) = force_b.';
        pair_moment_club_w(k, :) = moment_b_side.';
        pair_contact_point_w(k, :) = contact_pt.';
    end

    F_club_w = F_side_L + F_side_R;
    M_club_w = (M_side_L + local_cross3(p_b_L - p_club, F_side_L)) + ...
               (M_side_R + local_cross3(p_b_R - p_club, F_side_R));
end
"""

MATLAB_HELPER_FUNCTIONS = """
function c = local_cross3(a, b)
    c = [a(2)*b(3) - a(3)*b(2); ...
         a(3)*b(1) - a(1)*b(3); ...
         a(1)*b(2) - a(2)*b(1)];
end

function u = local_orthogonal_unit(axis)
    nrm = sqrt(axis(1)*axis(1) + axis(2)*axis(2) + axis(3)*axis(3));
    if nrm < 1.0e-12
        u = [1.0; 0.0; 0.0];
        return;
    end
    unit = axis / nrm;
    if abs(unit(3)) < 0.9
        ref = [0.0; 0.0; 1.0];
    else
        ref = [1.0; 0.0; 0.0];
    end
    ortho = local_cross3(unit, ref);
    onrm = sqrt(ortho(1)*ortho(1) + ortho(2)*ortho(2) + ortho(3)*ortho(3));
    if onrm > 1.0e-12
        u = ortho / onrm;
    else
        u = [1.0; 0.0; 0.0];
    end
end

function q = local_closest_point_segment(p, a, b)
    ab = b - a;
    denom = ab(1)*ab(1) + ab(2)*ab(2) + ab(3)*ab(3);
    if denom < 1.0e-14
        q = a;
        return;
    end
    ap = p - a;
    t = (ap(1)*ab(1) + ap(2)*ab(2) + ap(3)*ab(3)) / denom;
    t = min(max(t, 0.0), 1.0);
    q = a + t * ab;
end

function [f_n_mag, f_t_mag, spin_mag, slip_speed, penetration, signed_dist, ...
    force_a, moment_a, force_b, moment_b_side, contact_pt] = local_sphere_capsule( ...
    center_w, r_a, b0_w, b1_w, r_b, p_a, v_a0, omega_a, p_b, v_b0, omega_b, ...
    k_n, c_n, mu_s, mu_d, mu_v, v_tr, r_patch, omega_tr)
    p_axis = local_closest_point_segment(center_w, b0_w, b1_w);
    diff_v = p_axis - center_w;
    dist = sqrt(diff_v(1)*diff_v(1) + diff_v(2)*diff_v(2) + diff_v(3)*diff_v(3));
    if dist > 1.0e-12
        normal = diff_v / dist;
    else
        normal = local_orthogonal_unit(b1_w - b0_w);
    end
    signed_dist = dist - r_a - r_b;
    penetration = max(0.0, -signed_dist);
    contact_pt = center_w + r_a * normal;
    v_a = v_a0 + local_cross3(omega_a, contact_pt - p_a);
    v_b = v_b0 + local_cross3(omega_b, contact_pt - p_b);
    [f_n_mag, f_t_mag, spin_mag, slip_speed, force_a, spin_a] = ...
        local_plane_contact(-normal, penetration, v_a - v_b, omega_a - omega_b, ...
        k_n, c_n, mu_s, mu_d, mu_v, v_tr, r_patch, omega_tr);
    force_b = -force_a;
    moment_a = local_cross3(contact_pt - p_a, force_a) + spin_a;
    moment_b_side = local_cross3(contact_pt - p_b, force_b) - spin_a;
end

function [f_n_mag, f_t_mag, spin_mag, slip_speed, force_vec, spin_vec] = ...
    local_plane_contact(u_normal, penetration, v_rel, omega_rel, ...
    k_n, c_n, mu_s, mu_d, mu_v, v_tr, r_patch, omega_tr)
    v_n = u_normal(1)*v_rel(1) + u_normal(2)*v_rel(2) + u_normal(3)*v_rel(3);
    v_t = v_rel - u_normal * v_n;
    slip_speed = sqrt(v_t(1)*v_t(1) + v_t(2)*v_t(2) + v_t(3)*v_t(3));
    if penetration <= 0.0
        f_n_mag = 0.0;
        f_t_mag = 0.0;
        spin_mag = 0.0;
        force_vec = zeros(3, 1);
        spin_vec = zeros(3, 1);
        return;
    end
    rate = -v_n;
    f_n_mag = max(0.0, k_n * penetration * (1.0 + c_n * rate));
    normal_force = u_normal * f_n_mag;
    if slip_speed > 0.0 && f_n_mag > 0.0
        ratio = slip_speed / v_tr;
        mu = mu_d + (mu_s - mu_d) * exp(-(ratio * ratio)) + mu_v * slip_speed;
        f_t_mag = mu * f_n_mag * tanh(ratio);
        friction_force = (-v_t / slip_speed) * f_t_mag;
    else
        f_t_mag = 0.0;
        friction_force = zeros(3, 1);
    end
    force_vec = normal_force + friction_force;
    if f_n_mag > 0.0 && r_patch > 0.0
        spin_rate = u_normal(1)*omega_rel(1) + u_normal(2)*omega_rel(2) + u_normal(3)*omega_rel(3);
        spin_mag = mu_s * f_n_mag * r_patch * abs(tanh(spin_rate / omega_tr));
        spin_vec = -u_normal * (mu_s * f_n_mag * r_patch * tanh(spin_rate / omega_tr));
    else
        spin_mag = 0.0;
        spin_vec = zeros(3, 1);
    end
end
"""

__all__ = ["MATLAB_HELPER_FUNCTIONS", "MATLAB_LOOP_BODY"]
