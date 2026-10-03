function M = gs3dx_club_match(ref, sim, t, phases)
%GS3DX_CLUB_MATCH  How closely a simulated club follows a captured one (#11160).
%
%   M = GS3DX_CLUB_MATCH(REF, SIM, T, PHASES) is a pure comparison: it takes
%   two already-extracted club tracks (no model or capture access) and
%   reports position, orientation and speed error over the swing.
%
%   REF, SIM are structs, one frame set each, with fields (N the same for
%   both and for T):
%     .head         3xN, m, World, club-head position
%     .grip         3xN, m, World, grip (butt/hand) position
%     .face_normal  3xN, unit, World, club-face normal
%   T is 1xN, s, strictly increasing.  PHASES gives 1-based frame indices
%   with 1 <= address < top <= impact <= N (address, top of backswing and
%   ball impact, matching GS3DX_CAPTURE_MARKERS.impact_frame).
%
%   Per-frame errors:
%     head_err_m, grip_err_m   Euclidean distance, SIM to REF
%     shaft_err_deg   angle between the grip->head directions of REF and SIM
%     face_err_deg    angle between the face normals after each is projected
%                     onto the plane normal to REF's shaft direction.  The
%                     projection removes the component that rides along the
%                     shaft (loft/lie, which a pure club-face-open/closed
%                     metric should not charge), isolating the face-angle
%                     (open/closed) component the swing plane sees.
%     head_speed_ref_mps, head_speed_sim_mps   |d(head)/dt| by central
%                     differences (MATLAB GRADIENT) on T
%
%   M fields:
%     .frame   table: t, head_err_m, grip_err_m, shaft_err_deg, face_err_deg,
%              head_speed_ref_mps, head_speed_sim_mps (N rows)
%     .phase   table, one row per 'address_to_top', 'top_to_impact',
%              'address_to_impact': head_rms_mm, head_max_mm, grip_rms_mm,
%              grip_max_mm, shaft_rms_deg, shaft_max_deg, face_rms_deg,
%              face_max_deg (RMS/max of the per-frame errors over the
%              phase's inclusive frame span) and impact_speed_err_pct, the
%              percent speed error at PHASES.impact (the same value on every
%              row: it is a property of the impact instant, not the span)
%     .ok      struct against the #11160 targets (LOCAL_TARGETS), measured
%              on the 'address_to_impact' row: .head_rms, .head_max,
%              .shaft, .speed (each a logical) and .all, their conjunction
%     .targets the LOCAL_TARGETS struct these were checked against
%
%   Precondition: REF and SIM share N with matching finite 3xN/1xN fields
%   and unit-norm face normals; T is finite and increasing; PHASES is
%   ordered.  Errors use 'gs3dx:club_match'.

    arguments
        ref (1,1) struct
        sim (1,1) struct
        t (1,:) double
        phases (1,1) struct
    end
    n = numel(t);
    assert(all(isfinite(t)) && all(diff(t) > 0), 'gs3dx:club_match', 'T must be finite and increasing');
    local_check_track(ref, n, 'ref');
    local_check_track(sim, n, 'sim');
    local_check_phases(phases, n);

    M.frame = table(t(:), local_err(sim.head - ref.head), local_err(sim.grip - ref.grip), ...
        local_shaft_err(ref, sim), local_face_err(ref, sim), local_speed(ref.head, t), local_speed(sim.head, t), ...
        'VariableNames', {'t', 'head_err_m', 'grip_err_m', 'shaft_err_deg', 'face_err_deg', ...
        'head_speed_ref_mps', 'head_speed_sim_mps'});

    spans = {'address_to_top', phases.address:phases.top; 'top_to_impact', phases.top:phases.impact; ...
        'address_to_impact', phases.address:phases.impact};
    speed_err_pct = 100 * (M.frame.head_speed_sim_mps(phases.impact) - M.frame.head_speed_ref_mps(phases.impact)) / ...
        M.frame.head_speed_ref_mps(phases.impact);
    rows = cell(size(spans, 1), 1);
    for k = 1:size(spans, 1)
        f = M.frame(spans{k, 2}, :);
        rows{k} = table(string(spans{k, 1}), 1000 * rms(f.head_err_m), 1000 * max(f.head_err_m), ...
            1000 * rms(f.grip_err_m), 1000 * max(f.grip_err_m), rms(f.shaft_err_deg), max(f.shaft_err_deg), ...
            rms(f.face_err_deg), max(f.face_err_deg), speed_err_pct, 'VariableNames', {'phase', 'head_rms_mm', ...
            'head_max_mm', 'grip_rms_mm', 'grip_max_mm', 'shaft_rms_deg', 'shaft_max_deg', 'face_rms_deg', ...
            'face_max_deg', 'impact_speed_err_pct'});
    end
    M.phase = vertcat(rows{:});
    M.targets = local_targets();
    a = M.phase(M.phase.phase == "address_to_impact", :);
    M.ok.head_rms = a.head_rms_mm <= M.targets.head_rms_mm;
    M.ok.head_max = a.head_max_mm <= M.targets.head_max_mm;
    M.ok.shaft = a.shaft_max_deg <= M.targets.shaft_deg;
    M.ok.speed = abs(a.impact_speed_err_pct) <= M.targets.speed_pct;
    M.ok.all = M.ok.head_rms && M.ok.head_max && M.ok.shaft && M.ok.speed;
end

function targets = local_targets()
% The #11160 acceptance targets, address to impact, in one place.
    targets = struct('head_rms_mm', 10, 'head_max_mm', 20, 'shaft_deg', 2, 'speed_pct', 2);
end

function local_check_track(s, n, label)
    for f = ["head", "grip", "face_normal"]
        assert(isfield(s, f), 'gs3dx:club_match', '%s is missing field %s', label, f);
        v = s.(f);
        assert(isequal(size(v), [3 n]), 'gs3dx:club_match', '%s.%s is %s, expected [3 %d]', label, f, mat2str(size(v)), n);
        assert(all(isfinite(v), 'all'), 'gs3dx:club_match', '%s.%s has non-finite values', label, f);
    end
    assert(all(abs(vecnorm(s.face_normal) - 1) < 1e-3), 'gs3dx:club_match', '%s.face_normal is not unit length', label);
end

function local_check_phases(p, n)
    for f = ["address", "top", "impact"]
        assert(isfield(p, f), 'gs3dx:club_match', 'phases is missing field %s', f);
    end
    assert(1 <= p.address && p.address < p.top && p.top <= p.impact && p.impact <= n, 'gs3dx:club_match', ...
        'phases must satisfy 1 <= address (%d) < top (%d) <= impact (%d) <= %d', p.address, p.top, p.impact, n);
end

function e = local_err(d)
    e = vecnorm(d).';
end

function a = local_shaft_err(ref, sim)
    ur = local_unit(ref.head - ref.grip);
    us = local_unit(sim.head - sim.grip);
    a = local_angle(ur, us);
end

function a = local_face_err(ref, sim)
% Angle between face normals projected onto the plane normal to REF's shaft.
    shaft = local_unit(ref.head - ref.grip);
    pr = local_unit(local_project(ref.face_normal, shaft));
    ps = local_unit(local_project(sim.face_normal, shaft));
    a = local_angle(pr, ps);
end

function p = local_project(v, axis)
    p = v - sum(v .* axis, 1) .* axis;
end

function u = local_unit(v)
    u = v ./ vecnorm(v);
end

function a = local_angle(u, v)
% Angle in degrees between unit columns of U and V, robust near 0 and 180.
    c = sum(u .* v, 1);
    s = vecnorm(cross(u, v, 1));
    a = rad2deg(atan2(s, c)).';
end

function v = local_speed(head, t)
    rate = zeros(size(head));
    for r = 1:3
        rate(r, :) = gradient(head(r, :), t(:).');
    end
    v = vecnorm(rate).';
end
