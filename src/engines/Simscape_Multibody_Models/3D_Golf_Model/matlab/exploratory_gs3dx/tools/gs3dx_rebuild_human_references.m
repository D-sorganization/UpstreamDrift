function gs3dx_rebuild_human_references()
%GS3DX_REBUILD_HUMAN_REFERENCES  Rewrite GS3DX_Human joint refs from ROM-limited IK (#11158).
%   One-shot batch helper; see tests/test_gs3dx_joint_rom.m and docs/ROM.md.

    addpath(fileparts(fileparts(mfilename('fullpath'))));
    info = gs3dx_setup();
    rom = gs3dx_joint_rom();
    names = gs3dx_names();
    mdl = char(names.variants.human);
    fit = char(names.variants.fit);

    saved = false;
    max_excess_before = NaN;
    max_excess_after = NaN;

    jc = gs3dx_capture_joint_centres(gs3dx_capture_markers());
    f0 = jc.impact_frame;
    frames = unique([1:10:f0 f0]);
    cal = 1:15:f0 - 90;
    reg = {'posture_weight', 0.01, 'smooth_weight', 0.02, 'backward', false, 'gap_weight', 0.5};
    off = gs3dx_whole_body_ik(jc, frames=1, calibration_frames=cal).offsets;
    free = gs3dx_whole_body_ik(jc, 'frames', frames, 'offsets', off, reg{:});
    held = gs3dx_whole_body_ik(jc, 'frames', frames, 'offsets', off, reg{:}, 'rom_weight', 3);

    rms_free = mean(free.rms);
    rms_held = mean(held.rms);
    gate_lim = rms_free + 0.015;
    if rms_held > gate_lim
        fprintf('REF_GATE fail held=%.4f free=%.4f limit=%.4f\n', rms_held, rms_free, gate_lim);
        fprintf('GS3DX_BATCH_DONE\n');
        fprintf('SAVED 0\n');
        return;
    end
    fprintf('REF_GATE pass held=%.4f free=%.4f limit=%.4f\n', rms_held, rms_free, gate_lim);

    q_ik = gs3dx_rom_from_ik(held, rom);
    ik_rows = any(~isnan(q_ik), 2);
    bounded_ik = ik_rows & ~isnan(rom.min_deg);
    arc_ik = ik_rows & isnan(rom.neutral_deg);

    r_before = gs3dx_rom_check(rom, q_ik);
    max_excess_before = max(r_before.excess_deg(bounded_ik));
    fprintf('MAX_EXCESS before projection %.4f deg\n', max_excess_before);

    q_proj = q_ik;
    [q_proj, ik_abort] = local_project_bounded(rom, q_proj, 1, bounded_ik);
    if ik_abort
        fprintf('IK projection abort: bounded sample > 1 deg outside\n');
        fprintf('GS3DX_BATCH_DONE\n');
        fprintf('SAVED 0\n');
        return;
    end
    q_proj = local_project_arc(rom, q_proj, arc_ik);
    r_after = gs3dx_rom_check(rom, q_proj);
    max_excess_after = max(r_after.excess_deg(bounded_ik));
    fprintf('MAX_EXCESS after projection %.4f deg\n', max_excess_after);

    load_system(mdl);
    c = onCleanup(@() close_system(mdl, 0));
    ws = get_param(mdl, 'ModelWorkspace');
    T = ws.getVariable('LegReferenceTime');
    q_stored = gs3dx_rom_reference(ws, rom);

    leg_neck = rom.track_row > 0 & (rom.track == "LegReferenceAngle" | rom.track == "NeckReference");
    r_store = gs3dx_rom_check(rom, q_stored);
    leg_ex = r_store.excess_deg(leg_neck);
    if any(round(leg_ex, 0) > 2)
        bad = r_store(leg_neck & round(r_store.excess_deg, 0) > 2, {'joint', 'motion', 'excess_deg'});
        fprintf('REF_LEG fail\n');
        disp(bad);
        fprintf('GS3DX_BATCH_DONE\n');
        fprintf('SAVED 0\n');
        return;
    end

    q_new = q_stored;
    fix_leg = leg_neck & r_store.excess_deg > 0 & round(r_store.excess_deg, 0) <= 2;
    if any(fix_leg)
        q_leg = q_stored;
        fix_bounded = fix_leg & ~isnan(rom.min_deg);
        fix_arc = fix_leg & isnan(rom.neutral_deg);
        for k = find(fix_bounded(:).')
            row_tol = max(2, r_store.excess_deg(k) * (1 + 1e-9));
            rows1 = false(height(rom), 1);
            rows1(k) = true;
            [q_leg, leg_abort] = local_project_bounded(rom, q_leg, row_tol, rows1);
            if leg_abort
                fprintf('REF_LEG projection abort\n');
                fprintf('GS3DX_BATCH_DONE\n');
                fprintf('SAVED 0\n');
                return;
            end
        end
        q_leg = local_project_arc(rom, q_leg, fix_arc);
        q_new(fix_leg, :) = q_leg(fix_leg, :);
        fprintf('REF_LEG projected rows: %s\n', strjoin(string(unique(rom.joint(fix_leg))), ', '));
    end

    track_angle = rom.track_row > 0 & endsWith(string(rom.track), "TrackAngle");
    for k = find(track_angle(:).')
        if ~any(~isnan(q_proj(k, :)))
            continue;
        end
        row = q_proj(k, :);
        if isnan(rom.neutral_deg(k))
            if max(abs(diff(row))) > 180
                row = rad2deg(unwrap(deg2rad(row)));
            end
            q_new(k, :) = interp1(held.t, row, T, 'linear', 'extrap');
        else
            a = rom.sign(k) * local_wrap180(row - rom.neutral_deg(k));
            a_full = interp1(held.t, a, T, 'linear', 'extrap');
            q_new(k, :) = rom.neutral_deg(k) + rom.sign(k) * a_full;
        end
    end

    bounded_track = track_angle & ~isnan(rom.min_deg);
    q_new = local_clamp_bounded(rom, q_new, bounded_track);
    q_new = local_project_arc(rom, q_new, arc_ik);

    r_final = gs3dx_rom_check(rom, q_new);
    if any(r_final.excess_deg > 0)
        bad = r_final(~r_final.ok, {'joint', 'motion', 'excess_deg'});
        fprintf('Final ROM check fail\n');
        disp(bad);
        fprintf('GS3DX_BATCH_DONE\n');
        fprintf('SAVED 0\n');
        return;
    end

    b = gs3dx_block_budget(mdl, compiled=true);
    if b.compiled_total > 975
        fprintf('Block budget %d > 975\n', b.compiled_total);
        fprintf('GS3DX_BATCH_DONE\n');
        fprintf('SAVED 0\n');
        return;
    end

    local_write_references(ws, rom, q_new);
    gs3dx_save_model(mdl, info);
    saved = true;
    fprintf('SAVED 1\n');
    fprintf('GS3DX_BATCH_DONE\n');
end

function q_out = local_clamp_bounded(rom, q, rows)
    q_out = q;
    for k = find(rows(:).')
        v = q(k, :);
        if all(isnan(v))
            continue;
        end
        for j = 1:numel(v)
            if isnan(v(j))
                continue;
            end
            a = rom.sign(k) * local_wrap180(v(j) - rom.neutral_deg(k));
            a = min(max(a, rom.min_deg(k)), rom.max_deg(k));
            q_out(k, j) = rom.neutral_deg(k) + rom.sign(k) * a;
        end
    end
end

function [q_out, abort] = local_project_bounded(rom, q, tol_deg, rows)
    q_out = q;
    abort = false;
    for k = find(rows(:).')
        v = q(k, :);
        if all(isnan(v))
            continue;
        end
        for j = 1:numel(v)
            if isnan(v(j))
                continue;
            end
            a = rom.sign(k) * local_wrap180(v(j) - rom.neutral_deg(k));
            ex = 0;
            if a < rom.min_deg(k)
                ex = rom.min_deg(k) - a;
            elseif a > rom.max_deg(k)
                ex = a - rom.max_deg(k);
            end
            if ex > tol_deg
                abort = true;
                return;
            end
            if ex > 0
                a = min(max(a, rom.min_deg(k)), rom.max_deg(k));
                q_out(k, j) = rom.neutral_deg(k) + rom.sign(k) * a;
            end
        end
    end
end

function q_out = local_project_arc(rom, q, rows)
    q_out = q;
    for k = find(rows(:).')
        v = q(k, :);
        if all(isnan(v))
            continue;
        end
        vu = rad2deg(unwrap(deg2rad(v)));
        span = max(vu) - min(vu);
        if span > rom.span_deg(k) && span > 0
            vu = min(vu) + (vu - min(vu)) * (rom.span_deg(k) / span);
            q_out(k, :) = vu;
        end
    end
end

function local_write_references(ws, rom, q)
    tracks = unique(rom.track(rom.track ~= ""));
    for ti = 1:numel(tracks)
        name = char(tracks(ti));
        if ~hasVariable(ws, name)
            continue;
        end
        v = ws.getVariable(name);
        rows = find(rom.track == tracks(ti));
        for k = rows(:).'
            if all(isnan(q(k, :)))
                continue;
            end
            row = q(k, :);
            if strcmp(name, 'NeckReference')
                row = deg2rad(row);
            end
            v(rom.track_row(k), :) = row;
        end
        assignin(ws, name, v);
    end
end

function a = local_wrap180(a)
    a = mod(a + 180, 360) - 180;
end
