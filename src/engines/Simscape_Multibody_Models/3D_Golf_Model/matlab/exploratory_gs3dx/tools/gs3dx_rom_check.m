function report = gs3dx_rom_check(rom, q)
%GS3DX_ROM_CHECK  How far a joint-angle history leaves the human range of motion (#11158).
%
%   REPORT = GS3DX_ROM_CHECK(ROM, Q) checks Q (height(ROM) x N primitive
%   angles, deg, row k for ROM row k as GS3DX_JOINT_ROM gives it; a row of
%   NaN is not checked) against ROM.  Each primitive becomes the anatomical
%   angle A = SIGN * wrap180(Q - NEUTRAL) (GS3DX_JOINT_ROM).  A row with no
%   neutral is checked on its arc only: the unwrapped max(Q) - min(Q).
%
%   REPORT is ROM's joint, key and motion with:
%     min_deg, max_deg  the anatomical range reached (NaN where unbounded)
%     span_deg          the arc reached
%     excess_deg        how far the history leaves the range: the larger of
%                       MIN - min(A), max(A) - MAX and span - SPAN (<= 0 when
%                       inside; NaN when the row was not checked)
%     frames            number of frames outside [MIN, MAX]
%     ok                excess_deg <= 0 (true for an unchecked row)

    arguments
        rom table
        q (:,:) double
    end
    assert(size(q, 1) == height(rom), 'gs3dx:rom_check', ...
        'Q has %d rows, ROM %d', size(q, 1), height(rom));
    n = height(rom);
    report = rom(:, {'joint', 'key', 'motion'});
    [report.min_deg, report.max_deg, report.span_deg, report.excess_deg] = deal(nan(n, 1));
    report.frames = zeros(n, 1);
    for k = 1:n
        v = q(k, :);
        if all(isnan(v))
            continue
        end
        assert(all(isfinite(v)), 'gs3dx:rom_check', 'Row %d (%s) is partly missing', k, rom.key(k));
        arc = rad2deg(unwrap(deg2rad(v)));
        report.span_deg(k) = max(arc) - min(arc);
        excess = report.span_deg(k) - rom.span_deg(k);
        if ~isnan(rom.neutral_deg(k))
            a = rom.sign(k) * wrap180(v - rom.neutral_deg(k));
            report.min_deg(k) = min(a);
            report.max_deg(k) = max(a);
            excess = max([excess, rom.min_deg(k) - min(a), max(a) - rom.max_deg(k)]);
            report.frames(k) = nnz(a < rom.min_deg(k) | a > rom.max_deg(k));
        end
        report.excess_deg(k) = excess;
    end
    report.ok = ~(report.excess_deg > 0);
end

function a = wrap180(a)
    a = mod(a + 180, 360) - 180;
end
