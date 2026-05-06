function probe_swing_sheets(xlsx_path)
%PROBE_SWING_SHEETS  Inspect each sheet of a Wiffle-style xlsx mocap file.
%
%   PROBE_SWING_SHEETS(XLSX_PATH) iterates over the four canonical sheets
%   ("TW_wiffle", "TW_ProV1", "GW_wiffle", "GW_ProV11") in XLSX_PATH and
%   prints, for each:
%     * Frames after NaN-filter
%     * Event markers (A/T/I/F sample numbers + CHS_mph) from the row-1 header
%     * Median mid-hands -> clubhead distance (sanity-check 0.7 - 1.4 m)
%     * First-frame and impact-frame raw values (grip/clubhead positions)
%
%   No XLSX_PATH provided -> probes the canonical file under
%   src/apps/golf_gui/Motion Capture Plotter/.
%
%   This script is a developer probe; it never asserts.  It catches loader
%   errors and prints them so all four sheets get reported even if some fail.

    if nargin < 1 || strlength(string(xlsx_path)) == 0
        here = fileparts(mfilename("fullpath"));
        xlsx_path = fullfile(here, "..", "..", "..", "src", "apps", ...
            "golf_gui", "Motion Capture Plotter", ...
            "Wiffle_ProV1_club_3D_data.xlsx");
    end
    xlsx_path = string(xlsx_path);

    if exist(xlsx_path, "file") ~= 2
        error("probe_swing_sheets:fileMissing", ...
              "xlsx not found at %s", xlsx_path);
    end

    fprintf("=== probe_swing_sheets: %s ===\n", xlsx_path);

    sheets = ["TW_wiffle", "TW_ProV1", "GW_wiffle", "GW_ProV11"];
    for k = 1:numel(sheets)
        sheet = sheets(k);
        fprintf("\n--- Sheet: %s ---\n", sheet);
        try
            t = load_club_target_excel(xlsx_path, sheet);
        catch ME
            fprintf("  LOAD FAILED: %s\n", ME.message);
            continue;
        end
        N = numel(t.time);
        fprintf("  Frames (post NaN-filter): %d\n", N);
        ev = t.events;
        fprintf("  Events: A=%g  T=%g  I=%g  F=%g  CHS_mph=%g\n", ...
            ev.A_sample, ev.T_sample, ev.I_sample, ev.F_sample, ev.CHS_mph);
        shaft = vecnorm(t.clubhead - t.grip, 2, 2);
        fprintf("  Shaft length (m): median=%.3f  min=%.3f  max=%.3f\n", ...
            median(shaft), min(shaft), max(shaft));
        ok = median(shaft) >= 0.7 && median(shaft) <= 1.4;
        if ok
            fprintf("    -> within 0.7-1.4 m (PLAUSIBLE)\n");
        else
            fprintf("    -> OUTSIDE 0.7-1.4 m (suspect)\n");
        end
        fprintf("  Frame[1]: grip=[%.3f %.3f %.3f]  clubhead=[%.3f %.3f %.3f]\n", ...
            t.grip(1,1), t.grip(1,2), t.grip(1,3), ...
            t.clubhead(1,1), t.clubhead(1,2), t.clubhead(1,3));
        ii = t.impact_idx;
        fprintf("  Frame[impact_idx=%d, t=%.3fs]: grip=[%.3f %.3f %.3f]  clubhead=[%.3f %.3f %.3f]\n", ...
            ii, t.time(ii), ...
            t.grip(ii,1), t.grip(ii,2), t.grip(ii,3), ...
            t.clubhead(ii,1), t.clubhead(ii,2), t.clubhead(ii,3));
    end
    fprintf("\n=== probe complete ===\n");
end
