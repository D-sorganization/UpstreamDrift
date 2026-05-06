function opts = default_align_options(overrides)
%DEFAULT_ALIGN_OPTIONS  Canonical options struct for the club-target loaders.
%
%   OPTS = DEFAULT_ALIGN_OPTIONS() returns the default options consumed by
%   load_club_target_excel, load_club_target_c3d, and align_to_simulation_grid.
%
%   OPTS = DEFAULT_ALIGN_OPTIONS(OVERRIDES) merges name-value overrides into
%   the defaults.  Unknown fields raise an error to avoid silent typos.
%
%   Fields:
%     sample_rate         - simulation sample rate in Hz (default 1000).
%     pre_impact_s        - seconds before impact to retain (default 0.25).
%     post_impact_s       - seconds after impact to retain  (default 0.05).
%     expected_impact_s   - simulation-frame target impact time (default 0.25).
%     time_alignment      - "impact" | "address" | "none" (default "impact").
%     verbosity           - "Silent" | "Normal" | "Verbose" | "Debug"
%                           (default "Normal").
%     subject_id          - free-text subject identifier (default "").
%     trial_id            - free-text trial identifier (default "").
    arguments
        overrides (1,1) struct = struct()
    end

    opts = struct( ...
        "sample_rate",       1000, ...
        "pre_impact_s",      0.25, ...
        "post_impact_s",     0.05, ...
        "expected_impact_s", 0.25, ...
        "time_alignment",    "impact", ...
        "verbosity",         "Normal", ...
        "subject_id",        "", ...
        "trial_id",          "");

    fns = fieldnames(overrides);
    valid = string(fieldnames(opts));
    for i = 1:numel(fns)
        f = fns{i};
        if ~ismember(string(f), valid)
            error("default_align_options:unknownField", ...
                  "Unknown options field: %s", f);
        end
        opts.(f) = overrides.(f);
    end
end
