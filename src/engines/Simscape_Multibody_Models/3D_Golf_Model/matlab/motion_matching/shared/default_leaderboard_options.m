function opts = default_leaderboard_options()
%DEFAULT_LEADERBOARD_OPTIONS  Canonical options struct for leaderboard.m.
%
%   opts = DEFAULT_LEADERBOARD_OPTIONS() returns the default options struct
%   consumed by leaderboard.m. See VISUALIZATION_SPEC.md §"Comparison
%   across options" and issue #3992 for rationale.
%
%   Fields:
%     .sort_by         (string) column to sort ascending by   ("rmse_mm")
%                      one of "rmse_mm" | "wall_s" | "work_J"
%     .filter_swing_id (string) restrict rows to this swing   ("")
%                      empty string disables the filter
%     .write_csv       (logical) export tbl to .csv on call   (false)
%     .csv_path        (string) destination .csv path         ("leaderboard.csv")
%     .recheck         (logical) recompute rmse_mm via        (false)
%                      compute_cost on saved coefficients to verify
%                      provenance to within 1e-6
%     .format          (string) output format hint            ("table")
%                      one of "table" | "csv" | "markdown".
%                      Drives the second return value of leaderboard().
%     .build_figure    (logical) build comparison figure      (false)
%                      with uitable + grouped bar chart
%
%   Postconditions:
%     - All fields are present so callers can mutate selectively without
%       worrying about isfield gates.
    opts = struct();
    opts.sort_by         = "rmse_mm";
    opts.filter_swing_id = "";
    opts.write_csv       = false;
    opts.csv_path        = "leaderboard.csv";
    opts.recheck         = false;
    opts.format          = "table";
    opts.build_figure    = false;
end
