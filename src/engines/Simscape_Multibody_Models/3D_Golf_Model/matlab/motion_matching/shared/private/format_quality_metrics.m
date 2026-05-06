function metrics = format_quality_metrics(result, target)
%FORMAT_QUALITY_METRICS  Compute and format the metrics block for View 3.
%
%   METRICS = FORMAT_QUALITY_METRICS(RESULT, TARGET) returns a struct of
%   pre-formatted metric strings used by plot_fit_quality_card.m.  Keeping
%   this in a private helper preserves the LOD<=2 rule and lets the View 3
%   tests verify exact text formatting independently of the figure layout.
%
%   Output fields (all string):
%     .clubhead_rmse_mm   "2.3 mm" (one decimal)
%     .butt_rmse_mm       "1.8 mm" (one decimal)
%     .orient_err_deg     "0.41 deg" (two decimals)
%     .speed_at_impact    "112 mph (meas: 111)"
%     .total_work_J       "284 J"
%     .peak_power_kW      "1.2 kW (LE)" or "n/a" if not populated
%
%   Preconditions:
%     - RESULT is a scalar struct.  Required fields:
%         final_rmse_m   (scalar) OR sim_out.{r_clubhead,r_butt,...}
%       Optional:
%         final_total_work_J, peak_joint_power_W, peak_joint_name,
%         clubhead_speed_at_impact_mph, sim_out
%     - TARGET is a scalar struct conforming to CLUB_IK_SPEC.md
%       (time, butt, clubhead, club_quat, impact_idx).
%   Postconditions:
%     - METRICS has all six fields above as scalar strings.
%
%   GitHub issue: #3991.
    arguments
        result (1,1) struct
        target (1,1) struct
    end

    [ch_rmse_mm, butt_rmse_mm] = local_position_rmses_mm(result, target);
    ori_deg = local_mean_orientation_error_deg(result, target);
    [speed_sim_mph, speed_meas_mph] = local_impact_speeds_mph(result, target);
    work_J = local_get_scalar(result, "final_total_work_J", NaN);
    [peak_kW, peak_name] = local_peak_power(result);

    metrics = struct();
    metrics.clubhead_rmse_mm = local_fmt_mm(ch_rmse_mm);
    metrics.butt_rmse_mm     = local_fmt_mm(butt_rmse_mm);
    metrics.orient_err_deg   = local_fmt_deg(ori_deg);
    metrics.speed_at_impact  = local_fmt_speed(speed_sim_mph, speed_meas_mph);
    metrics.total_work_J     = local_fmt_work(work_J);
    metrics.peak_power_kW    = local_fmt_peak(peak_kW, peak_name);

    % Postcondition: all required fields populated as strings.
    required = ["clubhead_rmse_mm", "butt_rmse_mm", "orient_err_deg", ...
                "speed_at_impact", "total_work_J", "peak_power_kW"];
    for f = required
        assert(isfield(metrics, f) && (isstring(metrics.(f)) || ischar(metrics.(f))), ...
            "Postcondition: format_quality_metrics must return string field %s", f);
    end
end

% =====================================================================
% Local helpers (LOD <= 2)
% =====================================================================
function [ch_mm, butt_mm] = local_position_rmses_mm(result, target)
    ch_mm = NaN;
    butt_mm = NaN;
    if isfield(result, "sim_out") && isstruct(result.sim_out)
        s = result.sim_out;
        ch_sim = local_pick(s, ["r_clubhead", "clubhead"]);
        butt_sim = local_pick(s, ["r_butt", "butt"]);
        if ~isempty(ch_sim) && size(ch_sim, 1) == size(target.clubhead, 1)
            err_m = compute_pointwise_position_error(ch_sim, target.clubhead);
            ch_mm = sqrt(mean(err_m .^ 2)) * 1000;
        end
        if ~isempty(butt_sim) && size(butt_sim, 1) == size(target.butt, 1)
            err_m = compute_pointwise_position_error(butt_sim, target.butt);
            butt_mm = sqrt(mean(err_m .^ 2)) * 1000;
        end
    end
    if isnan(ch_mm) && isfield(result, "final_rmse_m") && isscalar(result.final_rmse_m)
        ch_mm = double(result.final_rmse_m) * 1000;
    end
end

function deg = local_mean_orientation_error_deg(result, target)
    deg = NaN;
    if isfield(result, "sim_out") && isstruct(result.sim_out)
        s = result.sim_out;
        q_sim = local_pick(s, ["club_quat", "q_club"]);
        if ~isempty(q_sim) && size(q_sim, 1) == size(target.club_quat, 1)
            errs = compute_pointwise_orientation_error(q_sim, target.club_quat);
            deg = mean(errs(~isnan(errs)));
        end
    end
    if isnan(deg) && isfield(result, "final_mean_orientation_deg")
        deg = double(result.final_mean_orientation_deg);
    end
end

function [sim_mph, meas_mph] = local_impact_speeds_mph(result, target)
    sim_mph = NaN;
    meas_mph = NaN;
    impact_idx = double(target.impact_idx);
    impact_idx = max(1, min(impact_idx, numel(target.time)));
    speeds = compute_clubhead_speed_mph(target.time, target.clubhead);
    meas_mph = speeds(impact_idx);
    if isfield(result, "sim_out") && isstruct(result.sim_out)
        s = result.sim_out;
        ch = local_pick(s, ["r_clubhead", "clubhead"]);
        t = local_pick(s, "time");
        if ~isempty(ch) && ~isempty(t)
            sim_speeds = compute_clubhead_speed_mph(t, ch);
            sim_mph = sim_speeds(min(impact_idx, numel(sim_speeds)));
        end
    end
    if isnan(sim_mph) && isfield(result, "clubhead_speed_at_impact_mph")
        sim_mph = double(result.clubhead_speed_at_impact_mph);
    end
end

function [kW, name] = local_peak_power(result)
    kW = NaN;
    name = "";
    if isfield(result, "peak_joint_power_W") && isscalar(result.peak_joint_power_W)
        kW = double(result.peak_joint_power_W) / 1000;
    elseif isfield(result, "peak_joint_power_kW") && isscalar(result.peak_joint_power_kW)
        kW = double(result.peak_joint_power_kW);
    end
    if isfield(result, "peak_joint_name")
        name = string(result.peak_joint_name);
    end
end

function v = local_pick(s, names)
    v = [];
    for n = string(names)
        if isfield(s, n)
            v = s.(n);
            return;
        end
    end
end

function v = local_get_scalar(s, name, default)
    if isfield(s, name) && isscalar(s.(name))
        v = double(s.(name));
    else
        v = default;
    end
end

function txt = local_fmt_mm(x)
    if isnan(x), txt = "n/a"; else, txt = sprintf("%.1f mm", x); end
end

function txt = local_fmt_deg(x)
    if isnan(x), txt = "n/a"; else, txt = sprintf("%.2f deg", x); end
end

function txt = local_fmt_speed(sim_mph, meas_mph)
    if isnan(sim_mph) && isnan(meas_mph)
        txt = "n/a";
    elseif isnan(sim_mph)
        txt = sprintf("n/a (meas: %.0f mph)", meas_mph);
    elseif isnan(meas_mph)
        txt = sprintf("%.0f mph", sim_mph);
    else
        txt = sprintf("%.0f mph (meas: %.0f)", sim_mph, meas_mph);
    end
end

function txt = local_fmt_work(x)
    if isnan(x), txt = "n/a"; else, txt = sprintf("%.0f J", x); end
end

function txt = local_fmt_peak(kW, name)
    if isnan(kW)
        txt = "n/a";
        return;
    end
    if strlength(name) > 0
        txt = sprintf("%.1f kW (%s)", kW, name);
    else
        txt = sprintf("%.1f kW", kW);
    end
end
