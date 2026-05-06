function target = to_canonical_target(tbl, opts)
%TO_CANONICAL_TARGET  Adapt a MachineLearning table to the canonical target.
%
%   TARGET = ML_COMPAT.TO_CANONICAL_TARGET(TBL) accepts either of the two
%   MachineLearning column conventions used in the Simscape 3D golf workflow:
%
%     - clubface_x/y/z, clubface_vx/vy/vz, clubface_ax/ay/az  (measured)
%     - ClubLogs_CHGlobalPosition_{1,2,3}, ClubLogs_CHGlobalVelocity_{1,2,3},
%       ClubLogs_CHGlobalAcceleration_{1,2,3}                  (sim output)
%
%   and produces a struct matching the canonical target schema:
%       .time, .butt, .clubhead, .club_quat, .impact_idx, .source
%
%   LOSSY: neither MachineLearning convention carries the butt-end position
%   or an orientation quaternion. The butt is reconstructed as
%   clubhead - shaft_length * v_hat, and the quaternion from the tangent
%   frame X = v_hat, Y orthogonalised acceleration, Z = X x Y. A warning is
%   issued on every call.
%
%   TARGET = ML_COMPAT.TO_CANONICAL_TARGET(TBL, OPTS) overrides:
%       opts.shaft_length      default 1.143 (m)
%       opts.subject_id        default "ML"
%       opts.trial_id          default "0"
%
%   See also: ML_COMPAT.TO_MACHINELEARNING_COLUMNS

    arguments
        tbl table
        opts.shaft_length (1,1) double = 1.143
        opts.subject_id (1,1) string = "ML"
        opts.trial_id (1,1) string = "0"
    end

    warning("ml_compat:lossy", ...
        "MachineLearning -> canonical target is lossy: butt and " + ...
        "club_quat are reconstructed.");

    cols = string(tbl.Properties.VariableNames);
    isClubface = all(ismember(["clubface_x","clubface_y","clubface_z"], cols));
    isClubLogs = all(ismember([...
        "ClubLogs_CHGlobalPosition_1", ...
        "ClubLogs_CHGlobalPosition_2", ...
        "ClubLogs_CHGlobalPosition_3"], cols));
    if ~(isClubface || isClubLogs)
        error("ml_compat:badInput", ...
            "Table must have clubface_* or ClubLogs_CHGlobalPosition_* columns.");
    end

    if isClubface
        posCols = ["clubface_x","clubface_y","clubface_z"];
        velCols = ["clubface_vx","clubface_vy","clubface_vz"];
        accCols = ["clubface_ax","clubface_ay","clubface_az"];
    else
        posCols = "ClubLogs_CHGlobalPosition_" + (1:3);
        velCols = "ClubLogs_CHGlobalVelocity_" + (1:3);
        accCols = "ClubLogs_CHGlobalAcceleration_" + (1:3);
    end

    n = height(tbl);
    if n < 2
        error("ml_compat:tooShort", "Table must have >= 2 rows.");
    end
    if ismember("time", cols)
        time = double(tbl.time(:));
    else
        time = (0:n-1).' / 1000.0;
    end
    time = time - time(1);

    clubhead = double(tbl{:, posCols});
    if all(ismember(velCols, cols))
        velocity = double(tbl{:, velCols});
    else
        velocity = gradient_along(clubhead, time);
    end
    if all(ismember(accCols, cols))
        acceleration = double(tbl{:, accCols});
    else
        acceleration = gradient_along(velocity, time);
    end

    quat = quat_from_va(velocity, acceleration);
    butt = butt_from_clubhead(clubhead, velocity, opts.shaft_length);

    [~, impact] = max(vecnorm(velocity, 2, 2));
    if ismember("impact_idx", cols) && ~isempty(tbl.impact_idx)
        impact = double(tbl.impact_idx(1));
    end
    impact = max(1, min(impact, n));

    target = struct( ...
        "time", time, ...
        "butt", butt, ...
        "clubhead", clubhead, ...
        "club_quat", quat, ...
        "impact_idx", impact, ...
        "source", struct( ...
            "filename", "ml_compat.dataframe", ...
            "format", "machinelearning", ...
            "subject_id", opts.subject_id, ...
            "trial_id", opts.trial_id, ...
            "sha256", repmat('0', 1, 64)));
end

function out = gradient_along(x, t)
    out = zeros(size(x));
    if size(x,1) > 1 && all(isfinite(t)) && numel(unique(t)) > 1
        for c = 1:size(x,2)
            out(:,c) = gradient(x(:,c), t);
        end
    else
        out(:) = NaN;
    end
end

function butt = butt_from_clubhead(clubhead, velocity, shaft)
    speeds = vecnorm(velocity, 2, 2);
    safe = speeds > 1e-9;
    direction = nan(size(velocity));
    if any(safe)
        direction(safe, :) = velocity(safe, :) ./ speeds(safe);
    end
    butt = clubhead - shaft * direction;
    bad = ~all(isfinite(butt), 2);
    if any(bad)
        butt(bad, :) = clubhead(bad, :) - repmat([0 0 shaft], nnz(bad), 1);
    end
end

function q = quat_from_va(v, a)
    n = size(v, 1);
    q = zeros(n, 4);
    for i = 1:n
        vi = v(i,:);
        ai = a(i,:);
        vn = norm(vi);
        if ~isfinite(vn) || vn < 1e-9
            q(i,:) = [1 0 0 0];
            continue;
        end
        x_hat = vi / vn;
        a_perp = ai - dot(ai, x_hat) * x_hat;
        an = norm(a_perp);
        if ~isfinite(an) || an < 1e-9
            q(i,:) = [1 0 0 0];
            continue;
        end
        y_hat = a_perp / an;
        z_hat = cross(x_hat, y_hat);
        R = [x_hat(:), y_hat(:), z_hat(:)];
        q(i,:) = rotmat_to_quat(R);
    end
end

function q = rotmat_to_quat(R)
    tr = R(1,1) + R(2,2) + R(3,3);
    if tr > 0
        s = 0.5 / sqrt(tr + 1.0);
        w = 0.25 / s;
        x = (R(3,2) - R(2,3)) * s;
        y = (R(1,3) - R(3,1)) * s;
        z = (R(2,1) - R(1,2)) * s;
    elseif (R(1,1) > R(2,2)) && (R(1,1) > R(3,3))
        s = 2 * sqrt(1 + R(1,1) - R(2,2) - R(3,3));
        w = (R(3,2) - R(2,3)) / s;
        x = 0.25 * s;
        y = (R(1,2) + R(2,1)) / s;
        z = (R(1,3) + R(3,1)) / s;
    elseif R(2,2) > R(3,3)
        s = 2 * sqrt(1 + R(2,2) - R(1,1) - R(3,3));
        w = (R(1,3) - R(3,1)) / s;
        x = (R(1,2) + R(2,1)) / s;
        y = 0.25 * s;
        z = (R(2,3) + R(3,2)) / s;
    else
        s = 2 * sqrt(1 + R(3,3) - R(1,1) - R(2,2));
        w = (R(2,1) - R(1,2)) / s;
        x = (R(1,3) + R(3,1)) / s;
        y = (R(2,3) + R(3,2)) / s;
        z = 0.25 * s;
    end
    q = [w x y z];
    nrm = norm(q);
    if nrm == 0
        q = [1 0 0 0];
    else
        q = q / nrm;
    end
    if q(1) < 0
        q = -q;
    end
end
