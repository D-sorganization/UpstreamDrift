function speed_mph = compute_clubhead_speed_mph(time_s, clubhead_xyz)
%COMPUTE_CLUBHEAD_SPEED_MPH  Per-frame clubhead speed in miles per hour.
%
%   SPEED_MPH = COMPUTE_CLUBHEAD_SPEED_MPH(TIME_S, CLUBHEAD_XYZ) returns an
%   (N,1) column vector of clubhead speeds in mph, computed from the
%   numerical derivative of the clubhead position trajectory.
%
%   Conversion factor:  1 m/s = 2.2369362920544 mph.
%
%   Preconditions:
%     - TIME_S is an (N,1) real, monotonic-non-decreasing vector of seconds.
%     - CLUBHEAD_XYZ is a real (N,3) matrix in metres.
%   Postconditions:
%     - SPEED_MPH is (N,1) double, non-negative.
    arguments
        time_s        (:,1) double {mustBeReal}
        clubhead_xyz  (:,3) double {mustBeReal}
    end
    if size(clubhead_xyz, 1) ~= numel(time_s)
        error("compute_clubhead_speed_mph:sizeMismatch", ...
              "clubhead_xyz must have the same number of rows as time_s.");
    end
    if numel(time_s) < 2
        speed_mph = zeros(numel(time_s), 1);
        return;
    end
    % gradient handles non-uniform spacing and preserves length N.
    vx = gradient(clubhead_xyz(:, 1), time_s);
    vy = gradient(clubhead_xyz(:, 2), time_s);
    vz = gradient(clubhead_xyz(:, 3), time_s);
    speed_mps = sqrt(vx .* vx + vy .* vy + vz .* vz);
    speed_mph = speed_mps * 2.2369362920544;
    assert(all(speed_mph >= 0 | isnan(speed_mph)), ...
        "Postcondition: clubhead speed must be non-negative");
end
