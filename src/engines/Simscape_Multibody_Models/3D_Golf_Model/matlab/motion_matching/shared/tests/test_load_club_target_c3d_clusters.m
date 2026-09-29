classdef test_load_club_target_c3d_clusters < matlab.unittest.TestCase
%TEST_LOAD_CLUB_TARGET_C3D_CLUSTERS  Validated cluster-marker C3D regression.
%
%   These tests assert against the externally-validated targets recorded in
%   PR #3982 (ezc3d 1.7.0):
%     - C3DExport Tour average.c3d        -> driver, 114.2 mph at impact
%                                            (frame 475, t=1.319 s)
%     - C3DExport tour average iron.c3d   -> iron,    88.6 mph at impact
%                                            (frame 478, t=1.331 s)
%
%   Tests that need the actual C3D files are gated via assumeFile so the
%   suite degrades gracefully if the data is absent.

    properties (Constant)
        DriverFile = "C3DExport Tour average.c3d";
        IronFile   = "C3DExport tour average iron.c3d";
        DriverImpactMph = 114.2;
        IronImpactMph   = 88.6;
        SpeedTolFrac    = 0.05;  % +/- 5%
        MpsToMph        = 2.2369362920544;
    end

    methods (TestClassSetup)
        function add_shared_to_path(testCase)
            here = fileparts(mfilename("fullpath"));
            shared_dir = fullfile(here, "..");
            addpath(shared_dir);
            testCase.addTeardown(@() rmpath(shared_dir));
        end
    end

    methods (Test)
        function test_loads_cluster_driver_returns_canonical_target(testCase)
            c3d = locate_c3d(testCase, testCase.DriverFile);
            opts = default_align_options();
            target = load_club_target_c3d(c3d, opts);
            verify_canonical_target(testCase, target);
        end

        function test_loads_cluster_iron_returns_canonical_target(testCase)
            c3d = locate_c3d(testCase, testCase.IronFile);
            opts = default_align_options();
            target = load_club_target_c3d(c3d, opts);
            verify_canonical_target(testCase, target);
        end

        function test_clubhead_speed_at_impact_within_5_pct_driver(testCase)
            c3d = locate_c3d(testCase, testCase.DriverFile);
            opts = default_align_options();
            opts.sample_rate_hz = 360;
            opts.simulation_time_s = 1.6;
            opts.impact_target_t_s = 1.319;
            target = load_club_target_c3d(c3d, opts);
            mph = peak_clubhead_speed_mph(target, testCase.MpsToMph);
            tol = testCase.DriverImpactMph * testCase.SpeedTolFrac;
            testCase.verifyEqual(mph, testCase.DriverImpactMph, "AbsTol", tol, ...
                sprintf("Driver impact %.2f mph outside +/-5%% of %.1f", ...
                        mph, testCase.DriverImpactMph));
        end

        function test_clubhead_speed_at_impact_within_5_pct_iron(testCase)
            c3d = locate_c3d(testCase, testCase.IronFile);
            opts = default_align_options();
            opts.sample_rate_hz = 359;
            opts.simulation_time_s = 1.6;
            opts.impact_target_t_s = 1.331;
            target = load_club_target_c3d(c3d, opts);
            mph = peak_clubhead_speed_mph(target, testCase.MpsToMph);
            tol = testCase.IronImpactMph * testCase.SpeedTolFrac;
            testCase.verifyEqual(mph, testCase.IronImpactMph, "AbsTol", tol);
        end

        function test_units_metres_no_inch_conversion(testCase)
            % Sanity: shaft length plausible for a driver (~1.0 - 1.2 m).
            c3d = locate_c3d(testCase, testCase.DriverFile);
            target = load_club_target_c3d(c3d, default_align_options());
            shaft_lengths = vecnorm(target.clubhead - target.butt, 2, 2);
            mean_shaft = mean(shaft_lengths, "omitnan");
            testCase.verifyGreaterThan(mean_shaft, 0.7, ...
                "Shaft length implausibly small; possible inch conversion bug");
            testCase.verifyLessThan(mean_shaft, 1.4, ...
                "Shaft length implausibly large");
        end

        function test_y_up_to_z_up_preserves_right_handed(testCase)
            here = fileparts(mfilename("fullpath"));
            addpath(fullfile(here, "..", "private"));
            % 3x3 rotation matrix should have det == +1.
            R = y_to_z_up(eye(3));
            testCase.verifyEqual(det(R), 1.0, "AbsTol", 1e-12);
            % Right-handed triad preserved.
            triad = eye(3);
            swapped = y_to_z_up(triad);
            cross_xy = cross(swapped(1, :), swapped(2, :));
            testCase.verifyEqual(cross_xy, swapped(3, :), "AbsTol", 1e-12);
        end

        function test_cluster_pose_against_known_rotation(testCase)
            here = fileparts(mfilename("fullpath"));
            addpath(fullfile(here, "..", "private"));
            reference = [ ...
                 1.0, 0.0, 0.0; ...
                -0.5, sqrt(3)/2, 0.0; ...
                -0.5, -sqrt(3)/2, 0.0];
            angle = 0.7;
            R_true = [cos(angle), -sin(angle), 0; ...
                      sin(angle),  cos(angle), 0; ...
                      0,           0,          1];
            translation = [0.1, -0.2, 0.3];
            moved = (reference * R_true.') + translation;
            cluster_t = reshape(moved, [3, 3, 1]);
            [R, c] = pose_from_cluster(cluster_t, reference);
            testCase.verifyEqual(R(:,:,1), R_true, "AbsTol", 1e-9);
            testCase.verifyEqual(c(1, :), translation, "AbsTol", 1e-9);
        end

        function test_short_nan_gaps_filled_via_spline_interpolation(testCase)
            % Indirectly validated: the loader must produce all-finite
            % butt/clubhead despite ~20-30 NaN gaps in the cluster markers.
            c3d = locate_c3d(testCase, testCase.DriverFile);
            target = load_club_target_c3d(c3d, default_align_options());
            testCase.verifyTrue(all(isfinite(target.butt(:))));
            testCase.verifyTrue(all(isfinite(target.clubhead(:))));
        end

        function test_sentinel_and_occluded_markers_excluded(testCase)
            here = fileparts(mfilename("fullpath"));
            addpath(fullfile(here, "..", "private"));
            map = cluster_marker_map();
            testCase.verifyTrue(any(map.excluded_markers == "Marker_0:0:0"));
            testCase.verifyTrue(any(map.excluded_markers == "RShoulderTop"));

            c3d = locate_c3d(testCase, testCase.DriverFile);
            target = load_club_target_c3d(c3d, default_align_options());
            % Sentinel value (-1.71, 0.79, -1.97) -- check no clubhead row is
            % clinging to it (after Y->Z swap: (-1.71, 1.97, 0.79)).
            sentinel_z_up = [-1.71, 1.97, 0.79];
            d = vecnorm(target.clubhead - sentinel_z_up, 2, 2);
            testCase.verifyGreaterThan(min(d), 0.05, ...
                "Sentinel marker leaked into the clubhead trace");
        end
    end
end


function c3d = locate_c3d(testCase, filename)
    here = fileparts(mfilename("fullpath"));
    base = fullfile(here, "..", "..", "..", "Data", "Mocap C3D Files");
    c3d_path = fullfile(base, filename);
    testCase.assumeTrue(exist(c3d_path, "file") == 2, ...
        sprintf("C3D file not present: %s", c3d_path));
    c3d = string(c3d_path);
end


function verify_canonical_target(testCase, target)
    testCase.verifyTrue(isstruct(target));
    for f = ["time", "butt", "clubhead", "club_quat", "impact_idx", "source"]
        testCase.verifyTrue(isfield(target, f), sprintf("Missing field: %s", f));
    end
    N = numel(target.time);
    testCase.verifyEqual(size(target.butt, 1), N);
    testCase.verifyEqual(size(target.clubhead, 1), N);
    testCase.verifyEqual(size(target.club_quat, 1), N);
    testCase.verifyEqual(size(target.club_quat, 2), 4);
    qn = sqrt(sum(target.club_quat .^ 2, 2));
    testCase.verifyLessThan(max(abs(qn - 1)), 1e-6);
    testCase.verifyTrue(all(isfinite(target.butt(:))));
    testCase.verifyTrue(all(isfinite(target.clubhead(:))));
    testCase.verifyEqual(target.source.format, "c3d");
end


function mph = peak_clubhead_speed_mph(target, mps_to_mph)
    t = target.time;
    p = target.clubhead;
    if numel(t) < 5
        mph = 0;
        return;
    end
    dt = t(3:end) - t(1:end-2);
    v = (p(3:end, :) - p(1:end-2, :)) ./ dt;
    speeds = vecnorm(v, 2, 2);
    mph = max(speeds) * mps_to_mph;
end
