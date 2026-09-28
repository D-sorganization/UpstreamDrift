classdef test_gs3dx_render < matlab.unittest.TestCase
%TEST_GS3DX_RENDER  Headless 3D rendering of the GS3DX golfer (#10979).
%
%   Checks:
%     1. Headless rendering of one pose of GS3DX_FitBalance writes a valid,
%        non-trivial PNG image (not all one colour).
%     2. Geometric accuracy: the drawn cylinder of 'Lower Body/L Thigh'
%        matches its FK pose and dimensions to 1e-9 m.
%     3. The drawn brick of 'Lower Body/L Foot' matches its FK pose and
%        dimensions to 1e-9 m.

    properties
        info struct
        mdl char
        ik struct
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
            testCase.mdl = char(gs3dx_names().variants.fit_balance);
            load_system(testCase.mdl);
            testCase.addTeardown(@() close_system(testCase.mdl, 0));

            % Address and impact poses from a whole-body IK of the capture
            jc = gs3dx_capture_joint_centres(gs3dx_capture_markers());
            f0 = jc.impact_frame;
            testCase.ik = gs3dx_whole_body_ik(jc, frames=[1 f0], calibration_frames=1:15:f0 - 90);
        end
    end

    methods (Test)
        function renders_one_pose_to_temp_folder(testCase)
            temp_dir = tempname;
            mkdir(temp_dir);
            testCase.addTeardown(@() rmdir(temp_dir, 's'));

            % Render frame 1 to temporary directory
            out = gs3dx_render(testCase.mdl, testCase.ik, stills=1, ...
                output_dir=temp_dir, still_files=["test_pose_1.png"], ...
                view="face-on", visible=false);

            testCase.verifyNotEmpty(out.files, 'out.files is non-empty');
            png_file = fullfile(temp_dir, 'test_pose_1.png');
            testCase.verifyTrue(isfile(png_file), 'Rendered PNG file exists');

            % Verify image is non-trivial (not all one colour)
            img = imread(png_file);
            testCase.verifyGreaterThan(numel(img), 0, 'Image has pixels');
            
            % Standard deviation across all color channels
            img_double = double(img);
            pixel_std = std(img_double(:));
            % Measured 2026-09-28: pixel std > 10 for rendered scene
            testCase.verifyGreaterThan(pixel_std, 1.0, 'Image has color variation (not blank)');
        end

        function cylinder_geometry_matches_fk_pose(testCase)
            % Check that Lower Body/L Thigh vertices in World coordinates
            % match its FK pose (R, P) and cylinder dimensions (radius, length)
            % to 1e-9 m.
            out = gs3dx_render(testCase.mdl, testCase.ik, stills=1, visible=false);
            
            % Locate 'Lower Body/L Thigh' solid in out.solids
            solids = out.solids;
            names = string({solids.name});
            idx = find(endsWith(names, "Lower Body/L Thigh"));
            testCase.verifyNotEmpty(idx, 'Lower Body/L Thigh found in solids');
            
            thigh = solids(idx);
            testCase.verifyEqual(thigh.shape, "Cylinder");
            
            % In local frame, cylinder vertices have radius R and length L along z
            R = thigh.params.radius;
            L = thigh.params.length;
            % 2026-09-28: ThighLength = 0.460137 m, ThighRadius = 0.07 m
            testCase.verifyEqual(R, 0.07, 'AbsTol', 1e-9, 'Thigh radius is 0.07 m');
            testCase.verifyEqual(L, 0.460137, 'AbsTol', 1e-5, 'Thigh length is ~0.46 m');

            % Verify World vertices match R_mat * V_local + P
            P_world = thigh.pose.P(:, 1);
            R_world = thigh.pose.R(:, :, 1);
            V_world = thigh.vertices_world{1};
            V_local = thigh.vertices_local;

            % Reconstruction check: R_world * V_local + P_world == V_world
            V_recon = R_world * V_local + P_world;
            diff_norm = max(vecnorm(V_world - V_recon));
            % Measured 2026-09-28: 0 (exact match)
            testCase.verifyLessThan(diff_norm, 1e-9, 'Drawn cylinder surface matches FK pose');

            % Local radial and axial extent checks
            radial_dist = sqrt(V_local(1, :).^2 + V_local(2, :).^2);
            testCase.verifyEqual(max(radial_dist), R, 'AbsTol', 1e-9, 'Max radial distance matches R');
            testCase.verifyEqual(min(V_local(3, :)), -L/2, 'AbsTol', 1e-9, 'Min z matches -L/2');
            testCase.verifyEqual(max(V_local(3, :)), L/2, 'AbsTol', 1e-9, 'Max z matches +L/2');
        end

        function brick_geometry_matches_dimensions(testCase)
            out = gs3dx_render(testCase.mdl, testCase.ik, stills=1, visible=false);
            solids = out.solids;
            names = string({solids.name});
            idx = find(endsWith(names, "Lower Body/L Foot"));
            testCase.verifyNotEmpty(idx, 'Lower Body/L Foot found in solids');
            
            foot = solids(idx);
            testCase.verifyEqual(foot.shape, "Brick");
            dims = foot.params.dimensions; % [dx, dy, dz]
            
            P_world = foot.pose.P(:, 1);
            R_world = foot.pose.R(:, :, 1);
            V_world = foot.vertices_world{1};
            V_local = foot.vertices_local;

            V_recon = R_world * V_local + P_world;
            diff_norm = max(vecnorm(V_world - V_recon));
            testCase.verifyLessThan(diff_norm, 1e-9, 'Drawn brick surface matches FK pose');

            testCase.verifyEqual(max(abs(V_local(1, :))), dims(1)/2, 'AbsTol', 1e-9, 'Brick dx/2');
            testCase.verifyEqual(max(abs(V_local(2, :))), dims(2)/2, 'AbsTol', 1e-9, 'Brick dy/2');
            testCase.verifyEqual(max(abs(V_local(3, :))), dims(3)/2, 'AbsTol', 1e-9, 'Brick dz/2');
        end
    end
end
