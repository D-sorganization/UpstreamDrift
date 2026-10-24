classdef test_gs3dx_rigid_stance < matlab.unittest.TestCase
    % The rigid-body stance proxy (#11709, director-approved option (b)):
    % one rigid body at the GS3DX golfer's total mass and centre of mass,
    % standing on the same six sole contacts and Spatial Contact Force law
    % and parameters as GS3DX_FullBodyContact.  A short STOP_TIME keeps
    % this unit test fast; the committed fixture uses the full 2 s settle.
    properties
        info
        r
        model_file
        model_mtime_before
    end
    methods (TestClassSetup)
        function setup(t)
            t.info = gs3dx_setup();
            mdl = char(gs3dx_names().variants.contact);
            t.model_file = which(mdl);
            t.model_mtime_before = dir(t.model_file).datenum;
            t.r = gs3dx_rigid_stance(t.info, stop_time=0.01, output_dt=0.005);
        end
    end
    methods (Test)
        function stays_under_the_license_block_limit(t)
            t.verifyLessThanOrEqual(t.r.budget.compiled_total, gs3dx_names().license_block_limit);
        end
        function contact_parameters_match_the_full_model(t)
            mdl = char(gs3dx_names().variants.contact);
            full = gs3dx_contact_geometry(mdl);
            for n = ["FootContactStiffness", "FootContactDamping", "FootContactTransitionWidth", ...
                     "FootStaticFriction", "FootDynamicFriction", "FootFrictionCriticalVelocity", ...
                     "FootContactRadius"]
                t.verifyEqual(t.r.geo.(n), full.(n), n);
            end
        end
        function mass_and_com_match_a_fresh_probe(t)
            c = gs3dx_contact_check(t.info, rest=true, stop_time=0.004);
            t.verifyEqual(t.r.mass_kg, c.mass);
            t.verifyEqual(t.r.com_m, c.com(:, 1));
        end
        function six_contacts_are_logged_over_time(t)
            n = numel(t.r.t);
            t.verifySize(t.r.forces_n, [3 6 n]);
            t.verifySize(t.r.com_p, [3 n]);
            t.verifySize(t.r.sphere_points_m, [3 6]);
        end
        function the_source_model_file_on_disk_is_unchanged(t)
            t.verifyEqual(dir(t.model_file).datenum, t.model_mtime_before);
        end
        function the_rig_is_closed_after_the_call(t)
            t.verifyFalse(bdIsLoaded('gs3dx_rigid_stance_rig'));
        end
    end
end
