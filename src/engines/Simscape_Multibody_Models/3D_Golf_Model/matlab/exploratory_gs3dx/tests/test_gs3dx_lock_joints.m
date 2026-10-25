classdef test_gs3dx_lock_joints < matlab.unittest.TestCase
    % The locked-joint stance (#11709): every joint except the 6-DOF pelvis
    % root and the welds goes to JointMode Locked, in memory only.
    properties
        mdl
        edited = {}
    end
    methods (TestClassSetup)
        function setup(t)
            gs3dx_setup();
            t.mdl = char(gs3dx_names().variants.contact);
        end
    end
    methods (TestMethodSetup)
        function load_model(t)
            load_system(t.mdl);
            t.addTeardown(@() t.discard());
        end
    end
    methods
        function [locked, edited] = lock(t)
            [locked, edited] = gs3dx_lock_joints(t.mdl);
            t.edited = edited;
        end
        function discard(t)
            for bd = [{t.mdl}, t.edited]
                if bdIsLoaded(bd{1})
                    close_system(bd{1}, 0);
                end
            end
        end
    end
    methods (Test)
        function every_non_root_joint_is_locked(t)
            locked = t.lock();
            t.verifyNumElements(locked, 18);   % 2 trunk + 10 arm + 6 leg
            for k = 1:numel(locked)
                t.verifyEqual(get_param(locked{k}, 'JointMode'), 'Locked', locked{k});
            end
        end
        function the_root_and_the_welds_stay_normal(t)
            locked = t.lock();
            for b = reshape(local_joints(t.mdl), 1, [])
                ref = get_param(b{1}, 'ReferenceBlock');
                if contains(ref, '6-DOF Joint') || contains(ref, 'Weld Joint')
                    t.verifyFalse(any(strcmp(locked, b{1})), b{1});
                    t.verifyEqual(get_param(b{1}, 'JointMode'), 'Normal', b{1});
                end
            end
        end
        function no_file_on_disk_changes(t)
            [~, edited] = t.lock();
            files = cellfun(@which, [{t.mdl}, edited], 'UniformOutput', false);
            before = cellfun(@(f) dir(f).datenum, files);
            t.discard();
            t.edited = {};
            t.verifyEqual(cellfun(@(f) dir(f).datenum, files), before);
            load_system(t.mdl);
            for b = reshape(local_joints(t.mdl), 1, [])
                t.verifyEqual(get_param(b{1}, 'JointMode'), 'Normal', b{1});
            end
        end
        function an_unloaded_model_is_refused(t)
            t.discard();
            t.verifyError(@() gs3dx_lock_joints(t.mdl), 'gs3dx:lock_joints');
        end
    end
end

function b = local_joints(mdl)
    b = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'Type', 'Block');
    b = b(contains(get_param(b, 'ReferenceBlock'), 'sm_lib/Joints/'));
end
