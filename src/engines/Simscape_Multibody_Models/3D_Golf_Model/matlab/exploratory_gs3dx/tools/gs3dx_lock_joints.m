function [locked, edited] = gs3dx_lock_joints(mdl)
%GS3DX_LOCK_JOINTS  Lock every joint but the pelvis root, in memory (#11709).
%
%   [LOCKED, EDITED] = GS3DX_LOCK_JOINTS(MDL) sets JointMode 'Locked' on
%   every Simscape Multibody joint block of the loaded model MDL except the
%   6-DOF pelvis root and the Weld Joints, and returns their paths.  The
%   body then moves as one rigid body on its sole contacts: the standing
%   test for the ground reactions, where a PD posture hold neither settles
%   nor keeps the momentum check (GROUND_CONTACT.md).  Actuation inputs of a
%   locked joint are ignored.
%
%   Most joints sit inside Subsystem References, whose instances share one
%   subsystem file and cannot be edited one by one.  Those joints are locked
%   inside the referenced subsystem file, and an update diagram then shows
%   the change in every instance.  EDITED lists the subsystem files changed that way.  Nothing is saved:
%   close MDL and then every EDITED file with CLOSE_SYSTEM(.., 0) to discard
%   the change.
%
%   Preconditions: MDL is loaded; no locked joint shares a subsystem file
%   with the root or a weld.
%   Postconditions: every path in LOCKED has JointMode 'Locked'; the root
%   and the welds keep their mode; no file on disk is changed.

    arguments
        mdl (1,:) char
    end
    if ~bdIsLoaded(mdl)
        error('gs3dx:lock_joints', 'Precondition: model %s is not loaded', mdl);
    end
    b = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'Type', 'Block');
    ref = get_param(b, 'ReferenceBlock');
    joint = contains(ref, 'sm_lib/Joints/');
    keep = contains(ref, '6-DOF Joint') | contains(ref, 'Weld Joint');
    locked = b(joint & ~keep);
    kept_files = setdiff(cellfun(@(x) strtok(local_file_path(x), '/'), ...
        b(joint & keep), 'UniformOutput', false), {mdl});
    edited = {};
    for k = 1:numel(locked)
        target = local_file_path(locked{k});
        file = strtok(target, '/');
        if any(strcmp(file, kept_files))
            error('gs3dx:lock_joints', ...
                'Precondition: %s shares a subsystem file with the root or a weld', locked{k});
        end
        if ~strcmp(file, mdl)
            edited = union(edited, {file});
        end
        set_param(target, 'JointMode', 'Locked');
    end
    if ~isempty(edited)
        % instances show an unsaved subsystem-file edit after an update
        set_param(mdl, 'SimulationCommand', 'update');
    end
end

function target = local_file_path(blk)
% The editable path of BLK: inside the innermost Subsystem Reference's
% subsystem file when BLK sits in one, otherwise BLK itself.
    target = blk;
    p = get_param(blk, 'Parent');
    while contains(p, '/')
        if strcmp(get_param(p, 'Type'), 'block') && ...
                ~isempty(get_param(p, 'ReferencedSubsystem'))
            ss = get_param(p, 'ReferencedSubsystem');
            load_system(ss);
            target = [ss blk(numel(p) + 1:end)];
            return;
        end
        p = get_param(p, 'Parent');
    end
end
