function [keys, ids] = gs3dx_joint_keys(mdl, jp)
%GS3DX_JOINT_KEYS  Model-independent names of a model's joint position variables (#10979).
%
%   [KEYS, IDS] = GS3DX_JOINT_KEYS(MDL) returns, for each joint position
%   variable of MDL's KinematicsSolver, its ID (such as "j15.Rz.q") and a
%   key made of the joint's block path below the model and the variable's
%   primitive ("Right Elbow Joint/Revolute Joint/Kinetically Driven
%   Revolute|Rz.q").  Simscape numbers joints in block-path order, so a
%   joint added to a model renumbers the joints after it; the keys stay the
%   same across GS3DX variants.  MDL is loaded if it is not, and closed
%   again.  GS3DX_JOINT_KEYS(MDL, JP) uses the table JP
%   (KinematicsSolver.jointPositionVariables) instead.

    arguments
        mdl (1,:) char
        jp table = table()
    end
    if isempty(jp)
        if ~bdIsLoaded(mdl)
            load_system(mdl);
            cleanup = onCleanup(@() close_system(mdl, 0));
        end
        jp = simscape.multibody.KinematicsSolver(mdl).jointPositionVariables;
    end
    ids = string(jp.ID);
    path = string(jp.BlockPath);
    assert(all(startsWith(path, [mdl '/'])), 'gs3dx:joint_keys', 'Joint block paths are not below %s', mdl);
    keys = extractAfter(path, strlength(mdl) + 1) + "|" + extractAfter(ids, ".");
    assert(numel(unique(keys)) == numel(keys), 'gs3dx:joint_keys', '%s has repeated joint keys', mdl);
end
