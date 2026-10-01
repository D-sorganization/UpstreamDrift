function fit = gs3dx_fit_lengths(jc)
%GS3DX_FIT_LENGTHS  Model segment lengths matched to the capture (#10979).
%
%   FIT = GS3DX_FIT_LENGTHS(JC) maps the joint-centre segment lengths of
%   GS3DX_CAPTURE_JOINT_CENTRES onto the GS3DX length variables.  The
%   golfer's anthropometry is unknown, so the lengths come from the markers:
%   the median over frames of each joint-centre distance.
%
%     .vars  model-workspace values for GS3DX_BUILD_FIT:
%              FitHubtoSLength     in   half the shoulder (GH-GH) width
%              FitUpperArmLength   in   lead upper arm (the trail acromion is
%                                       rebuilt in 80% of frames, so its arm
%                                       length is not used)
%              FitLowerArmLength   in   mean of both forearms (elbow-wrist)
%              FitLowerTorsoLength in   half the pelvis-to-shoulder-line height
%              FitUpperTorsoLength in   the other half (the original 12/12 split)
%              ThighLength         m    mean of both thighs
%              ShankLength         m    mean of both shanks
%     .source  per variable, the JC.lengths fields it came from
%
%   The upper-body values are in inches because the solids' CylinderLength
%   units are inches; the leg values are in metres like GS3DX_LEG_TABLE.
%   Body mass is not identifiable from kinematics and is not set here.

    arguments
        jc (1,1) struct
    end
    need = ["shoulders", "upper_arm_L", "forearm_L", "forearm_R", ...
        "pelvis_to_shoulders", "thigh_L", "thigh_R", "shank_L", "shank_R"];
    missing = need(~isfield(jc.lengths, need));
    assert(isempty(missing), 'gs3dx:fit', 'JC.lengths lacks: %s', strjoin(missing, ', '));
    L = @(f) jc.lengths.(f)(1);
    in = 0.0254;

    v = struct();
    v.FitHubtoSLength = L("shoulders") / 2 / in;
    v.FitUpperArmLength = L("upper_arm_L") / in;
    v.FitLowerArmLength = (L("forearm_L") + L("forearm_R")) / 2 / in;
    v.FitLowerTorsoLength = L("pelvis_to_shoulders") / 2 / in;
    v.FitUpperTorsoLength = v.FitLowerTorsoLength;
    v.ThighLength = (L("thigh_L") + L("thigh_R")) / 2;
    v.ShankLength = (L("shank_L") + L("shank_R")) / 2;

    fit.vars = v;
    fit.source = struct( ...
        'FitHubtoSLength', "shoulders/2", 'FitUpperArmLength', "upper_arm_L", ...
        'FitLowerArmLength', "mean(forearm_L, forearm_R)", ...
        'FitLowerTorsoLength', "pelvis_to_shoulders/2", 'FitUpperTorsoLength', "pelvis_to_shoulders/2", ...
        'ThighLength', "mean(thigh_L, thigh_R)", 'ShankLength', "mean(shank_L, shank_R)");
    vals = struct2array(v);
    assert(all(isfinite(vals) & vals > 0), 'gs3dx:fit', 'Postcondition: a fitted length is not positive');
end
