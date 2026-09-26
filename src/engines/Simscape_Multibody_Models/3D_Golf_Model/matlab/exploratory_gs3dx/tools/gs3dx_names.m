function names = gs3dx_names()
%GS3DX_NAMES  Single source of truth for original and exploratory model names.
%
%   NAMES = GS3DX_NAMES() returns a struct:
%     .prefix            "GS3DX_"
%     .original_model    "GolfSwing3D_Kinetic"
%     .original_subsys   map of role -> original referenced-subsystem name
%     .clone_subsys      map of role -> GS3DX referenced-subsystem name
%     .slim_subsys       map of role -> direct-torque-drive subsystem (#10954)
%     .spherical_subsys  Spherical-joint stand-in for GS3DX_KDS_Gimbal (#10955)
%     .variants          GS3DX top-level model names by stage
%
%   Every tool reads names from here (DRY) so a rename happens in one place.

    names = struct();
    names.prefix         = "GS3DX_";
    names.original_model = "GolfSwing3D_Kinetic";

    roles = ["Gimbal", "Revolute", "Universal"];
    names.roles = roles;
    names.original_subsys = containers.Map( ...
        cellstr(roles), ...
        {'Kinetically_Driven_Gimbal_Joint', ...
         'Kinetically_Driven_Revolute_Joint', ...
         'Kinetically_Driven_Universal_Joint'});
    names.clone_subsys = containers.Map( ...
        cellstr(roles), ...
        {'GS3DX_KD_Gimbal', 'GS3DX_KD_Revolute', 'GS3DX_KD_Universal'});
    names.slim_subsys = containers.Map( ...
        cellstr(roles), ...
        {'GS3DX_KDS_Gimbal', 'GS3DX_KDS_Revolute', 'GS3DX_KDS_Universal'});
    names.spherical_subsys = 'GS3DX_KDS_Spherical';

    names.variants = struct( ...
        'baseline', "GS3DX_Baseline", ...   % verbatim renamed clone
        'slim',     "GS3DX_Slim", ...       % direct joint torque drive (#10954)
        'quat',     "GS3DX_Quat", ...       % quaternion shoulders + hip (#10955/#10956)
        'fullbody', "GS3DX_FullBody", ...   % lower body added (#10957/#10958)
        'contact',  "GS3DX_FullBodyContact");   % foot-ground contact, leg-supported pelvis (#10986)

    names.simscape_prefixes   = ["sm_lib", "fl_lib", "nesl_utility", "ee_lib"];
    names.converter_refs      = ["nesl_utility/PS-Simulink Converter", ...
                                 "nesl_utility/Simulink-PS Converter"];
    names.license_block_limit = 1000;
end
