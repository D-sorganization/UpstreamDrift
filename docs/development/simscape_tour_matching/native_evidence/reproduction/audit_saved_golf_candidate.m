function audit_saved_golf_candidate(repo,run_dir,expected_channels)
%AUDIT_SAVED_GOLF_CANDIDATE Verify the available actuator logs in a saved fit.
assert(strcmp(version('-release'),'2025b'),'R2025b is required');
if nargin<3;expected_channels=27;end
assert(isscalar(expected_channels) && ismember(expected_channels,[22 27]),'Expected 27 current or explicitly 22 legacy channels');
source=fullfile(repo,'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab');
addpath(fullfile(source,'motion_matching/shared'));addpath(fullfile(source,'src/model'));addpath(genpath(fullfile(source,'src/functions')));
was_loaded=bdIsLoaded('GolfSwing3D_Kinetic');load_system('GolfSwing3D_Kinetic');
if ~was_loaded;cleanup=onCleanup(@()close_system('GolfSwing3D_Kinetic',0));end
workspace=get_param('GolfSwing3D_Kinetic','ModelWorkspace');tilt=getVariable(workspace,'PlaneTilt');
if isa(tilt,'Simulink.Parameter');tilt=tilt.Value;end
rotation=intrinsic_xyz_to_rotm([deg2rad(tilt) 0 0])';
saved=load(fullfile(run_dir,'final_native_replay.mat'),'fit_theta','fit_seed','fit_last_replay');
report=audit_golf_actuator_torques(saved.fit_last_replay.raw_output.CombinedSignalBus, ...
    saved.fit_theta,string(saved.fit_seed.coordinate_names)',rotation);
report.expected_channels=expected_channels;report.matlab=version;report.source_candidate=run_dir;report.plane_tilt_deg=tilt;

fid=fopen(fullfile(run_dir,'actuator_audit.json'),'w');assert(fid~=-1);
fprintf(fid,'%s',jsonencode(report,PrettyPrint=true));fclose(fid);
assert(numel(report.entries)==expected_channels && numel(report.unlogged_coordinates)==27-expected_channels ...
    && report.max_force_error_N<1e-8 && report.max_torque_error_Nm<1e-8,'Available native actuator logs differ from requested polynomials');
end
