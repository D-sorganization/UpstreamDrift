function native_geometry_seed_sweep(repo,run_dir)
%NATIVE_GEOMETRY_SEED_SWEEP Bounded exploratory geometry candidates at one pose.
assert(strcmp(version('-release'),'2025b'));
base=fullfile(repo,'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab');
addpath(fullfile(base,'src/model'));addpath(genpath(fullfile(base,'src/functions')));addpath(fullfile(base,'motion_matching/shared'));
load_system('GolfSwing3D_Kinetic');ws=get_param('GolfSwing3D_Kinetic','ModelWorkspace');
names={'UpperArmLength','LowerArmLength'};original=cellfun(@(name)getVariable(ws,name),names,'UniformOutput',false);
cleanup=onCleanup(@()restore(ws,names,original));
seed=jsondecode(fileread(fullfile(run_dir,'native_pose_seed_waist.json')));
target=jsondecode(fileread(fullfile(run_dir,'native_pose_seed_waist_target.json')));
points=vertcat(target.observations.target_m);
if size(points,2)~=3; points=reshape(points,3,[])'; end
assert(size(points,1)==numel(target.observations) && all(isfinite(points),'all'));
frames=string({target.observations.frame})';
candidates=[12 14;13.5 12;12.5 12;14.5 12;13.5 11;13.5 13;12.5 11;14.5 11;14.5 13];
report=struct('matlab',version,'source_revision','9a381f018c2fe96b1b36c23a3c7d8c5aaeb3f74e', ...
    'source_patch','fit_golf_pose_seed.m','qualification','single-pose surface-proxy geometry exploration only', ...
    'parameter_names',{names},'parameter_units','in','candidates',candidates,'results',{{}});
for c=1:size(candidates,1)
    for j=1:2;assignin(ws,names{j},candidates(c,j));end
    started=tic;
    try
        [ks,schema]=build_golf_kinematics();
        [found,order]=ismember(schema.coordinate_names,string(seed.coordinate_names));assert(all(found));
        result=fit_golf_pose_seed(ks,schema,seed.q(order),frames,points);
        result.parameter_values=candidates(c,:);result.elapsed_s=toc(started);result.status='pose-computed';
    catch error
        if ~ismember(string(error.identifier),["fit_golf_pose_seed:constraints","fit_golf_pose_seed:invalidKinematics"]); rethrow(error); end
        result=struct('parameter_values',candidates(c,:),'elapsed_s',toc(started),'status','rejected', ...
            'error_identifier',error.identifier,'error_message',error.message);
    end
    report.results{c}=result;
    fid=fopen(fullfile(run_dir,'native_geometry_seed_sweep.json'),'w');assert(fid~=-1);fprintf(fid,'%s',jsonencode(report,PrettyPrint=true));fclose(fid);
    snapshot=fullfile(run_dir,sprintf('native_geometry_seed_sweep_case_%02d.json',c));assert(~isfile(snapshot),'Use a fresh run directory to preserve prior checkpoints');
    fid=fopen(snapshot,'w');assert(fid~=-1);fprintf(fid,'%s',jsonencode(report,PrettyPrint=true));fclose(fid);
end
clear cleanup;close_system('GolfSwing3D_Kinetic',0);
end
function restore(ws,names,values)
for j=1:numel(names);assignin(ws,names{j},values{j});end
end


