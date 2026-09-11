function native_geometry_probe(repo,run_dir)
%NATIVE_GEOMETRY_PROBE Measure active length bindings in independent native snapshots.
assert(strcmp(version('-release'),'2025b'));
root=fullfile(repo,'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab');
addpath(fullfile(root,'src/model')); addpath(genpath(fullfile(root,'src/functions'))); addpath(fullfile(root,'motion_matching/shared'));
load_system('GolfSwing3D_Kinetic');
ws=get_param('GolfSwing3D_Kinetic','ModelWorkspace');
names={'UpperArmLength','LowerArmLength','LeftUpperArmLength','RightUpperArmLength'};
original_objects=cellfun(@(name)getVariable(ws,name),names,'UniformOutput',false);
original=cellfun(@parameter_value,original_objects);
restore=onCleanup(@()restore_geometry(ws,names,original_objects));
seed=jsondecode(fileread(fullfile(run_dir,'native_pose_seed_waist.json')));
report=struct('matlab',version,'source_revision','9a381f018c2fe96b1b36c23a3c7d8c5aaeb3f74e','parameter_names',{names},'original_values',original,'cases',struct([]));
changes=[0 0 0 0;0 0 1 1;1 1 0 0];
for c=1:size(changes,1)
    for j=1:numel(names); assignin(ws,names{j},original(j)+changes(c,j)); end
    started=tic; [ks,schema]=build_golf_kinematics();
    [found,order]=ismember(schema.coordinate_names,string(seed.coordinate_names)); assert(all(found)); q=seed.q(order);
    independent=~ismember(schema.coordinate_names,["RSInputX","RSInputY","RSInputZ","REInput","RWInputX","RWInputY"]);
    addTargetVariables(ks,schema.q_ids(independent)); addInitialGuessVariables(ks,schema.q_ids(~independent));
    addOutputVariables(ks,schema.q_ids); addOutputVariables(ks,schema.frame_ids);
    [values,status,targets]=solve(ks,q(independent),q(~independent));
    assert(ismember(status,[-1,1]) && all(isfinite(values)),'Geometry pose constraints failed');
    positions=reshape(values(numel(q)+1:end),3,[])';
    pairs=["LS","LE";"LE","LF";"LF","LW";"RS","RE";"RE","RF";"RF","RW"];
    lengths=zeros(size(pairs,1),1);
    for k=1:size(pairs,1)
        [found,ids]=ismember(pairs(k,:),string({schema.frames.name})); assert(all(found));
        lengths(k)=norm(positions(ids(1),:)-positions(ids(2),:));
    end
    row=struct('parameter_values',original+changes(c,:),'elapsed_s',toc(started),'solver_flag',status, ...
        'targets_satisfied',all(targets),'length_pairs',pairs,'lengths_m',lengths,'q',values(1:numel(q)));
    if c==1; report.cases=row; else; report.cases(c)=row; end
    fid=fopen(fullfile(run_dir,'native_geometry_probe.json'),'w'); assert(fid~=-1); fprintf(fid,'%s',jsonencode(report,PrettyPrint=true)); fclose(fid);
end
assert(max(abs(report.cases(2).lengths_m-report.cases(1).lengths_m))<1e-8,'Aliases unexpectedly drive native lengths');
expected=[.0254;.0127;.0127;.0254;.0127;.0127];
assert(max(abs(report.cases(3).lengths_m-report.cases(1).lengths_m-expected))<1e-8,'Active length response differs from inch bindings');
clear restore; close_system('GolfSwing3D_Kinetic',0);
end
function restore_geometry(ws,names,values)
for j=1:numel(names); assignin(ws,names{j},values{j}); end
end

function value=parameter_value(parameter)
if isa(parameter,'Simulink.Parameter'); value=parameter.Value; else; value=parameter; end
assert(isnumeric(value) && isscalar(value) && isfinite(value));
end


