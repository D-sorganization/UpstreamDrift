root='C:/Users/diete/SimscapeTour9921'; cd(root);
repo='C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour-runtime';
engine=fullfile(repo,'src','engines','Simscape_Multibody_Models','3D_Golf_Model','matlab');
addpath(fullfile(engine,'src','model')); addpath(genpath(fullfile(engine,'src','functions'))); addpath(fullfile(engine,'motion_matching','shared'));
assert(strcmp(version('-release'),'2025b'));
r=struct('matlab',version,'source_revision','0d8b3400421c8c71807356eba4d9b7cdb273c33f','source_patch','published 16-frame schema','qualification','surface-proxy-pose-seed-only');
try
 prior=jsondecode(fileread(fullfile(root,'native_pose_seed.json'))); sample=struct('joint_names',string(prior.coordinate_names)','q',prior.q');
 target=jsondecode(fileread(fullfile(root,'native_pose_seed_waist_target.json'))); r.target=target;
 load_system('GolfSwing3D_Kinetic'); [ks,schema]=build_golf_kinematics(); ks.MaxIterations=300;
 [found,columns]=ismember(schema.coordinate_names,sample.joint_names); assert(all(found)); initial=sample.q(1,columns)';
 independent=~ismember(schema.coordinate_names,["RSInputX","RSInputY","RSInputZ","REInput","RWInputX","RWInputY"]);
 addTargetVariables(ks,schema.q_ids(independent)); addInitialGuessVariables(ks,schema.q_ids(~independent));
 addOutputVariables(ks,schema.q_ids); addOutputVariables(ks,schema.frame_ids);
 [found,indices]=ismember(string({target.observations.frame}),string({schema.frames.name})); assert(all(found));
 points=vertcat(target.observations.target_m); if size(points,2)~=3; points=reshape(points,3,[])'; end
 x0=initial(independent); guess=initial(~independent); count=numel(initial);
 objective=@(x) pose_residual(x,ks,guess,independent,x0,count,indices,points);
 r.initial_residual=objective(x0); r.initial_proxy_rms_m=sqrt(mean(r.initial_residual(1:numel(points)).^2));
 lower=x0-2*pi; upper=x0+2*pi; lower(1:3)=-5; upper(1:3)=5;
 options=optimoptions('lsqnonlin','Display','iter','MaxIterations',100,'MaxFunctionEvaluations',2400,'FunctionTolerance',1e-8,'StepTolerance',1e-7,'FiniteDifferenceStepSize',1e-5);
 timer=tic; [x,resnorm,residual,exitflag,output]=lsqnonlin(objective,x0,lower,upper,options); r.fit_s=toc(timer);
 [result,flag,targetFlags]=solve(ks,x,guess); r.q=result(1:count); r.coordinate_names=schema.coordinate_names;
 positions=reshape(result(count+1:end),3,[])'; r.frame_positions_m=positions; r.frames=schema.frames;
 r.proxy_errors_m=positions(indices,:)-points; r.final_proxy_rms_m=sqrt(mean(r.proxy_errors_m.^2,'all'));
 r.exitflag=exitflag; r.optimizer=output; r.solver_flag=flag; r.target_flags=targetFlags; r.resnorm=resnorm; r.status='seed-computed';
 save(fullfile(root,'native_pose_seed_waist.mat'),'r','schema','x','initial');
catch err
 r.status='failed'; r.error=struct('identifier',err.identifier,'message',err.message,'report',getReport(err,'extended','hyperlinks','off'));
end
fid=fopen(fullfile(root,'native_pose_seed_waist.json'),'w'); fwrite(fid,jsonencode(r,PrettyPrint=true)); fclose(fid);
function residual=pose_residual(x,ks,guess,independent,x0,count,indices,points)
 [result,flag]=solve(ks,x,guess);
 assert(all(isfinite(result)),'pose_seed:invalidKinematics','Kinematics solver returned nonfinite values.');
 positions=reshape(result(count+1:end),3,[])'; delta=positions(indices,:)-points; q=result(1:count);
 residual=[delta(:);0.005*(x-x0);10*(q(independent)-x)];
end
