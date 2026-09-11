function result = fit_golf_pose_seed(ks,schema,initial,frame_names,points,offsets)
%FIT_GOLF_POSE_SEED Fit native frame proxies with geometry held fixed.
% Resets the supplied KinematicsSolver variable roles. This is a pose seed,
% not inverse dynamics or anatomical marker calibration. Initial coordinates
% use schema ordering and SI units. Final targets and constraints must hold.
% Optional offsets are fixed body-local XYZ metres, one per observation.
% Repeated frame names permit multiple markers on one body. Omitting offsets
% retains frame-origin fitting. This local pose solution is not a global
% kinematic lower bound and does not enforce dynamics or joint limits.
    arguments
        ks
        schema (1,1) struct
        initial (:,1) double {mustBeReal,mustBeFinite}
        frame_names (:,1) string
        points (:,3) double {mustBeReal,mustBeFinite}
        offsets (:,3) double {mustBeReal,mustBeFinite} = zeros(size(points))
    end
    count=numel(schema.q_ids);
    assert(isequal(size(offsets),size(points)),'fit_golf_pose_seed:attachments', ...
        'Provide one body-local offset per observation.');
    assert(numel(initial)==count,'fit_golf_pose_seed:coordinates','Initial coordinate count differs.');
    [found,indices]=ismember(frame_names,string({schema.frames.name}));
    assert(all(found) && numel(indices)==size(points,1) && ~isempty(indices), ...
        'fit_golf_pose_seed:frames','Provide one finite position per known frame.');
    dependent=string(schema.dependent_coordinates);
    independent=~ismember(schema.coordinate_names,dependent);
    clearTargetVariables(ks); clearInitialGuessVariables(ks); clearOutputVariables(ks);
    addTargetVariables(ks,schema.q_ids(independent));
    addInitialGuessVariables(ks,schema.q_ids(~independent));
    addOutputVariables(ks,schema.q_ids); addOutputVariables(ks,schema.frame_ids);
    addOutputVariables(ks,schema.rotation_ids);
    ks.MaxIterations=300;
    x0=initial(independent); guess=initial(~independent);
    % Weak pose prior selects among underdetermined surface-proxy solutions.
    prior_weight=0.005; target_weight=10;
    objective=@residual;
    lower=x0-2*pi; upper=x0+2*pi;
    translations=startsWith(schema.coordinate_names(independent),'Translation');
    lower(translations)=-5; upper(translations)=5;
    options=optimoptions('lsqnonlin','Display','off','MaxIterations',100, ...
        'MaxFunctionEvaluations',2400,'FunctionTolerance',1e-8, ...
        'StepTolerance',1e-7,'FiniteDifferenceStepSize',1e-5);
    [x,resnorm,~,exitflag,optimizer]=lsqnonlin(objective,x0,lower,upper,options);
    [values,flag,targets]=solve(ks,x,guess);
    assert(flag==1 && all(targets) && all(isfinite(values)), ...
        'fit_golf_pose_seed:constraints','Final native pose must satisfy targets and constraints.');
    [markers,positions]=project(values);
    errors=markers-points;
    qualification='surface-proxy-pose-seed-only';
    if nargin>=6; qualification='fixed-attachment-pose-seed-only'; end
    result=struct('q',values(1:count),'coordinate_names',schema.coordinate_names, ...
        'frame_positions_m',positions,'marker_positions_m',markers,'proxy_errors_m',errors, ...
        'proxy_euclidean_rms_m',sqrt(mean(sum(errors.^2,2))), ...
        'solver_flag',flag,'target_flags',targets,'exitflag',exitflag, ...
        'optimizer',optimizer,'resnorm',resnorm,'qualification',qualification);
    function r=residual(x)
        [values,flag]=solve(ks,x,guess);
        assert(ismember(flag,[-1,1]) && all(isfinite(values)), ...
            'fit_golf_pose_seed:invalidKinematics','Native physical constraints failed.');
        markers=project(values);
        delta=markers-points; q=values(1:count);
        r=[delta(:);prior_weight*(x-x0);target_weight*(q(independent)-x)];
    end
    function [markers,positions]=project(values)
        last=count+numel(schema.frame_ids);
        positions=reshape(values(count+1:last),3,[])';
        rotations=intrinsic_xyz_to_rotm(reshape(values(last+1:end),3,[])');
        markers=project_body_markers(positions,rotations,indices(:),offsets);
    end
end
