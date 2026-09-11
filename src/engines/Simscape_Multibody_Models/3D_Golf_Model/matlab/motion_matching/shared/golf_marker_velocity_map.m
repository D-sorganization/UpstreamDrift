function result = golf_marker_velocity_map(ks,schema,q,bodies,offsets)
%GOLF_MARKER_VELOCITY_MAP Native tangent map for fixed body marker offsets.
% Reset caller-owned solver roles. Output rows are marker-major XYZ in m/s.
% Inputs are independent translation rates (m/s) and angular rates (rad/s).
% Angular frame velocities are resolved in follower axes; linear in world axes.
    arguments
        ks
        schema (1,1) struct
        q (:,1) double {mustBeReal,mustBeFinite}
        bodies (:,1) double {mustBeInteger,mustBePositive}
        offsets (:,3) double {mustBeReal,mustBeFinite}
    end
    n=numel(schema.frames);count=numel(schema.q_ids);
    assert(numel(q)==count && numel(bodies)==size(offsets,1) && all(bodies<=n), ...
        'golf_marker_velocity_map:dimensions','Coordinate and attachment dimensions differ.');
    independent=~ismember(schema.coordinate_names,string(schema.dependent_coordinates));
    variables=jointVelocityVariables(ks);parts=split(variables.ID,'.');
    keys=string(variables.BlockPath)+"|"+parts(:,2);ids=strings(count,1);
    for j=1:count
        coordinate=schema.coordinates(j);
        found=keys==string(coordinate.block_path)+"|"+string(coordinate.primitive);
        assert(nnz(found)==1,'golf_marker_velocity_map:coordinate','Velocity coordinate missing or ambiguous.');
        ids(j)=variables.ID(found);
        unit='rad/s';if startsWith(schema.coordinate_names(j),'Translation');unit='m/s';end
        setVariableUnit(ks,ids(j),unit);
    end
    linear=strings(3*n,1);angular=linear;
    for f=1:n
        group="Velocity"+string(schema.frames(f).name);
        linear(3*f-2:3*f)=group+".LinearVelocity."+["x";"y";"z"];
        angular(3*f-2:3*f)=group+".AngularVelocity."+["x";"y";"z"];
        existing=frameVariables(ks);
        if ~ismember(linear(3*f),existing.ID)
            addFrameVariables(ks,group,'LinearVelocity',schema.world_port,schema.frames(f).port,'LinearVelocityUnit','m/s');
        end
        if ~ismember(angular(3*f),existing.ID)
            addFrameVariables(ks,group,'AngularVelocity',schema.world_port,schema.frames(f).port,'AngularVelocityUnit','rad/s');
        end
    end
    clearTargetVariables(ks);clearInitialGuessVariables(ks);clearOutputVariables(ks);
    addTargetVariables(ks,[schema.q_ids;ids(independent)]);
    addOutputVariables(ks,[schema.rotation_ids;linear;angular;ids]);
    dimension=nnz(independent);jacobian=zeros(3*numel(bodies),dimension);joint_map=zeros(count,dimension);
    for j=0:dimension
        velocity=zeros(dimension,1);if j>0;velocity(j)=1;end
        [values,status,targets]=solve(ks,[q;velocity]);
        assert(status==1 && all(targets) && all(isfinite(values)), ...
            'golf_marker_velocity_map:constraints','Native position/velocity targets and constraints must hold.');
        rotation=intrinsic_xyz_to_rotm(reshape(values(1:3*n),3,[])');
        linear_values=reshape(values(3*n+1:6*n),3,[])';
        angular_values=reshape(values(6*n+1:9*n),3,[])';
        marker_velocity=zeros(numel(bodies),3);
        for m=1:numel(bodies)
            b=bodies(m);
            marker_velocity(m,:)=linear_values(b,:)+cross(angular_values(b,:),offsets(m,:))*rotation(:,:,b)';
        end
        if j==0
            assert(norm(marker_velocity,'fro')<1e-10 && norm(values(9*n+1:end))<1e-10, ...
                'golf_marker_velocity_map:zero','Zero independent rates must yield zero velocities.');
        else
            jacobian(:,j)=reshape(marker_velocity',[],1);joint_map(:,j)=values(9*n+1:end);
        end
    end
    result=struct('marker_jacobian',jacobian,'joint_velocity_map',joint_map, ...
        'independent_names',schema.coordinate_names(independent),'coordinate_names',schema.coordinate_names);
end
