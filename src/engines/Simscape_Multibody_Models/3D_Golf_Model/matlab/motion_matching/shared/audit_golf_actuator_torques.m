function result = audit_golf_actuator_torques(bus, theta, joint_names, world_to_hip_base)
%AUDIT_GOLF_ACTUATOR_TORQUES Compare explicit actuator logs with native A..G.
% Covers 24 angular and three translational actuator-sensing channels when instrumented. Other
% coordinates remain explicitly unlogged; reaction loads are never substituted. Compare
% on each scalar log's native clock, without interpolation or state resampling.
    arguments
        bus (1,1) struct
        theta (:,1) double {mustBeReal,mustBeFinite}
        joint_names (1,:) string {mustBeNonempty}
        world_to_hip_base (:,:) double {mustBeReal,mustBeFinite} = []
    end
    assert(numel(theta)==7*numel(joint_names) && numel(unique(joint_names))==numel(joint_names), ...
        'audit_golf_actuator_torques:coefficientShape','Expected seven coefficients per unique coordinate');
    coefficients=reshape(theta,7,[])';
    force_names=["TranslationInputX","TranslationInputY","TranslationInputZ"];
    if any(ismember(joint_names,force_names))
        assert(isequal(size(world_to_hip_base),[3 3]) && ...
            norm(world_to_hip_base*world_to_hip_base'-eye(3),'fro')<1e-10 && ...
            abs(det(world_to_hip_base)-1)<1e-10,'audit_golf_actuator_torques:forceFrame', ...
            'Root-force commands are world-resolved; provide the proper world-to-hip-base rotation');
        [found,columns]=ismember(force_names,joint_names);
        assert(all(found),'audit_golf_actuator_torques:forceComponents','All three force coefficient rows are required');
        coefficients(columns,:)=world_to_hip_base*coefficients(columns,:);
    end
    entries=struct('coordinate',{},'source',{},'sample_count',{},'start_s',{},'end_s',{}, ...
        'unit',{},'requested_start',{},'requested_end',{},'max_abs_error',{});
    logged=false(size(joint_names));
    for j=1:numel(joint_names)
        name=joint_names(j);optional=false;
        axis=extractAfter(name,'Input');unit="Nm";
        if ~isempty(regexp(name,'^TranslationInput[XYZ]$','once'))
            path=["HipLogs","TranslationForce"+axis+"Input"];unit="N";
        elseif ~isempty(regexp(name,'^HipInput[XYZ]$','once'))
            path=["HipLogs","HipTorque"+axis+"Input"];
        elseif ~isempty(regexp(name,'^([LR]ScapInput[XY]|[LR]SInput[XYZ]|[LR]WInput[XY]|SpineInput[XY])$','once'))
            base=extractBefore(name,'Input');path=[base+"Logs", "ActuatorTorque"+axis];
            if ismember(base,["LW","RW"]);path=[path(1),base+"Joint",path(2)];end
        elseif ~isempty(regexp(name,'^([LR][EF]Input|TorsoInput)$','once'))
            base=extractBefore(name,'Input');path=[base+"Logs","ActuatorTorque"];optional=true;
            if ismember(base,["LE","RE"]);path=[path(1),base+"Joint",path(2)];end
        else
            continue
        end
        signal=bus;found=true;
        for field=path
            if ~isstruct(signal) || ~isfield(signal,field)
                if optional;found=false;break;end
                error('audit_golf_actuator_torques:missingLog','Missing explicit actuator channel for %s',name);
            end
            signal=signal.(field);
        end
        if ~found;continue;end % Historical revolute logs lack the added sensor.
        assert(isa(signal,'timeseries'),'audit_golf_actuator_torques:invalidLog','Expected a native timeseries');
        clock=double(signal.Time(:));values=double(signal.Data(:));
        assert(numel(clock)>=2 && numel(values)==numel(clock) && all(isfinite([clock;values])) ...
            && all(diff(clock)>0),'audit_golf_actuator_torques:invalidLog','Expected a finite scalar log on an increasing clock');
        expected=polyval(coefficients(j,:),clock);
        entries(end+1)=struct('coordinate',name,'source',"CombinedSignalBus."+join(path,'.'), ...
            'sample_count',numel(clock),'start_s',clock(1),'end_s',clock(end), ...
            'unit',unit,'requested_start',expected(1),'requested_end',expected(end), ...
            'max_abs_error',max(abs(values-expected))); %#ok<AGROW>
        logged(j)=true;
    end
    units=string({entries.unit});
    result=struct('schema','golf-actuator-audit/2','entries',entries,'unlogged_coordinates',joint_names(~logged), ...
        'force_command_frame','world','force_measurement_frame','hip_joint_base', ...
        'world_to_hip_base',world_to_hip_base, ...
        'max_force_error_N',max([entries(units=="N").max_abs_error]), ...
        'max_torque_error_Nm',max([entries(units=="Nm").max_abs_error]));
end
