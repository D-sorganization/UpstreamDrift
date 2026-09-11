function result = extract_golf_joint_kinematics(logs, joint_names, time)
%EXTRACT_GOLF_JOINT_KINEMATICS Convert direct CombinedSignalBus joint logs to SI.
% Contract: GolfSwing3D_Kinetic angular converters emit degrees; translations
% emit metres. This named schema is not an arbitrary-model unit inference.
% Missing signals remain NaN. Source names are retained for audit.
    arguments
        logs (1,1) struct
        joint_names (1,:) string {mustBeNonempty}
        time (:,1) double {mustBeReal, mustBeFinite}
    end
    count = numel(joint_names);
    result = struct('q', nan(numel(time),count), 'qd', nan(numel(time),count), ...
        'qdd', nan(numel(time),count), 'coordinate_units', repmat("rad",1,count));
    fields = ["q", "qd", "qdd"];
    quantities = ["Position", "Velocity", "Acceleration"];
    result.source_names = struct('q', strings(1,count), ...
        'qd', strings(1,count), 'qdd', strings(1,count));
    for j = 1:count
        name = joint_names(j);
        valid = ~isempty(regexp(name, ...
            '^(HipInput[XYZ]|TranslationInput[XYZ]|[LR]SInput[XYZ]|[LR]ScapInput[XY]|[LR]WInput[XY]|SpineInput[XY]|[LR][EF]Input|TorsoInput)$', 'once'));
        assert(valid, 'extract_golf_joint_kinematics:unknownJoint', ...
            'Unsupported golf coordinate: %s', name);
        scale = pi / 180;
        prefix = "Angular";
        base = extractBefore(name, "Input");
        axis = extractAfter(name, "Input");
        if startsWith(name, "TranslationInput")
            base = "Hip";
            prefix = "Hip";
            scale = 1;
            result.coordinate_units(j) = "m";
        elseif base == "Hip"
            prefix = "HipAngular";
        end
        root = base + "Logs";
        if ismember(base, ["LE", "RE", "LW", "RW"])
            root = root + "." + base + "Joint";
        end
        joint_logs = resolve_joint_logs(logs, root);
        for k = 1:numel(fields)
            source = prefix + quantities(k) + axis;
            if ismember(base, ["LS", "RS"]) && source == "AngularPositionZ"
                source = "AngularPosition_Z";
            end
            field = fields(k);
            result.source_names.(field)(j) = root + "." + source;
            if isfield(joint_logs, source)
                result.(field)(:,j) = scale * ...
                    resample_logged_signal(joint_logs.(source), time, 1, []);
            end
        end
    end
end

function logs = resolve_joint_logs(logs, root)
    chain = split(root, ".");
    for k = 1:numel(chain)
        if ~isstruct(logs) || ~isfield(logs, chain(k))
            logs = struct;
            return;
        end
        logs = logs.(chain(k));
    end
end
