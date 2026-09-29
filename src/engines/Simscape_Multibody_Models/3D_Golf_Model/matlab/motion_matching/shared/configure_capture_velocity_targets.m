function cleanup = configure_capture_velocity_targets(model_name)
%CONFIGURE_CAPTURE_VELOCITY_TARGETS Hold native initial-rate targets for a fit.
% Keep the returned onCleanup object alive across all forward replays. Its
% destruction releases Fast Restart and restores the original target settings.
% Configure once per fit session; no source model or MAT file is saved.
    arguments
        model_name (1,1) string = "GolfSwing3D_Kinetic"
    end
    schema=jsondecode(fileread(fullfile(fileparts(mfilename('fullpath')), ...
        'golf_kinematic_schema.json')));
    assert(model_name==string(schema.model_name) && bdIsLoaded(model_name), ...
        'configure_capture_velocity_targets:model','Load the qualified native golf model first.');
    set_param(model_name,'FastRestart','off');
    dependent=string(schema.dependent_coordinates);
    changes=struct('block',{},'parameter',{},'previous',{},'value',{});
    for j=1:numel(schema.coordinates)
        coordinate=schema.coordinates(j);block=char(coordinate.block_path);
        parent=get_param(block,'Parent');selector=char(string(coordinate.primitive)+"VelocityTargetPriority");
        enabled=~ismember(string(coordinate.name),dependent);
        parameters=get_param(parent,'DialogParameters');
        if isfield(parameters,selector)
            priority='None';if enabled;priority='High';end
            changes=add_change(changes,parent,selector,priority);
        else
            parameters=get_param(block,'DialogParameters');specify=char(string(coordinate.primitive)+"VelocityTargetSpecify");
            assert(isfield(parameters,selector) && isfield(parameters,specify), ...
                'configure_capture_velocity_targets:selector','Native velocity selector missing: %s',coordinate.name);
            value='off';if enabled;value='on';end
            changes=add_change(changes,block,specify,value);
            if enabled;changes=add_change(changes,block,selector,'High');end
        end
    end
    cleanup=onCleanup(@()restore_targets(model_name,changes));
    for j=1:numel(changes);set_param(changes(j).block,changes(j).parameter,changes(j).value);end
end
function changes=add_change(changes,block,parameter,value)
previous=get_param(block,parameter);
if ~strcmp(previous,value)
    changes(end+1)=struct('block',block,'parameter',parameter,'previous',previous,'value',value);
end
end
function restore_targets(model_name,changes)
if ~bdIsLoaded(model_name);return;end
set_param(model_name,'FastRestart','off');
for j=numel(changes):-1:1;set_param(changes(j).block,changes(j).parameter,changes(j).previous);end
end
