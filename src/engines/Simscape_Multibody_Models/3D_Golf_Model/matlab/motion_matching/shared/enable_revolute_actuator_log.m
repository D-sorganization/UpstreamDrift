function enable_revolute_actuator_log(model)
%ENABLE_REVOLUTE_ACTUATOR_LOG Add primitive effort sensing to the loaded reference.
% Mutates only the loaded diagram; caller owns saving an isolated model copy.
% Existing signal-bus elements and physical connections are preserved.
    arguments
        model (1,:) char
    end
    assert(strcmp(version('-release'),'2025b'),'enable_revolute_actuator_log:release','R2025b is required');
    assert(bdIsLoaded(model),'enable_revolute_actuator_log:loaded','Load the reference model first');
    joint=[model '/Kinetically Driven Revolute'];
    converter=[model '/ActuatorTorque'];
    bus=find_system(model,'SearchDepth',1,'BlockType','BusCreator');
    assert(numel(bus)==1,'enable_revolute_actuator_log:bus','Expected exactly one signal bus');
    if getSimulinkBlockHandle(converter)>0
        ports=get_param(bus{1},'PortHandles');
        assert(strcmp(get_param(joint,'SenseTorqueForce'),'on') && ...
            strcmp(get_param(get_param(ports.Inport(end),'Line'),'Name'),'ActuatorTorque'), ...
            'enable_revolute_actuator_log:existing','Existing actuator instrumentation is inconsistent');
        return
    end
    count=str2double(get_param(bus{1},'Inputs'));
    assert(count==12,'enable_revolute_actuator_log:bus','Expected the original twelve-channel reference bus');
    set_param(joint,'SenseTorqueForce','on');
    ports=get_param(joint,'PortHandles');
    free=ports.RConn(arrayfun(@(port)get_param(port,'Line')==-1,ports.RConn));
    assert(numel(free)==1,'enable_revolute_actuator_log:sensorPort','Expected one newly exposed actuator-sensing port');
    position=get_param([model '/TorqueLocal'],'Position')+[0 80 0 80];
    add_block([model '/TorqueLocal'],converter,'Position',position,'Unit','N*m');
    sensor=get_param(converter,'PortHandles');
    add_line(model,free,sensor.LConn(1),'autorouting','on');
    set_param(bus{1},'Inputs',num2str(count+1));
    bus_ports=get_param(bus{1},'PortHandles');
    line=add_line(model,sensor.Outport(1),bus_ports.Inport(end),'autorouting','on');
    set_param(line,'Name','ActuatorTorque');
end
