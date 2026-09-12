function export_native_solid_probe(inventoryPath, blockPath, output, originalLayout)
%EXPORT_NATIVE_SOLID_PROBE Independent native poses of a cylinder's custom frames.
    arguments
        inventoryPath (1,1) string
        blockPath (1,1) string
        output (1,1) string
        originalLayout (1,1) logical = false
    end
    assert(strcmp(version('-release'), '2025b'));
    assert(~isfile(output), 'Preserve existing native evidence');
    inventory = jsondecode(fileread(inventoryPath));
    blocks = inventory.blocks;
    if iscell(blocks); blocks = [blocks{:}]; end
    block = blocks(strcmp({blocks.path}, blockPath));
    assert(isscalar(block), 'Native block must be unique');
    parameters = block.parameters;
    if iscell(parameters); parameters = [parameters{:}]; end
    model = 'native_solid_probe';
    new_system(model);
    cleanupModel = onCleanup(@() close_system(model, 0)); %#ok<NASGU>
    add_block('sm_lib/Frames and Transforms/World Frame', [model '/world']);
    assert(ismember(block.library_reference, {'sm_lib/Body Elements/Cylindrical Solid', ...
        'sm_lib/Body Elements/Spherical Solid'}), 'Unsupported solid library');
    add_block(block.library_reference, [model '/solid']);
    worldPorts = get_param([model '/world'], 'PortHandles');
    solidPorts = get_param([model '/solid'], 'PortHandles');
    add_block('sm_lib/Joints/Revolute Joint', [model '/joint']);
    jointPorts = get_param([model '/joint'], 'PortHandles');
    add_line(model, worldPorts.RConn(1), jointPorts.LConn(1));
    add_line(model, jointPorts.RConn(1), solidPorts.RConn(1));
    add_block(sprintf('nesl_utility/Solver\nConfiguration'), [model '/solver']);
    solverPorts = get_param([model '/solver'], 'PortHandles');
    add_line(model, worldPorts.RConn(1), solverPorts.RConn(1));
    dimensions = ["CylinderRadius", "CylinderLength"];
    if strcmp(block.library_reference, 'sm_lib/Body Elements/Spherical Solid')
        dimensions = "SphereRadius";
    end
    for name = dimensions
        parameter = parameters(strcmp({parameters.name}, name));
        assert(parameter.resolved_numeric);
        unit = parameters(strcmp({parameters.name}, name + "Units"));
        set_param([model '/solid'], name, num2str(parameter.numeric_value, 17), ...
            name + "Units", unit.expression);
    end
    frames = parameters(strcmp({parameters.name}, 'SerializedFrames'));
    set_param([model '/solid'], 'SerializedFrames', frames.expression);
    factory = javax.xml.parsers.DocumentBuilderFactory.newInstance();
    builder = factory.newDocumentBuilder();
    serialized = frames.expression;
    if isempty(serialized); serialized = '<Frames/>'; end
    document = builder.parse(org.xml.sax.InputSource(java.io.StringReader(serialized)));
    names = document.getElementsByTagName('Name');
    firstName = 'R';
    if names.getLength() > 0; firstName = char(names.item(0).getTextContent()); end
    follower = [model '/solid/' firstName];
    measured = {};
    if originalLayout
        delete_line(get_param(jointPorts.RConn(1), 'Line'));
        expose = parameters(strcmp({parameters.name}, 'DoExposeReferenceFrame'));
        set_param([model '/solid'], 'DoExposeReferenceFrame', expose.expression);
        nativePorts = get_param([model '/solid'], 'PortHandles');
        anchor = [nativePorts.LConn, nativePorts.RConn];
        assert(~isempty(anchor));
        add_line(model, jointPorts.RConn(1), anchor(1));
        frameIds = document.getElementsByTagName('Id');
        for n = 0:names.getLength()-1
            measured{end+1} = struct('identity', char(frameIds.item(n).getTextContent()), ...
                'kind', 'named_frame', 'path', [model '/solid/' char(names.item(n).getTextContent())]); %#ok<AGROW>
        end
        if strcmp(expose.expression, 'on')
            measured{end+1} = struct('identity', 'R', 'kind', 'named_frame', ...
                'path', [model '/solid/R']);
        end
        for side = ["LConn", "RConn"]
            handles = nativePorts.(side);
            for n = 1:numel(handles)
                portName = char(side + string(n));
                sensor = [model '/sensor_' portName];
                add_block(sprintf('sm_lib/Frames and\nTransforms/Transform\nSensor'), sensor);
                sensorPorts = get_param(sensor, 'PortHandles');
                add_line(model, worldPorts.RConn(1), sensorPorts.LConn(1));
                add_line(model, handles(n), sensorPorts.RConn(1));
                measured{end+1} = struct('identity', portName, 'kind', 'physical_port', ...
                    'path', [sensor '/F']); %#ok<AGROW>
            end
        end
    end
    ks = simscape.multibody.KinematicsSolver(model, ...
        'DefaultAngleUnit', 'rad', 'DefaultLengthUnit', 'm');
    variables = jointPositionVariables(ks);
    assert(height(variables) == 1);
    addTargetVariables(ks, variables.ID);
    addFrameVariables(ks, 'top', 'Translation', [model '/world/W'], follower);
    addFrameVariables(ks, 'top', 'Rotation', [model '/world/W'], follower);
    ids = ["top.Translation.x";"top.Translation.y";"top.Translation.z"; ...
        "top.Rotation.x";"top.Rotation.y";"top.Rotation.z"];
    addOutputVariables(ks, ids);
    for n = 1:numel(measured)
        group = "measurement" + string(n);
        addFrameVariables(ks, group, 'Translation', [model '/world/W'], measured{n}.path);
        addFrameVariables(ks, group, 'Rotation', [model '/world/W'], measured{n}.path);
        addOutputVariables(ks, [group + ".Translation." + ["x";"y";"z"]; ...
            group + ".Rotation." + ["x";"y";"z"]]);
    end
    [values, flag] = solve(ks, 0);
    assert(flag == 1 && all(isfinite(values)));
    result = struct('matlab_release', version('-release'), 'block', blockPath, ...
        'translation_m', values(1:3), 'rotation_intrinsic_xyz_rad', values(4:6), ...
        'qualification', 'native isolated custom-frame pose only');
    for n = 1:numel(measured)
        measured{n}.translation_m = values(6*n+(1:3));
        measured{n}.rotation_intrinsic_xyz_rad = values(6*n+(4:6));
    end
    result.original_port_layout = originalLayout;
    result.measurements = measured;
    fid = fopen(output, 'w'); assert(fid ~= -1);
    cleanupOutput = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s', jsonencode(result, PrettyPrint=true));
end
