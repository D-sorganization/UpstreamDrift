function export_native_solid_probe(inventoryPath, blockPath, output)
%EXPORT_NATIVE_SOLID_PROBE Independent native poses of a cylinder's custom frames.
    arguments
        inventoryPath (1,1) string
        blockPath (1,1) string
        output (1,1) string
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
    add_block('sm_lib/Body Elements/Cylindrical Solid', [model '/solid']);
    worldPorts = get_param([model '/world'], 'PortHandles');
    solidPorts = get_param([model '/solid'], 'PortHandles');
    add_block('sm_lib/Joints/Revolute Joint', [model '/joint']);
    jointPorts = get_param([model '/joint'], 'PortHandles');
    add_line(model, worldPorts.RConn(1), jointPorts.LConn(1));
    add_line(model, jointPorts.RConn(1), solidPorts.RConn(1));
    add_block(sprintf('nesl_utility/Solver\nConfiguration'), [model '/solver']);
    solverPorts = get_param([model '/solver'], 'PortHandles');
    add_line(model, worldPorts.RConn(1), solverPorts.RConn(1));
    for name = ["CylinderRadius", "CylinderLength"]
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
    document = builder.parse(org.xml.sax.InputSource(java.io.StringReader(frames.expression)));
    names = document.getElementsByTagName('Name');
    assert(names.getLength() > 0, 'A custom frame is required');
    firstName = char(names.item(0).getTextContent());
    follower = [model '/solid/' firstName];
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
    [values, flag] = solve(ks, 0);
    assert(flag == 1 && all(isfinite(values)));
    result = struct('matlab_release', version('-release'), 'block', blockPath, ...
        'translation_m', values(1:3), 'rotation_intrinsic_xyz_rad', values(4:6), ...
        'qualification', 'native isolated custom-frame pose only');
    fid = fopen(output, 'w'); assert(fid ~= -1);
    cleanupOutput = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s', jsonencode(result, PrettyPrint=true));
end
