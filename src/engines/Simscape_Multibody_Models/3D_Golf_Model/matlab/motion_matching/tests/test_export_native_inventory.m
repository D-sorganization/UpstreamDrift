function tests = test_export_native_inventory
%TEST_EXPORT_NATIVE_INVENTORY Exercise a real loaded model without saving it.
    tests = functiontests(localfunctions);
end

function testResolvesParametersAndPreservesConnections(testCase)
    model = "native_inventory_contract_test";
    new_system(model);
    cleanupModel = onCleanup(@() close_system(model, 0)); %#ok<NASGU>
    add_block('simulink/Sources/Constant', model + "/Input", 'Value', '2*pi');
    add_block('simulink/Math Operations/Gain', model + "/Gain", 'Gain', '3');
    add_line(model, 'Input/1', 'Gain/1');
    output = string(tempname) + ".json";
    cleanupOutput = onCleanup(@() deleteIfPresent(output)); %#ok<NASGU>
    export_native_inventory(model, output);
    result = jsondecode(fileread(output));
    verifyEqual(testCase, result.matlab_release, '2025b');
    blocks = asStructArray(result.blocks);
    input = blocks(strcmp({blocks.path}, model + "/Input"));
    parameters = asStructArray(input.parameters);
    value = parameters(strcmp({parameters.name}, 'Value'));
    verifyEqual(testCase, value.expression, '2*pi');
    verifyTrue(testCase, value.resolved_numeric);
    verifyEqual(testCase, value.numeric_value, 2*pi, 'AbsTol', 1e-12);
    gain = blocks(strcmp({blocks.path}, model + "/Gain"));
    verifyTrue(testCase, any(contains(jsonencode(gain.connectivity), '/Input')));
    verifyError(testCase, @() export_native_inventory(model, output), ...
        'nativeInventory:ExistingOutput');
end

function result = asStructArray(value)
    result = value;
    if iscell(value)
        result = [value{:}];
    end
end

function deleteIfPresent(path)
    if isfile(path)
        delete(path);
    end
end
