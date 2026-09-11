function report = repair_gimbal_initial_targets(model_file)
%REPAIR_GIMBAL_INITIAL_TARGETS Restore the six declared instance target bindings.
% Applies only to the known archived gimbal defect (#9927). Unexpected bindings
% are rejected before editing. Repeated application leaves the file unchanged.
    arguments
        model_file (1,1) string
    end
    [~, model, extension] = fileparts(model_file);
    assert(isfile(model_file) && model == "Kinetically_Driven_Gimbal_Joint" && ...
        extension == ".slx", 'repair_gimbal_initial_targets:invalidFile', ...
        'Expected the existing Kinetically_Driven_Gimbal_Joint.slx file.');
    assert(~bdIsLoaded(model), 'repair_gimbal_initial_targets:alreadyLoaded', ...
        'Close the referenced subsystem before applying the migration.');
    load_system(model_file);
    cleanup = onCleanup(@() bdclose(model)); %#ok<NASGU>
    primitive = model + "/Kinetically Driven";
    keys = strings(1,6);
    before = strings(1,6);
    after = strings(1,6);
    index = 0;
    for axis = 'XYZ'
        for quantity = ["Position", "Velocity"]
            index = index + 1;
            keys(index) = "R" + lower(string(axis)) + quantity + "TargetValue";
            after(index) = "Start" + quantity + string(axis);
            before(index) = string(get_param(primitive, keys(index)));
        end
    end
    assert(all(before == after | before == "LS" + after), ...
        'repair_gimbal_initial_targets:unexpectedBinding', ...
        'Unexpected primitive target binding; the migration has not saved changes.');
    changed = any(before ~= after);
    if changed
        for index = 1:numel(keys)
            set_param(primitive, keys(index), after(index));
        end
        % Save natively in the executing release. Backward export of Simscape
        % models is unsupported; qualify the resulting file in the run release.
        save_system(model, model_file);
    end
    report = struct('file', model_file, 'changed', changed, ...
        'parameters', keys, 'before', before, 'after', after);
end
