function tests = test_checkpoint_security
    % TEST_CHECKPOINT_SECURITY - Security tests for checkpoint_manager
    %
    % This test suite verifies that the checkpoint_manager correctly handles
    % security-critical inputs and prevents vulnerabilities like path traversal.

    arguments
    end

    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    % Setup test environment
    arguments
        testCase
    end
    % Paths are assumed to be managed externally as per project standards
end

function test_path_traversal_prevention(testCase)
    % Test that stage names with path traversal characters are rejected
    arguments
        testCase
    end

    % Create a mock config
    config.cache_path = fullfile(tempdir, 'test_cache');
    config.enable_checkpoints = true;
    config.verbose = false;

    % Initialize checkpoint manager
    % We assume checkpoint_manager is in the path
    cm = checkpoint_manager(config);

    % Define invalid stage names
    invalid_names = {
        '../parent_dir', ...
        'dir/subdir', ...
        '..\windows_style', ...
        'dir\subdir', ...
        '~home', ...
        'name.with.dots' % Depending on policy, maybe allowed, but we restricted to alphanumeric
    };

    % dummy data
    data = struct('value', 1);

    for i = 1:length(invalid_names)
        name = invalid_names{i};
        verifyError(testCase, @() cm.save(name, data), ...
            'CheckpointManager:InvalidStageName', ...
            sprintf('Failed to reject invalid name: %s', name));

        verifyError(testCase, @() cm.load(name), ...
            'CheckpointManager:InvalidStageName', ...
            sprintf('Failed to reject invalid name in load: %s', name));
    end

    % Verify valid names still work (or at least don't throw InvalidStageName)
    valid_names = {
        'base_data', ...
        'Stage_1', ...
        'my-checkpoint'
    };

    for i = 1:length(valid_names)
        name = valid_names{i};
        try
            cm.save(name, data);
            % It might fail to save if directory doesn't exist/writable,
            % but it should NOT be InvalidStageName
        catch ME
             if strcmp(ME.identifier, 'CheckpointManager:InvalidStageName')
                 testCase.verifyFail(sprintf('Valid name rejected: %s', name));
             end
        end
    end
end
