function tests = test_load_body_target_json
    tests = functiontests(localfunctions);
end

function test_load_valid_json(testCase)
    % Create a temporary JSON file representing a BodyTarget.
    % We will use a 2x3x3 array for marker_xyz (N=2 frames, M=3 markers, 3 coords).
    tmp_file = [tempname '.json'];
    
    % MATLAB jsonencode of a 3D array might permute dimensions depending on how it's shaped.
    % In Python, a list of shape (2, 3, 3) becomes:
    % [[[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]],
    %  [[x4,y4,z4], [x5,y5,z5], [x6,y6,z6]]]
    
    json_str = ['{' ...
        '"time_s": [0.0, 0.1], ' ...
        '"marker_names": ["C7", "R.Shoulder", "L.Shoulder"], ' ...
        '"marker_xyz": [' ...
            '[[1, 2, 3], [4, 5, 6], [7, 8, 9]], ' ...
            '[[10, 11, 12], [13, 14, 15], [16, 17, 18]]' ...
        ']' ...
    '}'];

    fid = fopen(tmp_file, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
    
    % Clean up when test finishes
    c = onCleanup(@() delete(tmp_file));
    
    % Call the unit under test
    bt = load_body_target_json(tmp_file);
    
    % Assertions
    verifyTrue(testCase, isfield(bt, 'time_s'));
    verifyTrue(testCase, isfield(bt, 'marker_names'));
    verifyTrue(testCase, isfield(bt, 'marker_xyz'));
    
    % time_s should be a column or row, we don't strictly enforce but check length
    verifyEqual(testCase, numel(bt.time_s), 2);
    verifyEqual(testCase, bt.time_s(1), 0.0);
    verifyEqual(testCase, bt.time_s(2), 0.1);
    
    % marker_names should have length 3
    verifyEqual(testCase, numel(bt.marker_names), 3);
    
    % marker_xyz should be reshaped by load_body_target_json to [N, M, 3] = [2, 3, 3]
    % Note: jsondecode parses the Python JSON array (N=2, M=3, 3) as size (3, M, N)
    % so it is (3, 3, 2). Then load_body_target_json permutes it with [3, 2, 1] to (2, 3, 3).
    sz = size(bt.marker_xyz);
    verifyEqual(testCase, sz, [2, 3, 3]);
    
    % Verify the first marker of the first frame [1, 2, 3]
    % bt.marker_xyz(1, 1, :) should be [1, 2, 3]
    verifyEqual(testCase, squeeze(bt.marker_xyz(1, 1, :)), [1; 2; 3]);
    
    % Verify the second marker of the second frame [13, 14, 15]
    verifyEqual(testCase, squeeze(bt.marker_xyz(2, 2, :)), [13; 14; 15]);
end

function test_missing_fields_graceful(testCase)
    % If the JSON is missing marker_xyz or it is not numeric, it shouldn't crash.
    tmp_file = [tempname '.json'];
    json_str = '{"time_s": [0.0, 0.1], "marker_names": []}';
    fid = fopen(tmp_file, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
    
    c = onCleanup(@() delete(tmp_file));
    
    bt = load_body_target_json(tmp_file);
    verifyTrue(testCase, isfield(bt, 'time_s'));
    verifyFalse(testCase, isfield(bt, 'marker_xyz'));
end
