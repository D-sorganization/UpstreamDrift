function body_target = load_body_target_json(filepath)
%LOAD_BODY_TARGET_JSON Parse a body_target_json_v1 file into a MATLAB struct.
%
%   BODY_TARGET = LOAD_BODY_TARGET_JSON(FILEPATH) reads the JSON file at
%   FILEPATH and returns a MATLAB struct containing the BodyTarget data.
%   This is used by the Option-4 Python bridge to efficiently pass full-body
%   target data into MATLAB without the overhead of dict/array marshalling
%   across the matlab.engine boundary.

    txt = fileread(filepath);
    body_target = jsondecode(txt);

    % jsondecode transposes 1D arrays into columns, which is fine,
    % but we should ensure time_s is a column vector and 
    % marker_xyz is [N, M, 3] instead of getting flattened or permuted.
    % Actually, MATLAB's jsondecode natively parses [[[x,y,z],...],...]
    % as a 3D array, but we need to permute it because JSON is row-major
    % and MATLAB is col-major. 
    % For a Python nested list of shape (N, M, 3), jsondecode produces
    % an array of shape (3, M, N). We need to permute it to (N, M, 3)
    % so it matches the expected indexing in compute_cost.
    
    if isfield(body_target, 'marker_xyz') && isnumeric(body_target.marker_xyz)
        % For N >= 2, M >= 3, jsondecode usually makes size (3, M, N).
        % Let's verify the dimensions.
        sz = size(body_target.marker_xyz);
        if length(sz) == 3 && sz(1) == 3
            body_target.marker_xyz = permute(body_target.marker_xyz, [3, 2, 1]);
        end
    end
end
