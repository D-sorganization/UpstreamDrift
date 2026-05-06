function dataset = load_sweep_dataset(path, opts)
%LOAD_SWEEP_DATASET  Read the random-sweep parquet dataset.
%   DATASET = LOAD_SWEEP_DATASET(PATH) calls the Python loader via
%   pyrunfile and converts the result to MATLAB tables.
%
%   DATASET = LOAD_SWEEP_DATASET(PATH, OPTS) accepts an options struct.
%       opts.lazy (logical, default true) -- request a polars LazyFrame
%           for the timesteps table on the Python side. The MATLAB shim
%           always materialises to a pandas DataFrame before conversion.
%
%   The returned struct has fields:
%       trials          -- table, one row per simulation
%       timesteps       -- table, one row per simulation timestep
%       joint_names     -- string array, joint ordering
%       schema_version  -- string, schema version of the loaded dataset
%
%   See also: src/shared/python/motion_matching/dataset/sweep.py
    arguments
        path (1,1) string {mustBeNonzeroLengthText}
        opts.lazy (1,1) logical = true
    end

    py_lazy = opts.lazy;
    py_path = string(path);

    % Use pyrunfile via an inline pyrun call. The Python module is on the
    % repo's sys.path because conftest.py / package __init__ files mark
    % src/shared/python as a package root.
    py_result = pyrun([
        "from pathlib import Path"
        "from src.shared.python.motion_matching.dataset import load_sweep_dataset"
        "ds = load_sweep_dataset(Path(p), lazy=lazy)"
        "trials_records = ds.trials.to_dict(orient='records')"
        "ts = ds.timesteps"
        "if hasattr(ts, 'collect'):"
        "    ts = ts.collect().to_pandas()"
        "timesteps_records = ts.to_dict(orient='records')"
        "joint_names = list(ds.joint_names)"
        "schema_version = ds.schema_version"
        ], ...
        ["trials_records", "timesteps_records", "joint_names", "schema_version"], ...
        p=py_path, lazy=py_lazy);

    dataset = struct();
    dataset.trials = py_records_to_table(py_result{1});
    dataset.timesteps = py_records_to_table(py_result{2});
    dataset.joint_names = string(py_result{3});
    dataset.schema_version = string(py_result{4});
end

function tbl = py_records_to_table(py_records)
    cell_records = cell(py_records);
    if isempty(cell_records)
        tbl = table();
        return;
    end
    n = numel(cell_records);
    fields = string(py.list(py_records{1}.keys()));
    s(n) = struct();
    for i = 1:n
        rec = py_records{i};
        for f = 1:numel(fields)
            key = fields(f);
            s(i).(key) = matlab_from_py(rec{key});
        end
    end
    tbl = struct2table(s, "AsArray", true);
end

function v = matlab_from_py(x)
    if isa(x, "py.list") || isa(x, "py.tuple")
        v = double(py.numpy.asarray(x));
    elseif isa(x, "py.str")
        v = string(x);
    elseif isa(x, "py.int") || isa(x, "py.float")
        v = double(x);
    else
        v = x;
    end
end
