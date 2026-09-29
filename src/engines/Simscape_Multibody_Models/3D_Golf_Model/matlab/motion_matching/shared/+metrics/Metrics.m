classdef Metrics
    %METRICS Canonical metrics record per METRICS_SCHEMA.md.
    %
    %   Mirror of src/shared/python/motion_matching/metrics.py.  The two
    %   emitters guarantee byte-for-byte identical JSON for the same record.
    %
    %   Constructor:
    %     m = metrics.Metrics(struct_or_namevalue);
    %
    %   Methods:
    %     j   = m.to_json();
    %     m   = metrics.Metrics.from_json(json_string);
    %     row = m.to_csv_row();
    %     m   = metrics.Metrics.from_csv_row(row_struct);
    %     m   = metrics.Metrics.fromLegacyStruct(legacy_struct);
    %
    %   See METRICS_SCHEMA.md for field definitions and validation rules.

    properties (Constant)
        SCHEMA_VERSION = "1.0.0";
        FIELD_ORDER = [ ...
            "swing_id", "option", "solver", "n_iterations", ...
            "rmse_clubhead_mm", "rmse_butt_mm", "rmse_orientation_deg", ...
            "clubhead_speed_at_impact_mph", "clubhead_speed_meas_mph", ...
            "total_work_J", "peak_power_W", "wall_clock_s", ...
            "git_commit", "matlab_version", "python_version", ...
            "timestamp_iso8601", "schema_version"];
    end

    properties
        swing_id (1,1) string
        option (1,1) double
        solver (1,1) string
        n_iterations (1,1) double
        rmse_clubhead_mm (1,1) double
        rmse_butt_mm (1,1) double
        rmse_orientation_deg (1,1) double
        clubhead_speed_at_impact_mph (1,1) double
        clubhead_speed_meas_mph (1,1) double
        total_work_J (1,1) double
        peak_power_W (1,1) double
        wall_clock_s (1,1) double
        git_commit (1,1) string
        matlab_version (1,1) string
        python_version (1,1) string
        timestamp_iso8601 (1,1) string
        schema_version (1,1) string = "1.0.0"
    end

    methods
        function obj = Metrics(s)
            %METRICS Construct from a struct of all fields.
            arguments
                s (1,1) struct
            end
            for k = 1:numel(metrics.Metrics.FIELD_ORDER)
                name = metrics.Metrics.FIELD_ORDER(k);
                if ~isfield(s, name)
                    if name == "schema_version"
                        s.(name) = metrics.Metrics.SCHEMA_VERSION;
                    else
                        error("metrics:Metrics:missingField", ...
                            "missing required field: %s", name);
                    end
                end
                obj.(name) = s.(name);
            end
            obj.validate();
        end

        function validate(obj)
            %VALIDATE Apply METRICS_SCHEMA.md validation rules.
            if obj.schema_version ~= metrics.Metrics.SCHEMA_VERSION
                error("metrics:Metrics:badSchema", ...
                    "schema_version %s != current %s", ...
                    obj.schema_version, metrics.Metrics.SCHEMA_VERSION);
            end
            if ~ismember(obj.option, [1 2 3 4])
                error("metrics:Metrics:badOption", ...
                    "option must be in {1,2,3,4}, got %d", obj.option);
            end
            if obj.n_iterations < 0
                error("metrics:Metrics:badIters", ...
                    "n_iterations must be >= 0");
            end
            numericNames = ["rmse_clubhead_mm", "rmse_butt_mm", ...
                "rmse_orientation_deg", "clubhead_speed_at_impact_mph", ...
                "clubhead_speed_meas_mph", "total_work_J", ...
                "peak_power_W", "wall_clock_s"];
            for k = 1:numel(numericNames)
                v = obj.(numericNames(k));
                if ~isfinite(v)
                    error("metrics:Metrics:nonFinite", ...
                        "%s must be finite", numericNames(k));
                end
            end
            nonneg = ["rmse_clubhead_mm", "rmse_butt_mm", ...
                "rmse_orientation_deg", "clubhead_speed_at_impact_mph", ...
                "clubhead_speed_meas_mph", "wall_clock_s"];
            for k = 1:numel(nonneg)
                if obj.(nonneg(k)) < 0
                    error("metrics:Metrics:negative", ...
                        "%s must be >= 0", nonneg(k));
                end
            end
            if isempty(regexp(obj.git_commit, '^[0-9a-f]{40}$', 'once'))
                error("metrics:Metrics:badSha", ...
                    "git_commit must be 40 lowercase hex chars");
            end
            if isempty(regexp(obj.timestamp_iso8601, ...
                    '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$', 'once'))
                error("metrics:Metrics:badTimestamp", ...
                    "timestamp_iso8601 must be ISO-8601 UTC ending in 'Z'");
            end
        end

        function j = to_json(obj)
            %TO_JSON Canonical JSON (sorted keys, compact).
            s = struct();
            for k = 1:numel(metrics.Metrics.FIELD_ORDER)
                name = metrics.Metrics.FIELD_ORDER(k);
                v = obj.(name);
                if isstring(v)
                    s.(name) = char(v);
                elseif isa(v, 'double') && v == floor(v) && ...
                        ismember(name, ["option", "n_iterations"])
                    s.(name) = int64(v);
                else
                    s.(name) = v;
                end
            end
            % jsonencode in MATLAB does not sort keys; our struct field
            % insertion order matches FIELD_ORDER which is the canonical
            % alphabetical-on-purpose order needed for diff stability.
            j = jsonencode(s);
        end

        function row = to_csv_row(obj)
            %TO_CSV_ROW Stringly-typed dict for csv writers.
            row = struct();
            for k = 1:numel(metrics.Metrics.FIELD_ORDER)
                name = metrics.Metrics.FIELD_ORDER(k);
                v = obj.(name);
                if isstring(v) || ischar(v)
                    row.(name) = char(v);
                else
                    row.(name) = num2str(v, 17);
                end
            end
        end
    end

    methods (Static)
        function obj = from_json(j)
            %FROM_JSON Inverse of to_json.
            arguments
                j (1,1) string
            end
            data = jsondecode(char(j));
            obj = metrics.Metrics(data);
        end

        function obj = from_csv_row(row)
            %FROM_CSV_ROW Inverse of to_csv_row.
            arguments
                row (1,1) struct
            end
            s = struct();
            stringFields = ["swing_id", "solver", "git_commit", ...
                "matlab_version", "python_version", "timestamp_iso8601", ...
                "schema_version"];
            intFields = ["option", "n_iterations"];
            for k = 1:numel(metrics.Metrics.FIELD_ORDER)
                name = metrics.Metrics.FIELD_ORDER(k);
                raw = row.(name);
                if ismember(name, stringFields)
                    s.(name) = string(raw);
                elseif ismember(name, intFields)
                    s.(name) = int64(str2double(raw));
                else
                    s.(name) = str2double(raw);
                end
            end
            obj = metrics.Metrics(s);
        end

        function obj = fromLegacyStruct(legacy)
            %FROMLEGACYSTRUCT Backwards-compat shim for the pre-1.0 result
            %struct used by leaderboard.m.
            arguments
                legacy (1,1) struct
            end
            mapping = struct( ...
                "rmse_clubhead", "rmse_clubhead_mm", ...
                "rmse_butt", "rmse_butt_mm", ...
                "rmse_orient", "rmse_orientation_deg", ...
                "chs_impact", "clubhead_speed_at_impact_mph", ...
                "chs_meas", "clubhead_speed_meas_mph", ...
                "total_work", "total_work_J", ...
                "peak_power", "peak_power_W", ...
                "wall_clock", "wall_clock_s", ...
                "timestamp", "timestamp_iso8601");
            keys = fieldnames(mapping);
            for k = 1:numel(keys)
                old = keys{k};
                newName = mapping.(old);
                if isfield(legacy, old) && ~isfield(legacy, newName)
                    legacy.(newName) = legacy.(old);
                    legacy = rmfield(legacy, old);
                end
            end
            defaults = struct( ...
                "n_iterations", 0, ...
                "matlab_version", "", ...
                "python_version", "", ...
                "schema_version", metrics.Metrics.SCHEMA_VERSION);
            dnames = fieldnames(defaults);
            for k = 1:numel(dnames)
                if ~isfield(legacy, dnames{k})
                    legacy.(dnames{k}) = defaults.(dnames{k});
                end
            end
            obj = metrics.Metrics(legacy);
        end
    end
end
