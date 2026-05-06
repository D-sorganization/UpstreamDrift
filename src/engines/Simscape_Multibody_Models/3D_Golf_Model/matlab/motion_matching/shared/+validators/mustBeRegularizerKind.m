function mustBeRegularizerKind(name)
%MUSTBEREGULARIZERKIND Validate that NAME is an allowed regularizer kind.
%
%   mustBeRegularizerKind(NAME) errors when NAME (string or char) is not one
%   of the allowed regularizer kinds defined in COST_FUNCTION_SPEC.md:
%       "total_work" | "peak_power" | "torque_l2" | "coeff_l2"
%       | "effort_l2" | "smoothness_l2"
%
%   Used in arguments blocks and runtime validation for opts.regularizer.
%
%   Error identifiers:
%     validator:notTextScalar    - NAME is not a scalar string/char
%     validator:badRegularizer   - NAME is not in the allowed set
    allowed = ["total_work", "peak_power", "torque_l2", "coeff_l2", ...
               "effort_l2", "smoothness_l2"];
    if ~(isstring(name) && isscalar(name)) && ~(ischar(name) && isrow(name))
        error("validator:notTextScalar", ...
              "Regularizer name must be a scalar string or char row.");
    end
    s = lower(string(name));
    if ~any(s == allowed)
        error("validator:badRegularizer", ...
              "Regularizer '%s' is not one of: %s.", ...
              s, strjoin(allowed, ", "));
    end
end
