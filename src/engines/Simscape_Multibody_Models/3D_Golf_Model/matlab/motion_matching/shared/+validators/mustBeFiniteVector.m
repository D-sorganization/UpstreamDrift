function mustBeFiniteVector(v)
%MUSTBEFINITEVECTOR Validate that V is a real, finite numeric vector.
%
%   mustBeFiniteVector(V) errors when V is empty, non-numeric, complex,
%   non-vector, or contains any NaN/Inf entries. Used in arguments blocks
%   for coefficient vectors and other 1-D numeric inputs.
%
%   Error identifiers:
%     validator:notNumeric  - V is not numeric or is complex
%     validator:notVector   - V is not a vector (or is empty)
%     validator:notFinite   - V contains NaN or Inf
    if ~isnumeric(v) || ~isreal(v)
        error("validator:notNumeric", ...
              "Value must be a real numeric array.");
    end
    if isempty(v) || ~isvector(v)
        error("validator:notVector", ...
              "Value must be a non-empty vector.");
    end
    if any(~isfinite(v(:)))
        error("validator:notFinite", ...
              "Value must contain only finite entries (no NaN or Inf).");
    end
end
