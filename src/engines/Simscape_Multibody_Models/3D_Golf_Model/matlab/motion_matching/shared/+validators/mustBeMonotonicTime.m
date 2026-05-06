function mustBeMonotonicTime(t)
%MUSTBEMONOTONICTIME Validate that T is a strictly-increasing time vector.
%
%   mustBeMonotonicTime(T) errors when T is not a real, finite, strictly
%   increasing column or row vector. Used to validate target.time per
%   CLUB_IK_SPEC.md § Validation rule 1.
%
%   Error identifiers:
%     validator:notNumeric     - T is not numeric or is complex
%     validator:notVector      - T is empty or not a vector
%     validator:notFinite      - T contains NaN or Inf
%     validator:notMonotonic   - T is not strictly increasing
    if ~isnumeric(t) || ~isreal(t)
        error("validator:notNumeric", ...
              "Time vector must be a real numeric array.");
    end
    if isempty(t) || ~isvector(t)
        error("validator:notVector", ...
              "Time vector must be a non-empty vector.");
    end
    if any(~isfinite(t(:)))
        error("validator:notFinite", ...
              "Time vector must contain only finite entries.");
    end
    d = diff(t(:));
    if any(d <= 0)
        error("validator:notMonotonic", ...
              "Time vector must be strictly increasing (found non-positive step).");
    end
end
