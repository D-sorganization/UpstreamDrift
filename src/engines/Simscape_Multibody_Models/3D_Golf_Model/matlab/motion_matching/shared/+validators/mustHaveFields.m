function mustHaveFields(s, names)
%MUSTHAVEFIELDS Validate that struct S contains all required field NAMES.
%
%   mustHaveFields(S, NAMES) errors with identifier "validator:missingField"
%   when one or more elements of NAMES are not fields of S. The error
%   message lists the missing fields, comma-separated, so the caller can
%   see exactly what the contract violation was.
%
%   This is the canonical struct-shape validator used by every public
%   function in the motion_matching shared package. See COST_FUNCTION_SPEC.md
%   and CLUB_IK_SPEC.md for the schemas it enforces.
    arguments
        s (1,1) struct
        names (1,:) string
    end
    actual  = string(fieldnames(s));
    missing = setdiff(names, actual);
    if ~isempty(missing)
        error("validator:missingField", ...
              "Struct missing required fields: %s", strjoin(missing, ", "));
    end
end
