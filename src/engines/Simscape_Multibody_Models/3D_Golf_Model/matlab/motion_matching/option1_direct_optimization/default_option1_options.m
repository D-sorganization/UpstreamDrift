function options = default_option1_options()
%DEFAULT_OPTION1_OPTIONS  Canonical options struct for Option 1 fits.
%
%   OPTIONS = DEFAULT_OPTION1_OPTIONS() returns the default options struct
%   consumed by every fit_swing_* entry point. Override individual fields
%   and pass the result to fit_swing_*. Unknown fields produce a clean
%   error from validators.mustBeOption1Options (NOT a silent override).
%
%   See INTERFACES.md "default_option1_options" for the full table of
%   fields, types, and defaults.
%
%   Postconditions:
%     - All fields documented in INTERFACES.md are present.
%     - Field types and ranges satisfy validators.mustBeOption1Options.
%
%   GitHub issue: #024
    error("not implemented");
end
