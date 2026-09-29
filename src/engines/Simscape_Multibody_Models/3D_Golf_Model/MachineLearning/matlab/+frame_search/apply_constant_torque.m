function variables = apply_constant_torque(torqueRow, controlColumns)
%APPLY_CONSTANT_TORQUE  Build the workspace variable overrides for a candidate.
%
%   variables = frame_search.apply_constant_torque(torqueRow, controlColumns)
%   returns a struct whose field names are the polynomial-input scalar
%   variables expected by GolfSwing3D_Kinetic and whose values implement a
%   constant-in-time torque for this candidate. For each control column the
%   helper sets the polynomial constant term (suffix 'G') to the candidate
%   torque value and zeroes the higher-order coefficients (A..F) so the
%   model's polynomial input block evaluates to a flat torque over the
%   short horizon.
%
%   Inputs:
%     torqueRow      (1, K) double  Constant torque per control.
%     controlColumns 1xK cell of char or string array of manifest control
%                                    column names (e.g. 'LSLogs_ActuatorTorqueX').
%
%   Output:
%     variables      struct with fields like LSInputXA..LSInputXG.
%
%   Errors:
%     - Throws frame_search:apply_constant_torque:unknownControl if a column
%       cannot be mapped to a polynomial-input base name.
%     - Throws frame_search:apply_constant_torque:sizeMismatch if torqueRow
%       length does not match controlColumns length.
%
%   This helper is pure (no Simulink calls) and unit-testable.

    arguments
        torqueRow (1, :) double {mustBeFinite}
        controlColumns
    end

    if iscell(controlColumns)
        controlColumns = string(controlColumns);
    elseif ischar(controlColumns)
        controlColumns = string({controlColumns});
    end
    controlColumns = controlColumns(:).';

    if numel(torqueRow) ~= numel(controlColumns)
        error('frame_search:apply_constant_torque:sizeMismatch', ...
            'torqueRow has %d elements, controlColumns has %d.', ...
            numel(torqueRow), numel(controlColumns));
    end

    variables = struct();
    letters = 'ABCDEFG';
    for idx = 1:numel(controlColumns)
        base = frame_search.control_column_to_polynomial_base(controlColumns(idx));
        if strlength(base) == 0
            error('frame_search:apply_constant_torque:unknownControl', ...
                'Cannot map control column %s to a polynomial-input base.', ...
                controlColumns(idx));
        end
        for letterIdx = 1:numel(letters) - 1   % A..F = 0
            variables.(strcat(char(base), letters(letterIdx))) = 0.0;
        end
        % G is the polynomial constant term in the model convention used by
        % torqueColumnToPolynomialBase / export_torque_polynomials.
        variables.(strcat(char(base), 'G')) = double(torqueRow(idx));
    end
end
