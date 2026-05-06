function names = all_coefficient_names()
%ALL_COEFFICIENT_NAMES Full ordered list of polynomial-coefficient names.
%
%   NAMES = ALL_COEFFICIENT_NAMES() returns a 1xN cell array of strings
%   formed by appending each coefficient letter (A..G) to each unique
%   polynomial-input base, in the canonical first-seen order from
%   TORQUE_TO_POLYNOMIAL_BASE. The ordering matches the Python module
%   src/shared/python/motion_matching/control_names.all_coefficient_names().

    % Unique polynomial bases in first-seen order (mirrors Python).
    bases = { ...
        'LScapInputX', 'LScapInputY', ...
        'RScapInputX', 'RScapInputY', ...
        'LSInputX',    'LSInputY',    'LSInputZ', ...
        'RSInputX',    'RSInputY',    'RSInputZ', ...
        'SpineInputX', 'SpineInputY', ...
        'TranslationInputX', 'TranslationInputY', 'TranslationInputZ', ...
        'HipInputX',   'HipInputY',   'HipInputZ' ...
    };
    letters = control_names.get_coefficient_letters();

    n = numel(bases) * numel(letters);
    names = cell(1, n);
    k = 1;
    for ii = 1:numel(bases)
        for jj = 1:numel(letters)
            names{k} = [bases{ii} letters{jj}];
            k = k + 1;
        end
    end
end
