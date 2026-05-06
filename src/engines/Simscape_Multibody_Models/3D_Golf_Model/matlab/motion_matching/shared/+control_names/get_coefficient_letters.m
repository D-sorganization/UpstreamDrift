function letters = get_coefficient_letters()
%GET_COEFFICIENT_LETTERS Canonical coefficient suffix letters A..G.
%
%   LETTERS = GET_COEFFICIENT_LETTERS() returns the 1x7 cell array
%   {'A','B','C','D','E','F','G'} — the suffixes appended to each
%   polynomial-input base to form the 7 coefficients per joint
%   (degree-6 polynomial). Mirrors COEFFICIENT_LETTERS in
%   src/shared/python/motion_matching/control_names.py.
    letters = {'A', 'B', 'C', 'D', 'E', 'F', 'G'};
end
