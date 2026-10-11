function p = default_impact_input_mat()
%DEFAULT_IMPACT_INPUT_MAT  Path to the committed impact-pose model inputs.
%
%   P = DEFAULT_IMPACT_INPUT_MAT() is the char path of
%   matlab/src/model/inputs/3DModelInputs_Impact.mat, resolved from this
%   file (motion_matching/shared/ -> motion_matching/ -> matlab/).  It does
%   not check that the file exists; callers decide what a missing file means.
    matlab_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    p = fullfile(matlab_root, 'src', 'model', 'inputs', '3DModelInputs_Impact.mat');
end
