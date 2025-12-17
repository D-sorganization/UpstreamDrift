varsbefore = who; %// get names of current variables (note 1)


% The script


varsafter = []; %// initiallize so that this variable is seen by next 'who'
varsnew = []; %// initiallize too.
varsafter = who; %// get names of all variables in 'varsbefore' plus variables
%// defined in the script, plus 'varsbefore', 'varsafter'  and 'varsnew'
varsnew = setdiff(varsafter, varsbefore); %// variables  defined in the script
%// plus 'varsbefore', 'varsafter'  and 'varsnew'
clear(varsnew{:}) %// (note 2)
