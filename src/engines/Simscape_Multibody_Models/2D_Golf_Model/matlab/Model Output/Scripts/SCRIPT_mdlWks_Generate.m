%SCRIPT_mdlWksGenerate.m;
cd(fileparts(fileparts(mfilename('fullpath'))));
mdlWks=get_param('GolfSwing','ModelWorkspace');
mdlWks.DataSource = 'MAT-File';
mdlWks.FileName = 'ModelInputs.mat';
cd(fileparts(fileparts(mfilename('fullpath')))); %added to see if it fixes things
mdlWks.reload;