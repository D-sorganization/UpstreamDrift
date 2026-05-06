%SCRIPT_mdlWksGenerate.m;
scriptDir = fileparts(mfilename('fullpath'));
modelDir = fileparts(scriptDir);
cd(modelDir);
mdlWks=get_param('GolfSwing','ModelWorkspace');
mdlWks.DataSource = 'MAT-File';
mdlWks.FileName = 'ModelInputs.mat';
cd(modelDir); %added to see if it fixes things
mdlWks.reload;
