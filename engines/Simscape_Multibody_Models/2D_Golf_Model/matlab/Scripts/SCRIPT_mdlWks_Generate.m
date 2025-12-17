%SCRIPT_mdlWksGenerate.m;
cd(matlabdrive);
cd '2DModel';
mdlWks=get_param('GolfSwing','ModelWorkspace');
mdlWks.DataSource = 'MAT-File';
mdlWks.FileName = 'ModelInputs.mat';
cd(matlabdrive); 
cd '2DModel'%added to see if it fixes things
mdlWks.reload;