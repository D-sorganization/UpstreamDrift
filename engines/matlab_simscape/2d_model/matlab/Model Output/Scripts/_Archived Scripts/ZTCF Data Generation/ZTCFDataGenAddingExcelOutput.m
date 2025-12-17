%ZTCF Data Generation Script
Base=sim(GolfSwing);
openBase('simulink/AccessDataLoggedAsSingleSimulationOutputExample')

writematrix('Base.Globalx','BaseData')
assignin(mdlWks,'KillswitchStepTime',0);

% ZTCF000=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.01);
% writetable(ZTCF001,'ZTCF001Data')
% ZTCF001=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.02);
% writetable(ZTCF001,'ZTCF001Data')
% ZTCF002=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.03);
% writetable(ZTCF002,'ZTCF002Data')
% ZTCF003=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.04);
% ZTCF004=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.05);
% ZTCF005=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.06);
% ZTCF006=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.07);
% ZTCF007=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.08);
% ZTCF008=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.09);
% ZTCF009=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.10);
% ZTCF010=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.11);
% ZTCF011=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.12);
% ZTCF012=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.13);
% ZTCF013=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.14);
% ZTCF014=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.15);
% ZTCF015=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.16);
% ZTCF016=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.17);
% ZTCF017=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.18);
% ZTCF018=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.19);
% ZTCF019=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.20);
% ZTCF020=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.21);
% ZTCF021=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.22);
% ZTCF022=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.23);
% ZTCF023=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.24);
% ZTCF024=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.25);
% ZTCF025=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.26);
% ZTCF026=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.27);
% ZTCF027=sim(GolfSwing);
% assignin(mdlWks,'KillswitchStepTime',0.28);
% ZTCF028=sim(GolfSwing);
% 
