figure(324);
hold on;
plot(ZTCFQ.Time,ZTCFQ.BaseAV);
plot(ZTCFQ.Time,ZTCFQ.ChestAV);
plot(ZTCFQ.Time,ZTCFQ.LScapAV);
plot(ZTCFQ.Time,ZTCFQ.LUpperArmAV);
plot(ZTCFQ.Time,ZTCFQ.LForearmAV);
plot(ZTCFQ.Time,ZTCFQ.ClubhandleAV);

xlabel('Time (s)');
ylabel('Angular Speed (deg/s)');
grid 'on';

%Add Legend to Plot
legend('Hips','Torso','Scap','Left Upper Arm','Left Forearm','Clubhandle');
legend('Location','southeast');

%Add a Title
title('Kinematic Sequence');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Kinematic Sequence');
pause(PauseTime);

%Close Figure
close(324);