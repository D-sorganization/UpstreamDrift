figure(124);
hold on;
plot(BASEQ.Time,BASEQ.BaseAV);
plot(BASEQ.Time,BASEQ.ChestAV);
plot(BASEQ.Time,BASEQ.LScapAV);
plot(BASEQ.Time,BASEQ.LUpperArmAV);
plot(BASEQ.Time,BASEQ.LForearmAV);
plot(BASEQ.Time,BASEQ.ClubhandleAV);

xlabel('Time (s)');
ylabel('Angular Speed (deg/s)');
grid 'on';

%Add Legend to Plot
legend('Hips','Torso','Scap','Left Upper Arm','Left Forearm','Clubhandle');
legend('Location','southeast');

%Add a Title
title('Kinematic Sequence');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Kinematic Sequence');
pause(PauseTime);

%Close Figure
close(124);