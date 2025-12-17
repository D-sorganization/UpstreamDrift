figure(924);
hold on;
plot(Data.Time,Data.BaseAV);
plot(Data.Time,Data.ChestAV);
plot(Data.Time,Data.LScapAV);
plot(Data.Time,Data.LUpperArmAV);
plot(Data.Time,Data.LForearmAV);
plot(Data.Time,Data.ClubhandleAV);

xlabel('Time (s)');
ylabel('Angular Speed (deg/s)');
grid 'on';

%Add Legend to Plot
legend('Hips','Torso','Scap','Left Upper Arm','Left Forearm','Clubhandle');
legend('Location','southeast');

%Add a Title
title('Kinematic Sequence');
subtitle('Data');

%Save Figure
savefig('Data Charts/Plot - Kinematic Sequence');
pause(PauseTime);

%Close Figure
close(924);