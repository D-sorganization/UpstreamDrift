figure(902);
hold on;

plot(Data.Time,Data.LSonArmAngularPower);
plot(Data.Time,Data.RSonArmAngularPower);
plot(Data.Time,Data.LEonForearmAngularPower);
plot(Data.Time,Data.REonForearmAngularPower);
plot(Data.Time,Data.LWonClubAngularPower);
plot(Data.Time,Data.RWonClubAngularPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LS Angular Power','RS Angular Power','LE Angular Power','RE Angular Power','LW Angular Power','RW Angular Power');
legend('Location','southeast');

%Add a Title
title('Angular Power on Distal Segment');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Angular Power');
pause(PauseTime);

%Close Figure
close(902);