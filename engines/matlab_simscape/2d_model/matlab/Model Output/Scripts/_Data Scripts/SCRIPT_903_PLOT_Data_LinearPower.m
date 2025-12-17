figure(903);
hold on;

plot(Data.Time,Data.LSonArmLinearPower);
plot(Data.Time,Data.RSonArmLinearPower);
plot(Data.Time,Data.LEonForearmLinearPower);
plot(Data.Time,Data.REonForearmLinearPower);
plot(Data.Time,Data.LWonClubLinearPower);
plot(Data.Time,Data.RWonClubLinearPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LS Linear Power','RS Linear Power','LE Linear Power','RE Linear Power','LW Linear Power','RW Linear Power');
legend('Location','southeast');

%Add a Title
title('Linear Power on Distal Segment');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Linear Power');
pause(PauseTime);

%Close Figure
close(903);