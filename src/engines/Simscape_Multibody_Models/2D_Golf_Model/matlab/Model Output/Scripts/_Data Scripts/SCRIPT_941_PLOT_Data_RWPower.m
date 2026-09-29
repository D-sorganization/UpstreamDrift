figure(941);
hold on;

plot(Data.Time,Data.RWonClubLinearPower);
plot(Data.Time,Data.RWonClubAngularPower);
plot(Data.Time,Data.TotalRWPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('RW Linear Power','RW Angular Power','RW Total Power');
legend('Location','southeast');

%Add a Title
title('Right Wrist Power on Distal Segment');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Right Wrist Power');
pause(PauseTime);

%Close Figure
close(941);