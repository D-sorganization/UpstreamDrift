figure(907);
hold on;

plot(Data.Time,Data.TotalLSPower);
plot(Data.Time,Data.TotalRSPower);
plot(Data.Time,Data.TotalLEPower);
plot(Data.Time,Data.TotalREPower);
plot(Data.Time,Data.TotalLWPower);
plot(Data.Time,Data.TotalRWPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LS Total Power','RS Total Power','LE Total Power','RE Total Power','LW Total Power','RW Total Power');
legend('Location','southeast');

%Add a Title
title('Total Power on Distal Segment');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Total Power');
pause(PauseTime);

%Close Figure
close(907);