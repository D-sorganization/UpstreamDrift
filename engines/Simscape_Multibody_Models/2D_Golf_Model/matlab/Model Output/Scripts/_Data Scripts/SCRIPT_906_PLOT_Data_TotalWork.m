figure(906);
hold on;

plot(Data.Time,Data.TotalLSWork);
plot(Data.Time,Data.TotalRSWork);
plot(Data.Time,Data.TotalLEWork);
plot(Data.Time,Data.TotalREWork);
plot(Data.Time,Data.TotalLWWork);
plot(Data.Time,Data.TotalRWWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LS Total Work','RS Total Work','LE Total Work','RE Total Work','LW Total Work','RW Total Work');
legend('Location','southeast');

%Add a Title
title('Total Work on Distal Segment');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Total Work');
pause(PauseTime);

%Close Figure
close(906);