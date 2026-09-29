figure(940);
hold on;

plot(Data.Time,Data.RHLinearWorkonClub);
plot(Data.Time,Data.RWAngularWorkonClub);
plot(Data.Time,Data.TotalRWWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('RW Linear Work','RW Angular Work','RW Total Work');
legend('Location','southeast');

%Add a Title
title('Right Wrist Work on Distal Segment');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Right Wrist Work');
pause(PauseTime);

%Close Figure
close(940);