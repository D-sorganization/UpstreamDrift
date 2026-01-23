figure(938);
hold on;

plot(Data.Time,Data.LHLinearWorkonClub);
plot(Data.Time,Data.LWAngularWorkonClub);
plot(Data.Time,Data.TotalLWWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LW Linear Work','LW Angular Work','LW Total Work');
legend('Location','southeast');

%Add a Title
title('Left Wrist Work on Distal Segment');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Left Wrist Work');
pause(PauseTime);

%Close Figure
close(938);