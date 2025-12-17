figure(936);
hold on;

plot(Data.Time,Data.RELinearWorkonForearm);
plot(Data.Time,Data.REAngularWorkonForearm);
plot(Data.Time,Data.TotalREWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('RE Linear Work','RE Angular Work','RE Total Work');
legend('Location','southeast');

%Add a Title
title('Right Elbow Work on Distal Segment');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Right Elbow Work');
pause(PauseTime);

%Close Figure
close(936);