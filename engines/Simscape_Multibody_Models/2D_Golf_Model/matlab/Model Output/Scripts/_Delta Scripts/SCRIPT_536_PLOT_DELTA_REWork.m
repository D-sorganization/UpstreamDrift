figure(536);
hold on;

plot(DELTAQ.Time,DELTAQ.RELinearWorkonForearm);
plot(DELTAQ.Time,DELTAQ.REAngularWorkonForearm);
plot(DELTAQ.Time,DELTAQ.TotalREWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('RE Linear Work','RE Angular Work','RE Total Work');
legend('Location','southeast');

%Add a Title
title('Right Elbow Work on Distal Segment');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Right Elbow Work');
pause(PauseTime);

%Close Figure
close(536);