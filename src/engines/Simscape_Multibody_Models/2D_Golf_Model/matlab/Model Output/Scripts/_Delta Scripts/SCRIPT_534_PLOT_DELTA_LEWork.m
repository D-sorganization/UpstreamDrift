figure(534);
hold on;

plot(DELTAQ.Time,DELTAQ.LELinearWorkonForearm);
plot(DELTAQ.Time,DELTAQ.LEAngularWorkonForearm);
plot(DELTAQ.Time,DELTAQ.TotalLEWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LE Linear Work','LE Angular Work','LE Total Work');
legend('Location','southeast');

%Add a Title
title('Left Elbow Work on Distal Segment');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Left Elbow Work');
pause(PauseTime);

%Close Figure
close(534);