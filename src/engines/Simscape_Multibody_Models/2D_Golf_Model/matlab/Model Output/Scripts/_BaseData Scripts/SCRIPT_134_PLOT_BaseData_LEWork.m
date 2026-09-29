figure(134);
hold on;

plot(BASEQ.Time,BASEQ.LELinearWorkonForearm);
plot(BASEQ.Time,BASEQ.LEAngularWorkonForearm);
plot(BASEQ.Time,BASEQ.TotalLEWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LE Linear Work','LE Angular Work','LE Total Work');
legend('Location','southeast');

%Add a Title
title('Left Elbow Work on Distal Segment');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Left Elbow Work');
pause(PauseTime);

%Close Figure
close(134);