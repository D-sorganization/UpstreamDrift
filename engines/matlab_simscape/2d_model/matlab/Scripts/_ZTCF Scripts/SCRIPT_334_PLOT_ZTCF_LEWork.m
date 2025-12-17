figure(334);
hold on;

plot(ZTCFQ.Time,ZTCFQ.LELinearWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.LEAngularWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.TotalLEWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LE Linear Work','LE Angular Work','LE Total Work');
legend('Location','southeast');

%Add a Title
title('Left Elbow Work on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Left Elbow Work');
pause(PauseTime);

%Close Figure
close(334);