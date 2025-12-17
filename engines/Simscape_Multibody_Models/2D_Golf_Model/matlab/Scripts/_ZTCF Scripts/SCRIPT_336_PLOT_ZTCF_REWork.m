figure(336);
hold on;

plot(ZTCFQ.Time,ZTCFQ.RELinearWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.REAngularWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.TotalREWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('RE Linear Work','RE Angular Work','RE Total Work');
legend('Location','southeast');

%Add a Title
title('Right Elbow Work on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Right Elbow Work');
pause(PauseTime);

%Close Figure
close(336);