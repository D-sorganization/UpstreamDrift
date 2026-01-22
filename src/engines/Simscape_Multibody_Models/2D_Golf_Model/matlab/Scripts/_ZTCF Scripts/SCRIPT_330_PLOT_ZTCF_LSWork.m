figure(330);
hold on;

plot(ZTCFQ.Time,ZTCFQ.LSLinearWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.LSAngularWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.TotalLSWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LS Linear Work','LS Angular Work','LS Total Work');
legend('Location','southeast');

%Add a Title
title('Left Shoulder Work on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Left Shoulder Work');
pause(PauseTime);

%Close Figure
close(330);