figure(332);
hold on;

plot(ZTCFQ.Time,ZTCFQ.RSLinearWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.RSAngularWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.TotalRSWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('RS Linear Work','RS Angular Work','RS Total Work');
legend('Location','southeast');

%Add a Title
title('Right Shoulder Work on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Right Shoulder Work');
pause(PauseTime);

%Close Figure
close(332);