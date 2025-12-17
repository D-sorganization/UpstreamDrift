figure(338);
hold on;

plot(ZTCFQ.Time,ZTCFQ.LHLinearWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.LWAngularWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.TotalLWWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LW Linear Work','LW Angular Work','LW Total Work');
legend('Location','southeast');

%Add a Title
title('Left Wrist Work on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Left Wrist Work');
pause(PauseTime);

%Close Figure
close(338);