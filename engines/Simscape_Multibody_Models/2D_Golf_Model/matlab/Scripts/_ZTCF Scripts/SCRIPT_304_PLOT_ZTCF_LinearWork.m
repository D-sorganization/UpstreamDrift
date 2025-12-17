figure(304);
hold on;

plot(ZTCFQ.Time,ZTCFQ.LSLinearWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.RSLinearWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.LELinearWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.RELinearWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.LHLinearWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.RHLinearWorkonClub);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LS Linear Work','RS Linear Work','LE Linear Work','RE Linear Work','LW Linear Work','RW Linear Work');
legend('Location','southeast');

%Add a Title
title('Linear Work on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Linear Work on Distal');
pause(PauseTime);

%Close Figure
close(304);