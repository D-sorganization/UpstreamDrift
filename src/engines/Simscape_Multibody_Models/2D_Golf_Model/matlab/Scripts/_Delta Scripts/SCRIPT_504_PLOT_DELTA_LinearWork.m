figure(504);
hold on;

plot(ZTCFQ.Time,DELTAQ.LSLinearWorkonArm);
plot(ZTCFQ.Time,DELTAQ.RSLinearWorkonArm);
plot(ZTCFQ.Time,DELTAQ.LELinearWorkonForearm);
plot(ZTCFQ.Time,DELTAQ.RELinearWorkonForearm);
plot(ZTCFQ.Time,DELTAQ.LHLinearWorkonClub);
plot(ZTCFQ.Time,DELTAQ.RHLinearWorkonClub);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LS Linear Work','RS Linear Work','LE Linear Work','RE Linear Work','LW Linear Work','RW Linear Work');
legend('Location','southeast');

%Add a Title
title('Linear Work on Distal Segment');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Linear Work on Distal');
pause(PauseTime);

%Close Figure
close(504);