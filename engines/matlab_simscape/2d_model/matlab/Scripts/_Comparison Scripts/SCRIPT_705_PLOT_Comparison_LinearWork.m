figure(705);
hold on;

plot(BASEQ.Time,BASEQ.LSLinearWorkonArm);
plot(BASEQ.Time,BASEQ.RSLinearWorkonArm);
plot(BASEQ.Time,BASEQ.LELinearWorkonForearm);
plot(BASEQ.Time,BASEQ.RELinearWorkonForearm);
plot(BASEQ.Time,BASEQ.LHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.RHLinearWorkonClub);

plot(ZTCFQ.Time,ZTCFQ.LSLinearWorkonArm,'--');
plot(ZTCFQ.Time,ZTCFQ.RSLinearWorkonArm,'--');
plot(ZTCFQ.Time,ZTCFQ.LELinearWorkonForearm,'--');
plot(ZTCFQ.Time,ZTCFQ.RELinearWorkonForearm,'--');
plot(ZTCFQ.Time,ZTCFQ.LHLinearWorkonClub,'--');
plot(ZTCFQ.Time,ZTCFQ.RHLinearWorkonClub,'--');
ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LS Linear Work - BASE','RS Linear Work - BASE','LE Linear Work - BASE','RE Linear Work - BASE','LW Linear Work - BASE','RW Linear Work - BASE','LS Linear Work - ZTCF','RS Linear Work - ZTCF','LE Linear Work - ZTCF','RE Linear Work - ZTCF','LW Linear Work - ZTCF','RW Linear Work - ZTCF');
legend('Location','southeast');

%Add a Title
title('Linear Work on Distal Segment');
subtitle('COMPARISON');

%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Linear Work on Distal');
pause(PauseTime);

%Close Figure
close(705);