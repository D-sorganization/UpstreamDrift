figure(711);
hold on;

plot(BASEQ.Time,BASEQ.LHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.RHLinearWorkonClub);

plot(ZTCFQ.Time,ZTCFQ.LHLinearWorkonClub,'--');
plot(ZTCFQ.Time,ZTCFQ.RHLinearWorkonClub,'--');

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LW Linear Work - BASE','RW Linear Work - BASE','LW Linear Work - ZTCF','RW Linear Work - ZTCF');
legend('Location','southeast');

%Add a Title
title('Linear Work on Club');
subtitle('COMPARISON');

%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Linear Work on Club');
pause(PauseTime);

%Close Figure
close(711);