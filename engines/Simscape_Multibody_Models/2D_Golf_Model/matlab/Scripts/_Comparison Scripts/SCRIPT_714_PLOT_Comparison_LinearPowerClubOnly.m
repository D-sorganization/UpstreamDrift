figure(714);
hold on;

plot(BASEQ.Time,BASEQ.LWonClubLinearPower);
plot(BASEQ.Time,BASEQ.RWonClubLinearPower);

plot(ZTCFQ.Time,ZTCFQ.LWonClubLinearPower,'--');
plot(ZTCFQ.Time,ZTCFQ.RWonClubLinearPower,'--');


ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LW Linear Power - BASE','RW Linear Power - BASE','LW Linear Power - ZTCF','RW Linear Power - ZTCF');
legend('Location','southeast');

%Add a Title
title('Linear Power on Club');
subtitle('COMPARISON');

%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Linear Power on Club');
pause(PauseTime);

%Close Figure
close(714);