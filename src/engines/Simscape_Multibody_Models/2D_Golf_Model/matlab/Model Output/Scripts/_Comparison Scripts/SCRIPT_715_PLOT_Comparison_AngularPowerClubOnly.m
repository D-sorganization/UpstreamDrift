figure(715);
hold on;

plot(BASEQ.Time,BASEQ.LWonClubAngularPower);
plot(BASEQ.Time,BASEQ.RWonClubAngularPower);

plot(ZTCFQ.Time,ZTCFQ.LWonClubAngularPower,'--');
plot(ZTCFQ.Time,ZTCFQ.RWonClubAngularPower,'--');

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LW Angular Power - BASE','RW Angular Power - BASE','LW Angular Power - ZTCF','RW Angular Power - ZTCF');
legend('Location','southeast');

%Add a Title
title('Angular Power on Club');
subtitle('COMPARISON');

%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Angular Power on Club');
pause(PauseTime);

%Close Figure
close(715);