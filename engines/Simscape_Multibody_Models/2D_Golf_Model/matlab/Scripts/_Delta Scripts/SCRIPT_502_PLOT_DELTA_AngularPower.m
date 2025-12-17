figure(502);
hold on;

plot(ZTCFQ.Time,DELTAQ.LSonArmAngularPower);
plot(ZTCFQ.Time,DELTAQ.RSonArmAngularPower);
plot(ZTCFQ.Time,DELTAQ.LEonForearmAngularPower);
plot(ZTCFQ.Time,DELTAQ.REonForearmAngularPower);
plot(ZTCFQ.Time,DELTAQ.LWonClubAngularPower);
plot(ZTCFQ.Time,DELTAQ.RWonClubAngularPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LS Angular Power','RS Angular Power','LE Angular Power','RE Angular Power','LW Angular Power','RW Angular Power');
legend('Location','southeast');

%Add a Title
title('Angular Power on Distal Segment');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Angular Power');
pause(PauseTime);

%Close Figure
close(502);