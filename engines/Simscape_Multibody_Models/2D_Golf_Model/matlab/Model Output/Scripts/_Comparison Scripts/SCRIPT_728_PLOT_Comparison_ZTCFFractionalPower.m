figure(728);
hold on;

plot(BASEQ.Time,BASEQ.ZTCFQLSFractionalPower);
plot(BASEQ.Time,BASEQ.ZTCFQRSFractionalPower);
plot(BASEQ.Time,BASEQ.ZTCFQLEFractionalPower);
plot(BASEQ.Time,BASEQ.ZTCFQREFractionalPower);
plot(BASEQ.Time,BASEQ.ZTCFQLWFractionalPower);
plot(BASEQ.Time,BASEQ.ZTCFQRWFractionalPower);
ylim([-5 5]);
ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LS Total ZTCF Fractional Power','RS Total ZTCF Fractional Power','LE Total ZTCF Fractional Power','RE Total ZTCF Fractional Power','LW Total ZTCF Fractional Power','RW Total ZTCF Fractional Power');
legend('Location','southeast');

%Add a Title
title('Total ZTCF Fractional Power on Distal Segment');
subtitle('COMPARISON');

%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Total ZTCF Fractional Power');
pause(PauseTime);

%Close Figure
close(728);