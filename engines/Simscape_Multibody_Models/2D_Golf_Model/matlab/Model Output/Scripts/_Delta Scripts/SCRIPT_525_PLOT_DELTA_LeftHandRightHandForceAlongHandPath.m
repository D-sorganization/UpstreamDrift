figure(525);
hold on;

plot(ZTCFQ.Time,DELTAQ.LeftHandForceAlongHandPath);
plot(ZTCFQ.Time,DELTAQ.RightHandForceAlongHandPath);
plot(ZTCFQ.Time,DELTAQ.ForceAlongHandPath);

ylabel('Force (N)');
grid 'on';

%Add Legend to Plot
legend('LH Force on Left Hand Path','RH Force on Right Hand Path','Net Force Along MP Hand Path');
legend('Location','southeast');

%Add a Title
title('Force Along Hand Path');
subtitle('DELTA');
%subtitle('Left Hand, Right Hand, Total');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Force Along Hand Path - LH RH Total');
pause(PauseTime);

%Close Figure
close(525);