figure(325);
hold on;

plot(ZTCFQ.Time,ZTCFQ.LeftHandForceAlongHandPath);
plot(ZTCFQ.Time,ZTCFQ.RightHandForceAlongHandPath);
plot(ZTCFQ.Time,ZTCFQ.ForceAlongHandPath);

ylabel('Force (N)');
grid 'on';

%Add Legend to Plot
legend('LH Force on Left Hand Path','RH Force on Right Hand Path','Net Force Along MP Hand Path');
legend('Location','southeast');

%Add a Title
title('Force Along Hand Path');
subtitle('ZTCF');
%subtitle('Left Hand, Right Hand, Total');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Force Along Hand Path - LH RH Total');
pause(PauseTime);

%Close Figure
close(325);