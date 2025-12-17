figure(125);
hold on;

plot(BASEQ.Time,BASEQ.LeftHandForceAlongHandPath);
plot(BASEQ.Time,BASEQ.RightHandForceAlongHandPath);
plot(BASEQ.Time,BASEQ.ForceAlongHandPath);

ylabel('Force (N)');
grid 'on';

%Add Legend to Plot
legend('LH Force on Left Hand Path','RH Force on Right Hand Path','Net Force Along MP Hand Path');
legend('Location','southeast');

%Add a Title
title('Force Along Hand Path');
%subtitle('Left Hand, Right Hand, Total');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Force Along Hand Path - LH RH Total');
pause(PauseTime);

%Close Figure
close(125);