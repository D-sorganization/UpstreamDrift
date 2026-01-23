figure(122);
plot(BASEQ.Time,BASEQ.ForceAlongHandPath);
xlabel('Time (s)');
ylabel('Force (N)');
grid 'on';

%Add Legend to Plot
legend('Force Along Hand Path');
legend('Location','southeast');

%Add a Title
title('Force Along Hand Path');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Force Along Hand Path');
pause(PauseTime);

%Close Figure
close(122);