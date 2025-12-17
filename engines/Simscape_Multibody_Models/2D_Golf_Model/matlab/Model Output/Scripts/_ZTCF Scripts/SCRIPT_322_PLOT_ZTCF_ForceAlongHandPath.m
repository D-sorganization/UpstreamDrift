figure(322);
plot(ZTCFQ.Time,ZTCFQ.ForceAlongHandPath);
xlabel('Time (s)');
ylabel('Force (N)');
grid 'on';

%Add Legend to Plot
legend('Force Along Hand Path');
legend('Location','southeast');

%Add a Title
title('Force Along Hand Path');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Force Along Hand Path');
pause(PauseTime);

%Close Figure
close(322);