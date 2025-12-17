figure(326);
plot(ZTCFQ.Time,ZTCFQ.LinearImpulseonClub);
xlabel('Time (s)');
ylabel('Impulse (Ns)');
grid 'on';

%Add Legend to Plot
legend('Linear Impulse');
legend('Location','southeast');

%Add a Title
title('Linear Impulse');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Linear Impulse');
pause(PauseTime);

%Close Figure
close(326);