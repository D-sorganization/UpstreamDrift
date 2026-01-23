figure(526);
plot(ZTCFQ.Time,DELTAQ.LinearImpulseonClub);
xlabel('Time (s)');
ylabel('Impulse (Ns)');
grid 'on';

%Add Legend to Plot
legend('Linear Impulse');
legend('Location','southeast');

%Add a Title
title('Linear Impulse');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Linear Impulse');
pause(PauseTime);

%Close Figure
close(526);