figure(126);
plot(BASEQ.Time,BASEQ.LinearImpulseonClub);
xlabel('Time (s)');
ylabel('Impulse (Ns)');
grid 'on';

%Add Legend to Plot
legend('Linear Impulse');
legend('Location','southeast');

%Add a Title
title('Linear Impulse');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Linear Impulse');
pause(PauseTime);

%Close Figure
close(126);