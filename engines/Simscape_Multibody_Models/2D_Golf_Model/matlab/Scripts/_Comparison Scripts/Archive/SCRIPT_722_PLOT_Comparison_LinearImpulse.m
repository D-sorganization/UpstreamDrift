figure(15);
plot(BASEQ.Time,BASEQ.LinearImpulseonClub);
xlabel('Time (s)');
ylabel('Impulse (Ns)');
grid 'on';

%Add Legend to Plot
legend('Linear Impulse');
legend('Location','southeast');

%Add a Title
title('Linear Impulse');
subtitle('COMPARISON');

%Save Figure
savefig('Comparison Charts/COMPARISON_Plot - Linear Impulse');

%Close Figure
close(15);