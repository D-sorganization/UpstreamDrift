figure(16);
hold on;

plot(BASEQ.Time,BASEQ.LHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.RHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.LinearWorkonClub);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LH Linear Work','RH Linear Work','Net Force Linear Work (midpoint)');
legend('Location','southeast');

%Add a Title
title('Linear Work');
subtitle('COMPARISON');
%subtitle('Left Hand, Right Hand, Total');

%Save Figure
savefig('Comparison Charts/COMPARISON_Plot - Linear Work');

%Close Figure
close(16);