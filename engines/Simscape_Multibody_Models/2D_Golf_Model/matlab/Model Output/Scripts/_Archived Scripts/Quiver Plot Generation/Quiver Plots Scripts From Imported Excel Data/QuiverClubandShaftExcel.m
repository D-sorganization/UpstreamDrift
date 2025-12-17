%Reads from a Table named Data in main workspace Generated from the Quivers
%Plots tab on the main worksheets. 

%Generate Shaft Quivers
Shaft=quiver3(Data{:,5},Data{:,6},Data{:,7},Data{:,68},Data{:,69},Data{:,70},0);
Shaft.ShowArrowHead='off';		%Turn off arrow heads
Shaft.LineWidth=1;			%Adjust line weighting
Shaft.Color=[0.5 0.5 0.5];		%Set shaft color gray
hold on;				%Hold the current plot when you generate new

%Generate Grip Quivers
Grip= quiver3(Data{:,2},Data{:,3},Data{:,4},Data{:,65},Data{:,66},Data{:,67},0);
Grip.ShowArrowHead='off';
Grip.LineWidth=1;			%Set grip line width
Grip.Color=[0 0 0];			%Set grip color to black

%Add Midpoint Hand Path
HandPath=quiver3(Data{:,8},Data{:,9},Data{:,10},Data{:,74},Data{:,75},Data{:,76},0);
HandPath.Color=[0 0 0];
HandPath.LineWidth=1;
HandPath.ShowArrowHead='off';

