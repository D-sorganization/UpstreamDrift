%Reads from a Table named ZTCF in main workspace Generated from the Quivers
%Plots tab on the main worksheets. 

%Generate Shaft Quivers
Shaft=quiver3(ZTCFQ.RWx(:,1),ZTCFQ.RWy(:,1),ZTCFQ.RWz(:,1),ZTCFQ.Shaftdx(:,1),ZTCFQ.Shaftdy(:,1),ZTCFQ.Shaftdz(:,1),0);
Shaft.ShowArrowHead='off';		%Turn off arrow heads
Shaft.LineWidth=1;			    %Adjust line weighting
Shaft.Color=[0.5 0.5 0.5];		%Set shaft color gray
hold on;				        %Hold the current plot when you generate new

%Generate Grip Quivers
Grip=quiver3(ZTCFQ.Buttx(:,1),ZTCFQ.Butty(:,1),ZTCFQ.Buttz(:,1),ZTCFQ.Gripdx(:,1),ZTCFQ.Gripdy(:,1),ZTCFQ.Gripdz(:,1),0);
Grip.ShowArrowHead='off';
Grip.LineWidth=1;			    %Set grip line width
Grip.Color=[0 0 0];			    %Set grip color to black

%Calculate height of table
H=height(ZTCFQ);

%Calculate how many rows to copy
h=H-1;

%Generate Hand Path Quivers
HandPath=quiver3(ZTCFQ.MPx(1:h,1),ZTCFQ.MPy(1:h,1),ZTCFQ.MPz(1:h,1),ZTCFQ.MPPx(1:h,1),ZTCFQ.MPPy(1:h,1),ZTCFQ.MPPz(1:h,1),0);
HandPath.ShowArrowHead='off';		%Turn off arrow heads
HandPath.LineWidth=1;			    %Adjust line weighting
HandPath.Color=[0 0 0];		        %Set shaft color black

%Generate Club Path Quivers
ClubPath=quiver3(ZTCFQ.CHx(1:h,1),ZTCFQ.CHy(1:h,1),ZTCFQ.CHz(1:h,1),ZTCFQ.CHPx(1:h,1),ZTCFQ.CHPy(1:h,1),ZTCFQ.CHPz(1:h,1),0);
ClubPath.ShowArrowHead='off';		%Turn off arrow heads
ClubPath.LineWidth=1;			    %Adjust line weighting
ClubPath.Color=[0 0 0];		        %Set shaft color black

% clear Grip;
% clear Shaft;
% clear HandPath;
% clear ClubPath;
clear h;
clear H;