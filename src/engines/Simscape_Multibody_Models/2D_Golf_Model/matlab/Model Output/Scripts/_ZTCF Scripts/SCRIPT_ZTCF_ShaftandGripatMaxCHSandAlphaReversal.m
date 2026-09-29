
%Grip Quiver for ZTCFAlpha Reversal
GripZTCFAlpha=quiver3(ClubQuiverZTCFAlphaReversal.ButtxZTCFAlphaReversal,ClubQuiverZTCFAlphaReversal.ButtyZTCFAlphaReversal,ClubQuiverZTCFAlphaReversal.ButtzZTCFAlphaReversal,ClubQuiverZTCFAlphaReversal.GripdxZTCFAlphaReversal,ClubQuiverZTCFAlphaReversal.GripdyZTCFAlphaReversal,ClubQuiverZTCFAlphaReversal.GripdzZTCFAlphaReversal,0);
GripZTCFAlpha.ShowArrowHead='off';
GripZTCFAlpha.LineWidth=3;			            %Set grip line width
GripZTCFAlpha.Color=[0 0 0];			        %Set grip color to black
hold on;

%Shaft Quiver for Alpha Reversal
ShaftZTCFAlpha=quiver3(ClubQuiverZTCFAlphaReversal.RWxZTCFAlphaReversal,ClubQuiverZTCFAlphaReversal.RWyZTCFAlphaReversal,ClubQuiverZTCFAlphaReversal.RWzZTCFAlphaReversal,ClubQuiverZTCFAlphaReversal.ShaftdxZTCFAlphaReversal,ClubQuiverZTCFAlphaReversal.ShaftdyZTCFAlphaReversal,ClubQuiverZTCFAlphaReversal.ShaftdzZTCFAlphaReversal,0);
ShaftZTCFAlpha.ShowArrowHead='off';		        %Turn off arrow heads
ShaftZTCFAlpha.LineWidth=3;			            %Adjust line weighting
ShaftZTCFAlpha.Color=[0.0745 0.6235 1.0000];
hold on;				                    %Hold the current plot when you generate new

%Grip Quiver for Max CHS
GripMaxCHS=quiver3(ClubQuiverMaxCHS.ButtxMaxCHS,ClubQuiverMaxCHS.ButtyMaxCHS,ClubQuiverMaxCHS.ButtzMaxCHS,ClubQuiverMaxCHS.GripdxMaxCHS,ClubQuiverMaxCHS.GripdyMaxCHS,ClubQuiverMaxCHS.GripdzMaxCHS,0);
GripMaxCHS.ShowArrowHead='off';
GripMaxCHS.LineWidth=3;			            %Set grip line width
GripMaxCHS.Color=[0 0 0];			        %Set grip color to black
hold on;

%Shaft Quiver for Max CHS
ShaftMaxCHS=quiver3(ClubQuiverMaxCHS.RWxMaxCHS,ClubQuiverMaxCHS.RWyMaxCHS,ClubQuiverMaxCHS.RWzMaxCHS,ClubQuiverMaxCHS.ShaftdxMaxCHS,ClubQuiverMaxCHS.ShaftdyMaxCHS,ClubQuiverMaxCHS.ShaftdzMaxCHS,0);
ShaftMaxCHS.ShowArrowHead='off';		    %Turn off arrow heads
ShaftMaxCHS.LineWidth=3;			        %Adjust line weighting
ShaftMaxCHS.Color=[0.9294 0.6941 0.1255];

%Set View
view(-0.0885,-10.6789);