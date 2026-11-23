# PR #15559 evidence: Ball flight and green tab

Both images show the real `BallFlightWidget` rendered with the Windows Qt platform plugin.

**The trajectory is SYNTHETIC.** It is a sine arc with a linear rollout, built to exercise layout.
It is not a model result and says nothing about carry accuracy. The widget stamps
"Uncalibrated launch; flat 10 ft stimp green, simplified bounce and spin" on the plot.

- `flight-1280.png`: 1280x900
- `flight-390x844.png`: 390x844 (phone width)

Captured from PR head after the layout fix: y-axis labels fully visible, wrapped banner clear of the legend.
