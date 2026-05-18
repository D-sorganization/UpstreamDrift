# EPIC: MacKenzie 2012 Club Position Dynamics

## Objective

Add calculations and features derived from Sasho MacKenzie's 2012 publication ("Club position relative to the golfer's swing plane meaningfully affects swing dynamics") to all of our golf models.

## Background

The relative position of the club center of mass to the swing plane has profound implications on the required kinetics at the grip. To accurately simulate realistic swings, our models must compute and utilize these geometric and kinetic relationships.

## Reference

http://www.sashomackenzie.com/publications/MacKenzie%202012%20Club%20position%20relative%20to%20the%20golfer's%20swing%20plane%20meaningfully%20affects%20swing%20dynamics.pdf

## Key Goals

1.  **Define the Swing Plane**: Implement robust calculation of the instantaneous and average swing planes during the downswing using the clubhead trajectory.
2.  **Calculate Club Position**: Implement algorithms to determine the club's COM position relative to this plane (above, below, or on the plane).
3.  **Kinetic Adjustments**: Modify the applied grip torques/forces in the forward dynamics simulation based on the club's out-of-plane position to match the findings in the paper.
4.  **Validation**: Compare simulation outputs with the empirical torque patterns described in the MacKenzie paper.

## Sub-Tasks

- Implement 3D swing plane fitting.
- Add COM tracking relative to the plane.
- Add metrics for "out-of-plane" distance to the dashboard/results.
