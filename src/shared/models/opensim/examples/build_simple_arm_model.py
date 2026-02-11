"""
OpenSim Simple Arm Model Example
================================

This example demonstrates how to build and simulate a simple arm model
using the OpenSim Python API.

Source: opensim-org/opensim-core (Bindings/Python/examples)
License: Apache 2.0

The model consists of:
- 2 bodies (humerus, radius)
- 2 pin joints (shoulder, elbow)
- 1 muscle (biceps) with Millard2012 equilibrium model
- 1 controller (PrescribedController with step function)

Usage:
    python build_simple_arm_model.py

Output:
    - Console output showing muscle fiber force and elbow angle
    - SimpleArm.osim file

Requirements:
    - OpenSim Python package: conda install -c opensim-org opensim
"""

import math
import sys

from src.shared.python.constants import GRAVITY_M_S2

try:
    import opensim as osim
except ImportError:
    print("ERROR: OpenSim Python package not installed.")
    print("Installation: conda install -c opensim-org opensim")
    print("Alternative: pip install opensim (if available)")
    sys.exit(1)


def build_simple_arm_model() -> osim.Model:
    """Build a simple 2-DOF arm model with a biceps muscle.

    Returns:
        Configured OpenSim Model ready for simulation.
    """
    # Define global model where the arm lives.
    arm = osim.Model()
    arm.setName("SimpleArm")

    # Gravity - use centralized constant for DRY
    arm.setGravity(osim.Vec3(0, -GRAVITY_M_S2, 0))

    # ---------------------------------------------------------------------------
    # Create bodies
    # ---------------------------------------------------------------------------
    humerus = osim.Body(
        "humerus",
        1.0,  # mass [kg]
        osim.Vec3(0, 0, 0),  # center of mass
        osim.Inertia(0.1, 0.1, 0.1),  # inertia
    )
    radius = osim.Body(
        "radius",
        1.0,  # mass [kg]
        osim.Vec3(0, 0, 0),
        osim.Inertia(0.1, 0.1, 0.1),
    )

    # ---------------------------------------------------------------------------
    # Create joints
    # ---------------------------------------------------------------------------
    shoulder = osim.PinJoint(
        "shoulder",
        arm.getGround(),  # parent frame
        osim.Vec3(0, 0, 0),  # location in parent
        osim.Vec3(0, 0, 0),  # orientation in parent
        humerus,  # child frame
        osim.Vec3(0, 1, 0),  # location in child (1m arm length)
        osim.Vec3(0, 0, 0),  # orientation in child
    )

    elbow = osim.PinJoint(
        "elbow",
        humerus,  # parent frame
        osim.Vec3(0, 0, 0),
        osim.Vec3(0, 0, 0),
        radius,  # child frame
        osim.Vec3(0, 1, 0),  # location in child
        osim.Vec3(0, 0, 0),
    )

    # ---------------------------------------------------------------------------
    # Create muscle (Millard2012 Hill-type model)
    # ---------------------------------------------------------------------------
    biceps = osim.Millard2012EquilibriumMuscle(
        "biceps",  # name
        200.0,  # max isometric force [N]
        0.6,  # optimal fiber length [m]
        0.55,  # tendon slack length [m]
        0.0,  # pennation angle [rad]
    )
    biceps.addNewPathPoint("origin", humerus, osim.Vec3(0, 0.8, 0))
    biceps.addNewPathPoint("insertion", radius, osim.Vec3(0, 0.7, 0))

    # ---------------------------------------------------------------------------
    # Create controller (step activation: 0.3 at t=0.5s, 1.0 at t=3.0s)
    # ---------------------------------------------------------------------------
    brain = osim.PrescribedController()
    brain.addActuator(biceps)
    brain.prescribeControlForActuator("biceps", osim.StepFunction(0.5, 3.0, 0.3, 1.0))

    # ---------------------------------------------------------------------------
    # Assemble model
    # ---------------------------------------------------------------------------
    arm.addBody(humerus)
    arm.addBody(radius)
    arm.addJoint(shoulder)
    arm.addJoint(elbow)
    arm.addForce(biceps)
    arm.addController(brain)

    # ---------------------------------------------------------------------------
    # Add reporter for console output
    # ---------------------------------------------------------------------------
    reporter = osim.ConsoleReporter()
    reporter.set_report_time_interval(1.0)
    reporter.addToReport(biceps.getOutput("fiber_force"))
    elbow_coord = elbow.getCoordinate().getOutput("value")
    reporter.addToReport(elbow_coord, "elbow_angle")
    arm.addComponent(reporter)

    # ---------------------------------------------------------------------------
    # Add visualization geometry
    # ---------------------------------------------------------------------------
    body_geometry = osim.Ellipsoid(0.1, 0.5, 0.1)
    body_geometry.setColor(osim.Gray)

    # Humerus visualization
    humerus_center = osim.PhysicalOffsetFrame()
    humerus_center.setName("humerusCenter")
    humerus_center.setParentFrame(humerus)
    humerus_center.setOffsetTransform(osim.Transform(osim.Vec3(0, 0.5, 0)))
    humerus.addComponent(humerus_center)
    humerus_center.attachGeometry(body_geometry.clone())

    # Radius visualization
    radius_center = osim.PhysicalOffsetFrame()
    radius_center.setName("radiusCenter")
    radius_center.setParentFrame(radius)
    radius_center.setOffsetTransform(osim.Transform(osim.Vec3(0, 0.5, 0)))
    radius.addComponent(radius_center)
    radius_center.attachGeometry(body_geometry.clone())

    return arm


def run_simulation(model: osim.Model, duration: float = 10.0) -> osim.State:
    """Run forward dynamics simulation.

    Args:
        model: Initialized OpenSim model.
        duration: Simulation duration in seconds.

    Returns:
        Final state after simulation.
    """
    # Initialize the system
    state = model.initSystem()

    # Get joint coordinates
    shoulder = model.getJointSet().get("shoulder")
    elbow = model.getJointSet().get("elbow")

    # Fix the shoulder at its default angle
    shoulder.getCoordinate().setLocked(state, True)

    # Begin with the elbow flexed (90 degrees)
    elbow.getCoordinate().setValue(state, 0.5 * osim.SimTK_PI)

    # Equilibrate muscles at initial pose
    model.equilibrateMuscles(state)

    # Create and run manager
    manager = osim.Manager(model)
    state.setTime(0)
    manager.initialize(state)

    print(f"\nRunning simulation for {duration} seconds...")
    print("-" * 50)

    final_state = manager.integrate(duration)

    print("-" * 50)
    print("Simulation complete!")

    return final_state


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("OpenSim Simple Arm Model Example")
    print("Golf Modeling Suite - Physics Engine Integration")
    print("=" * 60)

    # Build the model
    print("\n1. Building model...")
    arm = build_simple_arm_model()
    print(f"   Model name: {arm.getName()}")
    print(f"   Bodies: {arm.getBodySet().getSize()}")
    print(f"   Joints: {arm.getJointSet().getSize()}")
    print(f"   Muscles: {arm.getMuscles().getSize()}")

    # Run simulation
    print("\n2. Running simulation...")
    final_state = run_simulation(arm, duration=10.0)

    # Save model to file
    output_path = "SimpleArm.osim"
    arm.printToXML(output_path)
    import os

    print(f"\n3. Model saved to: {os.path.abspath(output_path)}")

    # Summary
    elbow = arm.getJointSet().get("elbow")
    final_angle = elbow.getCoordinate().getValue(final_state)
    print(
        f"\n4. Final elbow angle: {final_angle:.3f} rad ({math.degrees(final_angle):.1f}°)"
    )

    print("\n" + "=" * 60)
    print("Example complete! Try loading SimpleArm.osim in OpenSim GUI")
    print("=" * 60)


if __name__ == "__main__":
    main()
