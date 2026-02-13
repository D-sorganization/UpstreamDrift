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

import logging
import math
import sys

from src.shared.python.core.constants import GRAVITY_M_S2

logger = logging.getLogger(__name__)

try:
    import opensim as osim
except ImportError:
    logger.error("ERROR: OpenSim Python package not installed.")
    logger.info("Installation: conda install -c opensim-org opensim")
    logger.info("Alternative: pip install opensim (if available)")
    sys.exit(1)


def _create_arm_bodies():
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
    return humerus, radius


def _create_arm_joints(arm, humerus, radius):
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
    return shoulder, elbow


def _create_biceps_muscle(humerus, radius):
    biceps = osim.Millard2012EquilibriumMuscle(
        "biceps",  # name
        200.0,  # max isometric force [N]
        0.6,  # optimal fiber length [m]
        0.55,  # tendon slack length [m]
        0.0,  # pennation angle [rad]
    )
    biceps.addNewPathPoint("origin", humerus, osim.Vec3(0, 0.8, 0))
    biceps.addNewPathPoint("insertion", radius, osim.Vec3(0, 0.7, 0))
    return biceps


def _create_controller(biceps):
    brain = osim.PrescribedController()
    brain.addActuator(biceps)
    brain.prescribeControlForActuator("biceps", osim.StepFunction(0.5, 3.0, 0.3, 1.0))
    return brain


def _add_reporter(arm, biceps, elbow):
    reporter = osim.ConsoleReporter()
    reporter.set_report_time_interval(1.0)
    reporter.addToReport(biceps.getOutput("fiber_force"))
    elbow_coord = elbow.getCoordinate().getOutput("value")
    reporter.addToReport(elbow_coord, "elbow_angle")
    arm.addComponent(reporter)


def _attach_body_visualization(body, name):
    body_geometry = osim.Ellipsoid(0.1, 0.5, 0.1)
    body_geometry.setColor(osim.Gray)

    center = osim.PhysicalOffsetFrame()
    center.setName(name)
    center.setParentFrame(body)
    center.setOffsetTransform(osim.Transform(osim.Vec3(0, 0.5, 0)))
    body.addComponent(center)
    center.attachGeometry(body_geometry.clone())


def build_simple_arm_model() -> osim.Model:
    """Build a simple 2-DOF arm model with a biceps muscle.

    Returns:
        Configured OpenSim Model ready for simulation.
    """
    arm = osim.Model()
    arm.setName("SimpleArm")
    arm.setGravity(osim.Vec3(0, -GRAVITY_M_S2, 0))

    humerus, radius = _create_arm_bodies()
    shoulder, elbow = _create_arm_joints(arm, humerus, radius)
    biceps = _create_biceps_muscle(humerus, radius)
    brain = _create_controller(biceps)

    arm.addBody(humerus)
    arm.addBody(radius)
    arm.addJoint(shoulder)
    arm.addJoint(elbow)
    arm.addForce(biceps)
    arm.addController(brain)

    _add_reporter(arm, biceps, elbow)

    _attach_body_visualization(humerus, "humerusCenter")
    _attach_body_visualization(radius, "radiusCenter")

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

    logger.info(f"\nRunning simulation for {duration} seconds...")
    logger.info("-" * 50)

    final_state = manager.integrate(duration)

    logger.info("-" * 50)
    logger.info("Simulation complete!")

    return final_state


def main() -> None:
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("OpenSim Simple Arm Model Example")
    logger.info("Golf Modeling Suite - Physics Engine Integration")
    logger.info("=" * 60)

    # Build the model
    logger.info("\n1. Building model...")
    arm = build_simple_arm_model()
    logger.info(f"   Model name: {arm.getName()}")
    logger.info(f"   Bodies: {arm.getBodySet().getSize()}")
    logger.info(f"   Joints: {arm.getJointSet().getSize()}")
    logger.info(f"   Muscles: {arm.getMuscles().getSize()}")

    # Run simulation
    logger.info("\n2. Running simulation...")
    final_state = run_simulation(arm, duration=10.0)

    # Save model to file
    output_path = "SimpleArm.osim"
    arm.printToXML(output_path)
    import os

    logger.info(f"\n3. Model saved to: {os.path.abspath(output_path)}")

    # Summary
    elbow = arm.getJointSet().get("elbow")
    final_angle = elbow.getCoordinate().getValue(final_state)
    logger.info(
        f"\n4. Final elbow angle: {final_angle:.3f} rad ({math.degrees(final_angle):.1f}°)"
    )

    logger.info("\n" + "=" * 60)
    logger.info("Example complete! Try loading SimpleArm.osim in OpenSim GUI")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
