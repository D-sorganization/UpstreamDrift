Quick Start Guide
=================

Get up and running with your first golf swing simulation in minutes.

Launch the Suite
----------------

Start the Golf Modeling Suite GUI:

.. code-block:: bash

   golf-suite

Or using Python:

.. code-block:: python

   from golf_modeling_suite import launch_gui
   launch_gui()

Your First Simulation
---------------------

1. **Select Engine**: Choose MuJoCo for your first simulation
2. **Load Model**: Use the default golf swing model
3. **Set Parameters**: 
   - Swing speed: 100 mph
   - Club: Driver
   - Ball position: Standard
4. **Run Simulation**: Click "Simulate"
5. **View Results**: Analyze trajectory and biomechanics

Basic Python API
-----------------

For programmatic access:

.. code-block:: python

   from golf_modeling_suite.engines.mujoco import MuJoCoGolfModel
   
   # Create model
   model = MuJoCoGolfModel()
   
   # Set swing parameters
   model.set_swing_speed(100)  # mph
   model.set_club_type("driver")
   
   # Run simulation
   results = model.simulate()
   
   # Analyze results
   print(f"Ball distance: {results.ball_distance:.1f} yards")
   print(f"Launch angle: {results.launch_angle:.1f} degrees")
   
   # Visualize
   model.visualize(results)

Example Workflows
-----------------

Swing Optimization
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from golf_modeling_suite.optimization import SwingOptimizer
   
   optimizer = SwingOptimizer(engine="mujoco")
   optimal_params = optimizer.optimize_for_distance()
   print(f"Optimal swing speed: {optimal_params.speed} mph")

Motion Capture Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from golf_modeling_suite.analysis import MotionCaptureAnalyzer
   
   analyzer = MotionCaptureAnalyzer()
   data = analyzer.load_c3d("swing_data.c3d")
   biomechanics = analyzer.analyze_swing(data)

Multi-Engine Comparison
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from golf_modeling_suite.comparison import EngineComparison
   
   comparison = EngineComparison(["mujoco", "drake", "pinocchio"])
   results = comparison.run_swing_analysis()
   comparison.plot_comparison(results)

Next Steps
----------

* Explore the :doc:`user_guide/index` for detailed tutorials
* Check out :doc:`examples/index` for more complex workflows
* Learn about each :doc:`engines/index` and their strengths
* Join the community for tips and best practices

Key Features to Explore
-----------------------

* **Advanced Biomechanics**: 290-muscle musculoskeletal modeling
* **Trajectory Optimization**: Find optimal swing parameters
* **Motion Capture Integration**: Analyze real swing data
* **Multi-Engine Validation**: Compare results across physics engines
* **Professional Export**: Generate reports and videos