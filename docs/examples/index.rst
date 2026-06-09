Examples
========

Runnable, dependency-light end-to-end examples. Each script is self-contained
and can be executed directly from the repository root::

    python docs/examples/run_mock_engine_sim.py
    python docs/examples/motion_matching_synthetic.py
    python docs/examples/estimate_kinematics.py

They are covered by a smoke test
(``tests/unit/docs/test_examples_runnable.py``) so they cannot silently rot.

Load an engine and run a simulation
-----------------------------------

Loads the dependency-light :class:`MockPhysicsEngine` (same protocol as the
real MuJoCo / Drake / Pinocchio backends) and integrates a short swing.

.. literalinclude:: run_mock_engine_sim.py
   :language: python
   :linenos:

Motion matching on a synthetic trajectory
------------------------------------------

Scores a candidate joint trajectory against a reference using the same
pose-error + velocity-error decomposition as the Drake motion-matching
pipeline, with ``numpy`` only.

.. literalinclude:: motion_matching_synthetic.py
   :language: python
   :linenos:

Estimate velocity and acceleration (calc/estimation)
----------------------------------------------------

Recovers joint velocities and accelerations from a sampled position signal via
central finite differences and validates against a known analytic signal.

.. literalinclude:: estimate_kinematics.py
   :language: python
   :linenos:

Related guides
--------------

- :doc:`../user_guide/getting_started`
- :doc:`../motion_pipeline/README`
- :doc:`../engines/engine_selection_guide`
