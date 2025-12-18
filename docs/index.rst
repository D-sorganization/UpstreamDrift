Golf Modeling Suite Documentation
==================================

Welcome to the Golf Modeling Suite, a comprehensive platform for golf swing analysis and simulation using multiple physics engines.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide/index
   api/index
   engines/index
   examples/index
   development/index

Overview
--------

The Golf Modeling Suite provides:

* **Multiple Physics Engines**: MuJoCo, Drake, Pinocchio, and MATLAB Simscape
* **Advanced Biomechanics**: 290-muscle musculoskeletal modeling
* **Motion Capture Integration**: C3D, CSV, and JSON format support
* **Trajectory Optimization**: Advanced swing optimization algorithms
* **Professional Analysis**: Video export, ML-based analysis, performance metrics

Quick Links
-----------

* :doc:`installation` - Get started with installation
* :doc:`quickstart` - Your first golf swing simulation
* :doc:`user_guide/index` - Comprehensive user guide
* :doc:`api/index` - API reference
* :doc:`examples/index` - Example notebooks and scripts

Engines
-------

.. grid:: 2

   .. grid-item-card:: MuJoCo Engine
      :link: engines/mujoco
      :link-type: doc

      Advanced physics simulation with musculoskeletal modeling

   .. grid-item-card:: Drake Engine
      :link: engines/drake
      :link-type: doc

      Trajectory optimization and control

   .. grid-item-card:: Pinocchio Engine
      :link: engines/pinocchio
      :link-type: doc

      Fast analytical dynamics and kinematics

   .. grid-item-card:: MATLAB Simscape
      :link: engines/matlab
      :link-type: doc

      Biomechanical modeling and analysis

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`