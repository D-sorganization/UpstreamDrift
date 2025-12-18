Installation Guide
==================

The Golf Modeling Suite supports multiple installation methods depending on your needs.

Quick Installation
------------------

For basic functionality with MuJoCo engine:

.. code-block:: bash

   pip install golf-modeling-suite

Development Installation
------------------------

For development or to access all engines:

.. code-block:: bash

   git clone https://github.com/your-org/golf-modeling-suite.git
   cd golf-modeling-suite
   pip install -e .[dev,all]

Engine-Specific Installation
----------------------------

Core Engines (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Basic installation with MuJoCo
   pip install golf-modeling-suite

   # Add advanced engines
   pip install golf-modeling-suite[engines]

Advanced Engines
~~~~~~~~~~~~~~~~

For Drake and Pinocchio engines, additional setup may be required:

.. code-block:: bash

   # Using conda (recommended for Drake/Pinocchio)
   conda create -n golf-suite python=3.11
   conda activate golf-suite
   conda install -c conda-forge pinocchio drake
   pip install golf-modeling-suite

MATLAB Integration
------------------

For MATLAB Simscape engine support:

1. Install MATLAB R2021b or later
2. Install MATLAB Engine for Python:

.. code-block:: bash

   cd "matlabroot/extern/engines/python"
   python setup.py install

3. Verify installation:

.. code-block:: python

   import matlab.engine
   eng = matlab.engine.start_matlab()

System Requirements
-------------------

Minimum Requirements
~~~~~~~~~~~~~~~~~~~~

* Python 3.11+
* 8 GB RAM
* 2 GB disk space
* OpenGL 3.3+ support

Recommended Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

* Python 3.11
* 16 GB RAM
* 10 GB disk space
* Dedicated GPU (for advanced visualization)
* MATLAB R2021b+ (for Simscape engine)

Verification
------------

Test your installation:

.. code-block:: python

   from golf_modeling_suite import validate_installation
   validate_installation()

Or run the validation script:

.. code-block:: bash

   python -m golf_modeling_suite.validate

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import errors with physics engines:**

.. code-block:: bash

   # For Drake/Pinocchio issues, use conda
   conda install -c conda-forge pinocchio drake

**MATLAB engine not found:**

.. code-block:: bash

   # Reinstall MATLAB Engine for Python
   cd "matlabroot/extern/engines/python"
   python setup.py install --force

**OpenGL issues:**

.. code-block:: bash

   # Install OpenGL libraries (Ubuntu/Debian)
   sudo apt-get install libgl1-mesa-glx libglu1-mesa

Getting Help
~~~~~~~~~~~~

* Check the :doc:`user_guide/troubleshooting` section
* Open an issue on `GitHub <https://github.com/your-org/golf-modeling-suite/issues>`_
* Join our community discussions