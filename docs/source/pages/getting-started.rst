Getting Started
===============

Pre-requisites
--------------

Before building and using SlicerSOFA you should be familiar with:

- Linux, CMake and building C++ projects.
- Building Slicer from source (required to compile the C++ extensions).
- Basic SOFA knowledge (scene files, solvers, meshes) is helpful to author or modify SOFA scenes.

Recommended environment
-----------------------

- Ubuntu LTS (e.g. 20.04, 22.04 depending on your Slicer build).
- Slicer built from source (see Slicer developer guide).
- System Qt (install with apt; do not install a conflicting Qt from upstream unless you know how to adapt the build).
- Required libraries and tools are orchestrated by the included SuperBuild (see pages/superbuild).

Cloning
-------

Assuming a development workspace:

.. code-block:: bash

   mkdir -p ~/dev
   cd ~/dev
   git clone https://github.com/yourorg/SlicerSOFA.git
   cd SlicerSOFA

Building overview
-----------------

There are two main approaches to build SlicerSOFA depending on how you want to manage external dependencies:

1. Use the included SuperBuild to fetch and build SOFA and other third-party libraries. This is useful if you do not already have SOFA available.
2. If you already have a Slicer build and external dependencies (SOFA and libraries) installed on your system, build only the Slicer extension and point CMake to your Slicer build directory.

Basic build example (using SuperBuild)
-------------------------------------

.. code-block:: bash

   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DSlicer_DIR:PATH=/path/to/your/Slicer-build
   make -j4

Alternatively, if building the Slicer module only:

.. code-block:: bash

   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DSlicer_DIR:PATH=/path/to/your/Slicer-build -DUSE_SUPERBUILD=OFF
   make -j4

Loading the module into Slicer
------------------------------

After a successful build:

- Ensure Slicer can find SlicerSOFA binaries and any required dynamic libraries. If you built against a Slicer build tree and used the build install steps, the module will be placed into the Slicer extensions path in the build area.

- Start Slicer from the same shell where environment variables are correct (if you built/install in custom locations set LD_LIBRARY_PATH accordingly). Example:

.. code-block:: bash

   cd /path/to/Slicer-build
   ./Slicer

Inside Slicer the SlicerSOFA module(s) should appear under the Modules menu (names: SlicerSofa, SoftTissueSimulation, SparseGridSimulation).

Quick start with Python
-----------------------

Once the module is loaded you can control simulations from Slicer's Python interactor. Example (pseudo-API; see integration docs for real API):

.. code-block:: python

   import slicer
   logic = slicer.util.getModuleLogic('SlicerSofa')
   # Load a scene file (path relative to your project or absolute)
   sofaSceneNode = logic.LoadSofaScene('/path/to/scene.scn')
   # Start simulation (run on a timer integrated with Slicer UI)
   sofaSceneNode.Play()
   # Pause and step
   sofaSceneNode.Pause()
   sofaSceneNode.Step()

See pages/integration.rst for the API details and examples.
