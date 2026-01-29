SuperBuild and dependencies
===========================

Purpose
-------

The SuperBuild in this repository automates fetching and building external dependencies required by SlicerSOFA and the included example modules. External components may include:

- SOFA
- Boost
- Eigen3
- GLEW
- pybind11
- tinyxml2

Configuration
-------------

The SuperBuild contains CMake scripts (e.g. External_Boost.cmake, External_Sofa.cmake). Typical options you may pass to cmake include:

- -DSlicer_DIR:PATH=/path/to/Slicer-build
- -DUSE_SUPERBUILD=ON/OFF
- -DSOFA_SOURCE_DIR:PATH=... (optional override)
- -DCMAKE_BUILD_TYPE=Release/Debug

Building with SuperBuild
------------------------

From the project root:

.. code-block:: bash

   mkdir build && cd build
   cmake .. -DUSE_SUPERBUILD=ON -DSlicer_DIR:PATH=/path/to/Slicer-build
   make -j4

This will start fetching and compiling SOFA and other libraries, then build the Slicer modules against them.

Using an external SOFA
----------------------

If you already have SOFA installed or built separately, disable SuperBuild for SOFA and point the module CMake to your SOFA installation via CMAKE_PREFIX_PATH or use -DSOFA_DIR:PATH=/path/to/sofa/install.

Common issues
-------------

- Ensure you use compatible compiler versions with Slicer and SOFA.
- On systems with a different Qt than Slicer, prefer using the system Qt (APT) to avoid conflicts.
- Building SOFA can take a long time and may require additional libraries (OpenMP, CUDA, etc.) depending on enabled features.
