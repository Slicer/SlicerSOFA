Features
========

Overview
--------

SlicerSOFA provides an integration layer between the SOFA simulation framework and 3D Slicer enabling researchers to build, run and visualize physics-based simulations (soft tissue, deformable models, etc.) inside 3D Slicer.

Main features include:

- Load and run SOFA scenes from within Slicer.
- Visualize SOFA simulation state inside Slicer's 2D and 3D views.
- Example modules demonstrating typical use-cases:
  - SoftTissueSimulation: interactive soft tissue scenes.
  - SparseGridSimulation: specialized sparse-grid based simulations.
- Utilities to map SOFA data structures (meshes, transforms, pointclouds) to VTK/MRML.
- Python and C++ interfaces to control, step, pause and interact with simulations.
- Integrated build of ching/building required external dependencies (SOFA, Eigen, Boost, GLEW, pybind11, etc.).

Design goals
------------

- Keep simulations reproducible and scriptable through Slicer’s Python interface.
- Make simulation outputs (meshes, transforms) available as MRML nodes for further processing and visualization inside Slicer.
- Provide example simulations and a framework to extend Slicer with custom SOFA scenes and components.

Acknowledgments
---------------

The initial core developers are:

* Rafael Palomar (Oslo University Hospital / NTNU, Norway)
* Paul Baksic (Inria, France)
* Steve Pieper (Isomics, Inc.)
