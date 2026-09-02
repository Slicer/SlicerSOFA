Overview
========

What is SlicerSOFA?
-------------------

SlicerSOFA is an open-source extension that integrates the SOFA simulation
framework into 3D Slicer, a versatile platform for medical image analysis and
visualization. This integration facilitates the development and execution of
biomechanical models, with applications in surgical planning, medical training,
and biomedical research. The software is distributed under the MIT License.

The extension is implemented entirely in Python: it embeds SOFA (with its
Python bindings, `SofaPython3 <https://sofapython3.readthedocs.io/>`_) into
Slicer's Python environment and provides a scripted-module infrastructure on
top of it.

Architecture at a glance
------------------------

The integration works at two levels:

**Fundamental infrastructure level.**
SOFA and SofaPython3 are built and shipped with the extension, so any Slicer
user or developer can ``import Sofa`` from Slicer's Python environment and use
SOFA's simulation capabilities directly.

**Scripted module infrastructure level.**
On top of the core integration, the hidden ``SlicerSofa`` support module
provides base classes and utilities for building simulation-based Slicer
modules:

- **Simulation lifecycle management** — start, step, stop, and reset a SOFA
  simulation from Slicer, driven by a Qt timer.
- **Data interchange** — a registry of mapping functions that synchronize MRML
  nodes (models, markups, transforms) with SOFA components in both directions
  on every simulation step (see :doc:`mappings-reference`).
- **Recording and playback** — opt-in, step-aligned recording of simulation
  state into Slicer's Sequences module, one recorded frame per executed
  simulation step.

See :doc:`architecture` for the details.

Included modules
----------------

The extension ships one support module and three example modules (found under
the *Examples* category in Slicer):

- **SlicerSofa** *(hidden)* — base classes (``SlicerSofaLogic``,
  ``SlicerSofaWidget``, ``SofaParameterNodeWrapper``) and the
  ``SlicerSofaUtils.Mappings`` conversion library.
- **Soft Tissue Simulation** — FEM simulation of a deformable organ (sample
  data: a low-resolution right lung tetrahedral mesh) under gravity, with
  fixed-boundary selection and von Mises stress visualization.
  See :doc:`softtissue-simulation`.
- **Sparse Grid Simulation** — deformation of a surface model embedded in a
  sparse hexahedral grid; exports the displacement field as an MRML grid
  transform that can be applied to volumes and segmentations.
  See :doc:`sparsegrid-simulation`.
- **SOFA Scene Loader** — loads an arbitrary SOFA Python scene file and
  auto-maps its renderable components to MRML model nodes.
  See :doc:`sofasceneloader`.

.. image:: ../images/SoftTissueSimulationScreenshot_1.png
   :alt: Soft Tissue Simulation module
   :align: center

Bundled SOFA
------------

The extension builds and bundles its own SOFA (a Slicer-maintained fork of
SOFA v26.06) together with a set of plugins, including SofaPython3, STLIB,
BeamAdapter, Registration, Cosserat, and MultiThreading. Scenes may only use
components from SOFA modules and plugins available in this bundle (see
:doc:`limitations`).

Acknowledgments
---------------

Contributors:

- Rafael Palomar (Oslo University Hospital / NTNU, Norway)
- Paul Baksic (INRIA, France)
- Steve Pieper (Isomics, Inc., USA)
- Andras Lasso (Queen's University, Canada)
- Sam Horvath (Kitware, Inc., USA)
- Jean-Christophe Fillion-Robin (Kitware, Inc., USA)

This project has been funded by Oslo University Hospital.
