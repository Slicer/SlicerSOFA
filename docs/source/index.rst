SlicerSOFA documentation
========================

**SlicerSOFA** is a `3D Slicer <https://www.slicer.org/>`_ extension that
integrates the `SOFA simulation framework <https://www.sofa-framework.org/>`_
into Slicer, enabling physics-based simulation — soft tissue deformation,
biomechanical modeling — inside a medical imaging platform.

.. image:: images/slicer-sofa-overview.png
   :alt: What SlicerSOFA provides: live two-way synchronization between 3D Slicer data and SOFA simulations, recording into Sequences, and a Python API to build simulation modules
   :align: center

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   pages/overview
   pages/installation

.. toctree::
   :maxdepth: 2
   :caption: User guide

   pages/softtissue-simulation
   pages/sparsegrid-simulation
   pages/sofasceneloader

.. toctree::
   :maxdepth: 2
   :caption: Developer guide

   pages/architecture
   pages/module-development
   pages/mappings-reference
   pages/building
   pages/testing

.. toctree::
   :maxdepth: 1
   :caption: Reference

   pages/limitations
   pages/resources
