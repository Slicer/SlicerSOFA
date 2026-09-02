Building from source
====================

Most users should install SlicerSOFA from the Extensions Manager
(:doc:`installation`). Building from source is for developers working on the
extension itself or needing a custom Slicer/SOFA combination.

Prerequisites
-------------

- A **3D Slicer built from source** (the extension is built against Slicer's
  build tree; see the `Slicer developer guide
  <https://slicer.readthedocs.io/en/latest/developer_guide/build_instructions/index.html>`_).
- CMake ≥ 3.16, Git, and the same compiler toolchain and Qt used for the
  Slicer build.
- Build time and disk space for SOFA: the SuperBuild compiles SOFA and its
  dependencies, which takes substantially longer than the extension itself.

Configure and build
-------------------

.. code-block:: bash

   git clone https://github.com/Slicer/SlicerSOFA.git
   mkdir SlicerSOFA-build && cd SlicerSOFA-build
   cmake ../SlicerSOFA \
     -DCMAKE_BUILD_TYPE=Release \
     -DSlicer_DIR:PATH=/path/to/Slicer-SuperBuild/Slicer-build
   cmake --build . --parallel

The SuperBuild
--------------

``SlicerSOFA_SUPERBUILD`` is ``ON`` by default. The SuperBuild first builds
the external dependencies, then the extension modules in the ``inner-build``
subdirectory:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - External project
     - Role
   * - Boost, Eigen3, GLEW, pybind11, tinyxml2
     - SOFA build dependencies.
   * - Sofa
     - A Slicer-maintained fork of SOFA (currently based on v26.06), built
       with SofaPython3 against Slicer's Python, plus the STLIB, BeamAdapter,
       Registration, and Cosserat plugins.

The SOFA and plugin versions are pinned in
``SuperBuild/External_Sofa.cmake``; the remaining external projects live in
the other ``SuperBuild/External_*.cmake`` files.

Running the built extension
---------------------------

Point your Slicer at the built modules, either by launching with the
generated launcher settings:

.. code-block:: bash

   /path/to/Slicer-SuperBuild/Slicer-build/Slicer \
     --launcher-additional-settings \
     /path/to/SlicerSOFA-build/inner-build/AdditionalLauncherSettings.ini

or by adding the module directories under
**Edit → Application Settings → Modules → Additional module paths**.

Packaging
---------

To create an installable extension package:

.. code-block:: bash

   cd SlicerSOFA-build/inner-build
   cmake --build . --target package

Packaging bundles the SOFA runtime with the extension (on macOS this includes
a fixup step that rewrites library paths into the app bundle).
