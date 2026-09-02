Installation
============

From the Extensions Manager (recommended)
-----------------------------------------

SlicerSOFA is distributed through the 3D Slicer Extensions Manager:

1. Start 3D Slicer.
2. Open **View → Extensions Manager**.
3. Search for **SlicerSOFA** (category *Simulation*) and click **Install**.
4. Restart Slicer when prompted.

After the restart, the example modules appear in the module list under the
**Examples** category:

- Soft Tissue Simulation
- Sparse Grid Simulation
- SOFA Scene Loader

The ``SlicerSofa`` support module itself is hidden — it provides
infrastructure, not a user interface.

Sample data
-----------

The example modules register their datasets with Slicer's **Sample Data**
module under the **SOFA** category:

- *RightLungLowTetra* — a low-resolution tetrahedral mesh of a right lung,
  used by the Soft Tissue Simulation module.
- *LiverSimulationScene* — a complete scene (``.mrb``) for the Sparse Grid
  Simulation module.

Using SOFA from Slicer's Python console
---------------------------------------

The bundled SOFA is importable from the Slicer Python console once the
extension is installed:

.. code-block:: python

   from SofaEnvironment import Sofa, SofaRuntime

   root = Sofa.Core.Node("root")

Importing ``SofaEnvironment`` (rather than ``Sofa`` directly) ensures that
``SOFA_ROOT`` and the plugin paths are configured for the bundled SOFA.

Building from source
--------------------

Developers who want to build the extension themselves (for example against a
custom Slicer build) should follow :doc:`building`.
