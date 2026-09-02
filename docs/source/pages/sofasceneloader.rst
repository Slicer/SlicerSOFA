SOFASceneLoader
===============

Overview
--------

The SOFASceneLoader module lets you load and run an arbitrary SOFA scene defined in a Python file directly from Slicer, without writing any Slicer-specific code. It automatically discovers renderable SOFA objects in the scene and maps them to MRML model nodes so they appear in the 3D view.

Unlike SoftTissueSimulation and SparseGridSimulation — which hard-code a specific scene and its SOFA-to-MRML mappings — SOFASceneLoader is generic: it works with any SOFA Python scene that exposes a ``createScene(root)`` function.

UI controls
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Control
     - Purpose
   * - File path field + Load Scene
     - Enter a path or browse for a ``.py`` scene file and load it. The
       working directory is set to the file's parent folder so that relative
       asset paths in the scene resolve correctly.
   * - Start Simulation
     - Initialise SOFA and begin stepping.
   * - Stop Simulation
     - Pause stepping (scene state is preserved).
   * - Reset Simulation
     - Reload the scene from the file and restart from step 0.
   * - dt
     - Simulation time step (default 0.01).
   * - Total Steps
     - Number of steps to run; ``-1`` means run indefinitely. Defaults to 0
       in this module, so set it before starting.
   * - Progress
     - Read-only display of ``currentStep / totalSteps``.

Scene file requirements
-----------------------

The selected file must be a Python module that defines a top-level function with the exact signature:

.. code-block:: python

   def createScene(root):
       ...
       return root

The ``root`` argument that SOFASceneLoader passes in is a ``SOFANodeWrapper`` proxy (not a bare ``Sofa.Core.Node``). The proxy is transparent — all standard SOFA calls (``addObject``, ``addChild``, attribute access) work normally — but it intercepts ``addObject`` and ``addChild`` calls to build the automatic MRML mappings (see below).

Auto-mapping mechanism
----------------------

When the scene file calls ``root.addObject("ComponentType", ...)`` or any ``addChild(...)`` chain, the ``SOFANodeWrapper`` checks whether ``ComponentType`` appears in the ``SOFA2MRML_dict`` lookup table. If it does:

1. A ``vtkMRMLModelNode`` is created (or reused if one with the same singleton tag already exists in the scene) and made visible in the 3D view.
2. A SOFA-to-MRML mapping is registered so that every simulation step copies the SOFA component's state into the MRML node.

The node name / singleton tag is derived from the dot-separated path of the SOFA object in the scene tree, with dots replaced by underscores (e.g. a ``MechanicalObject`` added inside ``root → FEM → Collision`` becomes ``FEM_Collision_<objectName>``).

Currently auto-mapped component types
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - SOFA component type
     - Data transferred to MRML
   * - ``MechanicalObject``
     - Vertex positions
   * - ``MeshTopology``
     - Mixed topology (tets/triangles/edges)
   * - ``EdgeSetTopologyContainer``
     - Edge topology
   * - ``TriangleSetTopologyContainer``
     - Triangle topology
   * - ``TetrahedronSetTopologyContainer``
     - Tetrahedron topology
   * - ``TetrahedralCorotationalFEMForceField``
     - Von Mises stress per element
   * - ``TetrahedronFEMForceField``
     - Von Mises stress per element
   * - ``OglModel``
     - Triangle/edge topology + positions

Example scene file
------------------

.. code-block:: python

   import Sofa

   def createScene(root):
       root.addObject("DefaultAnimationLoop")
       root.addObject("DefaultVisualManagerLoop")

       fem = root.addChild("FEM")
       fem.addObject("EulerImplicitSolver", rayleighStiffness=0.1, rayleighMass=0.1)
       fem.addObject("CGLinearSolver", iterations=25, tolerance=1e-9, threshold=1e-9)

       # Topology — auto-mapped because component type is in SOFA2MRML_dict
       fem.addObject("MeshGmshLoader", name="loader", filename="mesh/liver.msh")
       fem.addObject("TetrahedronSetTopologyContainer", src="@loader", name="topology")
       fem.addObject("MechanicalObject", src="@topology", name="dofs")
       fem.addObject("TetrahedronFEMForceField", youngModulus=5000, poissonRatio=0.4)

       return root

Workflow
--------

1. Open the module in Slicer (*Examples → SOFA Scene Loader*).
2. Type the path to your ``.py`` scene file in the file path field, or click **Load Scene** to browse.
3. The scene is parsed immediately: auto-detected objects appear as model nodes in the 3D view even before simulation starts.
4. Set **dt** and **Total Steps** as needed.
5. Click **Start Simulation**. The 3D view updates after each step.
6. Click **Stop Simulation** to pause, or **Reset Simulation** to reload and restart from step 0.

Reset behaviour
---------------

Clicking **Reset Simulation** calls ``_restoreState()``, which:

1. Stops the current simulation.
2. Clears all registered SOFA-to-MRML mappings.
3. Re-executes ``createScene`` against a fresh ``Sofa.Core.Node``, re-discovering and re-registering all auto-mapped objects.
4. Resets ``currentStep`` to 0 and pushes the initial state to MRML.

This means the scene Python file is re-imported on every reset, so in-place edits to the file take effect without restarting Slicer.

Extending auto-mapping
-----------------------

To map additional SOFA component types, add an entry to the ``SOFA2MRML_dict`` dictionary at the top of ``SOFASceneLoader.py``:

.. code-block:: python

   from SlicerSofaUtils.Mappings import myCustomMappingFunction

   SOFA2MRML_dict["MyCustomComponent"] = myCustomMappingFunction

The mapping function must have the signature ``fn(mrmlModelNode, sofaObject) -> None``.
