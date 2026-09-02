Architecture
============

SlicerSOFA is a pure-Python extension. It consists of three layers, all under
the hidden ``SlicerSofa`` support module, plus the example modules built on
top of them.

SofaEnvironment
---------------

``SofaEnvironment`` is a small package that makes the bundled SOFA importable
from Slicer's Python environment. On import it:

- sets ``SOFA_ROOT`` to the bundled SOFA (handling install-tree and build-tree
  layouts on Linux, macOS, and Windows),
- prepends the SofaPython3 ``site-packages`` directories of the bundled
  plugins to ``sys.path``,
- imports ``Sofa`` and ``SofaRuntime`` and restores Slicer's exception hook
  (SOFA replaces ``sys.excepthook`` on import).

Modules should therefore use ``from SofaEnvironment import *`` (or
``from SofaEnvironment import Sofa, SofaRuntime``) instead of importing
``Sofa`` directly.

SlicerSofa base classes
-----------------------

``SlicerSofa.py`` provides three building blocks:

``SofaParameterNodeWrapper``
   A class decorator that combines ``dataclass`` and Slicer's
   ``parameterNodeWrapper`` and injects the standard simulation-control
   fields into the decorated class:

   .. list-table::
      :header-rows: 1
      :widths: 30 20 50

      * - Field
        - Default
        - Meaning
      * - ``dt``
        - ``0.01``
        - Simulation time step passed to ``Sofa.Simulation.animate``.
      * - ``totalSteps``
        - ``-1``
        - Number of steps to run; a negative value means unlimited.
      * - ``currentStep``
        - ``0``
        - Steps executed since the simulation started.
      * - ``isSimulationRunning``
        - ``False``
        - Lifecycle flag; drives GUI enabling/disabling.
      * - ``simulationProgress``
        - ``''``
        - Read-only text, e.g. ``"42/100"`` or ``"42/∞"``.

``SlicerSofaLogic``
   The simulation engine, subclassing ``ScriptedLoadableModuleLogic``.
   Concrete modules implement ``createScene(parameterNode)`` (returning a
   ``Sofa.Core.Node`` root) and ``setupMappings()``, and may override the
   ``_saveState()`` / ``_restoreState()`` hooks used by simulation reset.

``SlicerSofaWidget``
   A ``ScriptedLoadableModuleWidget`` base that manages parameter-node GUI
   binding on enter/exit and scene close, and implements
   ``updateWidgetOnSimulation()``: any widget in the module UI carrying the
   dynamic property ``SlicerDisableOnSimulation`` is automatically
   enabled/disabled when the simulation starts or stops.

Simulation lifecycle
--------------------

A running simulation is driven by a ``qt.QTimer`` owned by the module widget
(interval 0: one step per event-loop iteration, on the main thread):

.. code-block:: text

   startSimulation()
     ├─ _saveState()                  # module hook: snapshot for reset
     ├─ resetRunOnceFlags()
     ├─ setupScene(parameterNode)
     │    ├─ createScene(parameterNode)   # module: build the SOFA scene
     │    ├─ __updateSofa__()             # apply MRML→SOFA mappings once
     │    └─ Sofa.Simulation.init(root)
     ├─ currentStep = 0; isSimulationRunning = True
     ├─ setupSequenceRecording()      # only opted-in fields, see below
     └─ onSimulationStarted()         # module hook (default: update GUI)

   simulationStep()                   # called by the widget timer
     ├─ __updateSofa__()              # MRML→SOFA mappings
     ├─ Sofa.Simulation.animate(root, dt); currentStep += 1
     ├─ __updateMRML__()              # SOFA→MRML mappings
     └─ SaveProxyNodesState()         # one recorded frame per step

   stopSimulation()
     ├─ Sofa.Simulation.unload(root)
     ├─ isSimulationRunning = False
     ├─ stopSequenceRecording()
     └─ onSimulationStopped()

   resetSimulation() → _restoreState()   # module hook: restore snapshot

When ``currentStep`` reaches ``totalSteps`` (if non-negative), the logic
flips ``isSimulationRunning`` to ``False`` and calls
``onSimulationStopped()``; the timer keeps firing but steps become no-ops
until the widget stops it.

The mapping registry
--------------------

Data interchange is declarative. During ``setupMappings()`` a module registers
mapping functions against parameter-node fields and SOFA scene paths:

.. code-block:: python

   self.registerMRMLToSOFAMapping('boundaryROI', 'FEM.FixedROI.BoxROI',
                                  mrmlMarkupsROIToSofaBoxROI)
   self.registerSOFAToMRMLMapping('modelNode', 'FEM.Collision.dofs',
                                  sofaMechanicalObjectToMRMLModelGrid)

- **MRML→SOFA** mappings run before each ``animate`` call (Slicer state, such
  as a moved ROI, is pushed into the SOFA scene).
- **SOFA→MRML** mappings run after each ``animate`` call (simulation results
  are pulled back into MRML nodes).
- The *SOFA path* is a dot-separated path resolved from the root node through
  ``getChild``/``getObject`` (e.g. ``"FEM.Collision.dofs"``); the empty
  string resolves to the root node itself.
- A mapping function has the signature ``fn(parameterNodeFieldValue,
  sofaObject)``.
- ``runOnce=True`` makes a mapping execute only on the first step after a
  simulation start (typically used to transfer static topology).
- Mappings whose parameter-node field is ``None``, or whose SOFA path cannot
  be resolved, are skipped.

The ready-made mapping functions are documented in :doc:`mappings-reference`.

Sequence recording
------------------

Recording into Slicer's Sequences module is opt-in per field:

.. code-block:: python

   self.setRecordSequenceFlag('modelNode', lambda: pn.recordSequence)

At ``startSimulation()`` time, each registered field whose flag function
evaluates truthy (and whose node is set) gets a sequence node in a shared
*SOFA Simulation Browser* sequence browser. Capture is driven explicitly from
``simulationStep()`` — exactly one frame per executed step — rather than by
the browser's clock-based recording. This makes the frame count deterministic
and guarantees that the last recorded frame equals the final simulation
state, so reading a recorded node after ``stopSimulation()`` is safe.
