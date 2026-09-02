Mappings reference
==================

``SlicerSofaUtils.Mappings`` provides ready-made mapping functions for the
mapping registry (see :doc:`architecture`), plus a few array utilities. All
mapping functions share the signature ``fn(mrmlNode, sofaObject)`` where the
first argument is the parameter-node field value and the second the SOFA
object resolved from the registered path.

MRML → SOFA
-----------

Registered with ``registerMRMLToSOFAMapping``; executed before each
simulation step.

``mrmlModelPolyToSofaTriangleTopologyContainer(modelNode, sofaNode)``
   Transfers points and triangles from the ``vtkPolyData`` of a model node to
   a SOFA ``TriangleSetTopologyContainer`` (``position`` and ``triangles``
   data fields). Typically registered with ``runOnce=True``.

``mrmlModelGridToSofaTetrahedronTopologyContainer(modelNode, sofaNode)``
   Transfers points and tetrahedra from the ``vtkUnstructuredGrid`` of a
   model node to a SOFA ``TetrahedronSetTopologyContainer``. Typically
   registered with ``runOnce=True``.

``mrmlMarkupsFiducialToSofaPointer(fiducialNode, sofaNode)``
   Writes the fiducial control-point positions into the target's
   ``position`` data field (e.g. a pointer/interactor mechanical object).

``mrmlMarkupsROIToSofaBoxROI(roiNode, sofaNode)``
   Writes the ROI bounds (as ``[R_min, A_min, S_min, R_max, A_max, S_max]``)
   into a SOFA ``BoxROI``'s ``box`` data field.

SOFA → MRML
-----------

Registered with ``registerSOFAToMRMLMapping``; executed after each simulation
step.

``sofaMechanicalObjectToMRMLModelPoly(modelNode, sofaNode)``
   Copies a ``MechanicalObject``'s positions into the points of the model
   node's ``vtkPolyData`` (topology unchanged).

``sofaMechanicalObjectToMRMLModelGrid(modelNode, sofaNode)``
   Copies a ``MechanicalObject``'s positions into the points of the model
   node's ``vtkUnstructuredGrid``.

``sofaSparseGridTopologyToMRMLModelGrid(modelNode, sofaNode)``
   Builds VTK hexahedral cells from a ``SparseGridTopology``'s ``hexahedra``.
   Typically registered with ``runOnce=True`` (topology is static).

``sofaTetrahedronTopologyToMRMLModelGrid(modelNode, sofaNode)``
   Builds VTK tetrahedral cells from a topology container's ``tetrahedra``.

``sofaTriangleTopologyToMRMLModelGrid(modelNode, sofaNode)``
   Builds VTK triangle cells from a topology container's ``triangles``.

``sofaEdgeTopologyToMRMLModelGrid(modelNode, sofaNode)``
   Builds VTK line cells from a topology container's ``edges``.

``sofaMeshTopologyToMRMLModelGrid(modelNode, sofaNode)``
   Dispatches to the tetrahedron, triangle, or edge variant above depending
   on which element type the SOFA ``MeshTopology`` actually contains.

``sofaOglModelToMRMLModelGrid(modelNode, sofaNode)``
   Transfers an ``OglModel``'s topology (triangles or edges) and positions.

``sofaVonMisesStressToMRMLModelGrid(modelNode, sofaNode)``
   Copies a FEM force field's ``vonMisesPerElement`` array into a cell-data
   array named ``VonMisesStress`` on the model node's unstructured grid, and
   updates the display node's scalar range. Requires the force field to be
   created with ``computeVonMisesStress`` enabled.

Utilities
---------

``arrayFromMarkupsROIPoints(roiNode)``
   Returns the ROI bounds as ``[R_min, A_min, S_min, R_max, A_max, S_max]``
   (all zeros if ``roiNode`` is ``None``) — the format expected by SOFA's
   ``BoxROI``.

``arrayVectorFromMarkupsLinePoints(lineNode)``
   Returns the normalized direction vector of a line markup as ``[x, y, z]``
   (zeros if ``lineNode`` is ``None``).

``arrayFromModelGridCells(modelNode)``
   Returns the tetrahedral connectivity of the model node's unstructured grid
   as an ``(N, 4)`` NumPy array — the format expected by SOFA's
   ``tetrahedra`` data fields.
