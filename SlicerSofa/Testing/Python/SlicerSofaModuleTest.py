"""Characterization tests for the SlicerSofaUtils.Mappings layer.

These pin the *observable behavior* of the mapping functions -- argument
convention, array shapes, sign and axis order, None-handling -- so that the
planned API refactor can be verified rather than hoped at.

Deliberately written against real MRML nodes and real SOFA components instead
of mocks.  The previous version of this file mocked a parameter node with a
`_rootNode` dict keyed by SOFA path string and called mappings as
`f(mockParameterNode, 'FEM.Container')`.  That convention no longer exists --
mappings now take `(mrmlNode, sofaObject)` with the path already resolved --
so the mocks documented a design rather than testing the code.

Assertions are analytic (closed-form expectations) with no stored reference
data, so the suite is hermetic: no downloads, nothing to keep in sync.
"""

import unittest

import numpy as np
import vtk
import vtk.util.numpy_support
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest

import Sofa
import Sofa.Core

from SlicerSofaUtils.Mappings import (
    mrmlModelGridToSofaTetrahedronTopologyContainer,
    mrmlMarkupsFiducialToSofaPointer,
    mrmlMarkupsROIToSofaBoxROI,
    sofaMechanicalObjectToMRMLModelGrid,
    arrayFromMarkupsROIPoints,
    arrayVectorFromMarkupsLinePoints,
    arrayFromModelGridCells,
)

# SOFA components live in plugins that must be loaded explicitly; without the
# RequiredPlugin the component type is never registered and addObject fails
# with "Object type <Name><> was not created".
REQUIRED_PLUGINS = (
    "Sofa.Component.StateContainer",              # MechanicalObject
    "Sofa.Component.Engine.Select",               # BoxROI
    "Sofa.Component.Topology.Container.Dynamic",  # TetrahedronSetTopologyContainer
)

# A single tetrahedron: 4 points, 1 cell.
TETRA_POINTS = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]


def makeSofaRoot():
    """A SOFA root with the plugins these tests need already loaded."""
    root = Sofa.Core.Node("root")
    for plugin in REQUIRED_PLUGINS:
        root.addObject("RequiredPlugin", name=plugin)
    return root


def makeTetrahedronModelNode(name="Tetra"):
    """MRML model node holding a single-tetrahedron unstructured grid."""
    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    for point in TETRA_POINTS:
        points.InsertNextPoint(*point)
    grid.SetPoints(points)

    tetra = vtk.vtkTetra()
    for i in range(4):
        tetra.GetPointIds().SetId(i, i)
    grid.InsertNextCell(tetra.GetCellType(), tetra.GetPointIds())

    modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
    modelNode.SetAndObserveMesh(grid)
    return modelNode


class SlicerSofaUtilsTest(ScriptedLoadableModuleTest):
    """Test case for the SlicerSofaUtils mapping and utility functions."""

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def tearDown(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run all tests.

        NOTE: Slicer's runner calls runTest() only, so each test must be listed
        explicitly here.  An unlisted test_* method is silently never executed --
        which is how this entire file rotted undetected.
        """
        for name in (
            "test_arrayFromMarkupsROIPoints",
            "test_arrayFromMarkupsROIPoints_handlesNone",
            "test_arrayVectorFromMarkupsLinePoints",
            "test_arrayVectorFromMarkupsLinePoints_handlesNone",
            "test_arrayFromModelGridCells",
            "test_arrayFromModelGridCells_rejectsNone",
            "test_sofaMechanicalObjectToMRMLModelGrid",
            "test_mrmlMarkupsROIToSofaBoxROI",
            "test_mrmlModelGridToSofaTetrahedronTopologyContainer",
            "test_mrmlMarkupsFiducialToSofaPointer",
            "test_mappingsRejectNoneArguments",
        ):
            self.setUp()
            getattr(self, name)()
        self.tearDown()

    # -- utility functions ----------------------------------------------------

    def test_arrayFromMarkupsROIPoints(self):
        """ROI bounds come back interleaved as [Rmin, Amin, Smin, Rmax, Amax, Smax]."""
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        roiNode.SetCenter([10.0, 20.0, 30.0])
        roiNode.SetSize(10.0, 20.0, 30.0)

        # Note the ordering: all minima precede all maxima, which is *not* the
        # VTK bounds convention (xmin,xmax,ymin,ymax,zmin,zmax).
        self.assertEqual(
            arrayFromMarkupsROIPoints(roiNode),
            [5.0, 10.0, 15.0, 15.0, 30.0, 45.0])

    def test_arrayFromMarkupsROIPoints_handlesNone(self):
        self.assertEqual(arrayFromMarkupsROIPoints(None), [0.0] * 6)

    def test_arrayVectorFromMarkupsLinePoints(self):
        """The line vector is returned normalized."""
        lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        lineNode.AddControlPoint([1.0, 2.0, 3.0])
        lineNode.AddControlPoint([4.0, 6.0, 3.0])

        vector = arrayVectorFromMarkupsLinePoints(lineNode)
        np.testing.assert_array_almost_equal(vector, [0.6, 0.8, 0.0])
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_arrayVectorFromMarkupsLinePoints_handlesNone(self):
        self.assertEqual(arrayVectorFromMarkupsLinePoints(None), [0.0] * 3)

    def test_arrayFromModelGridCells(self):
        """Cell connectivity drops the per-cell point count.

        NOTE: the implementation reshapes to (-1, 5) and slices [:, 1:5], so it
        assumes every cell is a tetrahedron.  Any other cell type silently
        yields garbage rather than raising.
        """
        modelNode = makeTetrahedronModelNode()
        connectivity = arrayFromModelGridCells(modelNode)

        self.assertEqual(connectivity.shape, (1, 4))
        np.testing.assert_array_equal(connectivity, [[0, 1, 2, 3]])

    def test_arrayFromModelGridCells_rejectsNone(self):
        with self.assertRaises(ValueError):
            arrayFromModelGridCells(None)

    # -- SOFA -> MRML ---------------------------------------------------------

    def test_sofaMechanicalObjectToMRMLModelGrid(self):
        """MechanicalObject positions land on the model node's grid points."""
        root = makeSofaRoot()
        mechanicalObject = root.addObject(
            "MechanicalObject", name="dofs", template="Vec3d", position=TETRA_POINTS)

        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Target")
        sofaMechanicalObjectToMRMLModelGrid(modelNode, mechanicalObject)

        grid = modelNode.GetUnstructuredGrid()
        self.assertIsNotNone(grid, "mapping should allocate the grid when absent")

        points = vtk.util.numpy_support.vtk_to_numpy(grid.GetPoints().GetData())
        np.testing.assert_array_almost_equal(points, np.array(TETRA_POINTS), decimal=5)

    # -- MRML -> SOFA ---------------------------------------------------------

    def test_mrmlMarkupsROIToSofaBoxROI(self):
        """ROI bounds are written into the SOFA BoxROI's box field."""
        root = makeSofaRoot()
        root.addObject("MechanicalObject", name="dofs", template="Vec3d",
                       position=TETRA_POINTS)
        boxROI = root.addObject("BoxROI", name="roi", box=[[0., 0., 0., 1., 1., 1.]])

        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        roiNode.SetCenter([1.0, 2.0, 3.0])
        roiNode.SetSize(2.0, 4.0, 6.0)

        mrmlMarkupsROIToSofaBoxROI(roiNode, boxROI)

        np.testing.assert_array_almost_equal(
            np.asarray(boxROI.box.array()).reshape(-1)[:6],
            [0.0, 0.0, 0.0, 2.0, 4.0, 6.0])

    def test_mrmlModelGridToSofaTetrahedronTopologyContainer(self):
        """Grid points and tetrahedra transfer into the SOFA container."""
        root = makeSofaRoot()
        container = root.addObject(
            "TetrahedronSetTopologyContainer", name="container",
            position=TETRA_POINTS, tetrahedra=[[0, 1, 2, 3]])

        modelNode = makeTetrahedronModelNode()
        mrmlModelGridToSofaTetrahedronTopologyContainer(modelNode, container)

        np.testing.assert_array_equal(
            np.asarray(container.tetrahedra.array()).reshape(-1, 4), [[0, 1, 2, 3]])
        np.testing.assert_array_almost_equal(
            np.asarray(container.position.array()).reshape(-1, 3),
            np.array(TETRA_POINTS), decimal=5)

    def test_mrmlMarkupsFiducialToSofaPointer(self):
        """Fiducial control points transfer into a MechanicalObject position."""
        root = makeSofaRoot()
        pointer = root.addObject("MechanicalObject", name="pointer",
                                 template="Vec3d", position=[[0., 0., 0.]])

        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        fiducialNode.AddControlPoint([7.0, 8.0, 9.0])

        mrmlMarkupsFiducialToSofaPointer(fiducialNode, pointer)

        np.testing.assert_array_almost_equal(
            np.asarray(pointer.position.array()).reshape(-1, 3), [[7.0, 8.0, 9.0]],
            decimal=5)

    # -- contract: None arguments -------------------------------------------

    def test_mappingsRejectNoneArguments(self):
        """Every mapping raises ValueError rather than failing obscurely later."""
        root = makeSofaRoot()
        mechanicalObject = root.addObject("MechanicalObject", name="dofs",
                                          template="Vec3d", position=TETRA_POINTS)
        modelNode = makeTetrahedronModelNode()

        for mapping in (mrmlModelGridToSofaTetrahedronTopologyContainer,
                        mrmlMarkupsFiducialToSofaPointer,
                        mrmlMarkupsROIToSofaBoxROI,
                        sofaMechanicalObjectToMRMLModelGrid):
            with self.assertRaises(ValueError, msg=f"{mapping.__name__} accepted None first arg"):
                mapping(None, mechanicalObject)
            with self.assertRaises(ValueError, msg=f"{mapping.__name__} accepted None second arg"):
                mapping(modelNode, None)
