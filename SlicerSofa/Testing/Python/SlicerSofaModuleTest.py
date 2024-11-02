import logging
import unittest
import vtk, slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest
from SlicerSofaUtils.Mappings import (
    mrmlModelNodeGridToSofaTetrahedronTopologyContainer,
    markupsFiducialNodeToSofaPointer,
    markupsROINodeToSofaBoxROI,
    sofaMechanicalObjectToMRMLModelNodeGrid,
    sofaVonMisesStressToMRMLModelNodeGrid,
    arrayFromMarkupsROIPoints,
    arrayVectorFromMarkupsLinePoints
)

class SlicerSofaUtilsTest(ScriptedLoadableModuleTest):
    """This is the test case for SlicerSofaUtils functions."""

    def setUp(self):
        """Set up the test environment."""
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run all tests."""
        self.setUp()
        self.test_arrayFromMarkupsROIPoints()
        self.test_arrayVectorFromMarkupsLinePoints()
        self.test_mrmlModelNodeGridToSofaTetrahedronTopologyContainer()
        self.test_markupsFiducialNodeToSofaPointer()
        self.test_markupsROINodeToSofaBoxROI()
        self.test_sofaMechanicalObjectToMRMLModelNodeGrid()
        self.test_sofaVonMisesStressToMRMLModelNodeGrid()
        self.tearDown()

    def test_arrayFromMarkupsROIPoints(self):
        """Test arrayFromMarkupsROIPoints function."""
        # Create a ROI node
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        roiNode.SetXYZ([10, 20, 30])
        roiNode.SetRadiusXYZ(5, 10, 15)

        # Expected boundaries
        expected_bounds = [
            10 - 5, 20 - 10, 30 - 15,  # R_min, A_min, S_min
            10 + 5, 20 + 10, 30 + 15   # R_max, A_max, S_max
        ]

        # Call the function
        bounds = arrayFromMarkupsROIPoints(roiNode)

        # Assert that the bounds are as expected
        self.assertEqual(bounds, expected_bounds)

    def test_arrayVectorFromMarkupsLinePoints(self):
        """Test arrayVectorFromMarkupsLinePoints function."""
        # Create a Line node
        lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        point1 = [0, 0, 0]
        point2 = [1, 0, 0]
        lineNode.AddControlPoint(point1)
        lineNode.AddControlPoint(point2)

        # Expected vector
        expected_vector = [1.0, 0.0, 0.0]

        # Call the function
        vector = arrayVectorFromMarkupsLinePoints(lineNode)

        # Assert that the vector is as expected
        import numpy as np
        np.testing.assert_array_almost_equal(vector, expected_vector)

    def test_mrmlModelNodeGridToSofaTetrahedronTopologyContainer(self):
        """Test mrmlModelNodeGridToSofaTetrahedronTopologyContainer function."""
        # Create a mock parameter node
        class MockParameterNode:
            def __init__(self):
                self._currentMappingMRMLNode = None
                self._rootNode = {'FEM.Container': MockSofaNode()}

        class MockSofaNode:
            def __init__(self):
                self.tetrahedra = None
                self.position = None

        # Create an MRML Model Node with a simple Unstructured Grid
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        unstructuredGrid = vtk.vtkUnstructuredGrid()

        # Create points
        points = vtk.vtkPoints()
        points.InsertNextPoint(0.0, 0.0, 0.0)
        points.InsertNextPoint(1.0, 0.0, 0.0)
        points.InsertNextPoint(0.0, 1.0, 0.0)
        points.InsertNextPoint(0.0, 0.0, 1.0)
        unstructuredGrid.SetPoints(points)

        # Create a tetrahedron
        tetra = vtk.vtkTetra()
        tetra.GetPointIds().SetId(0, 0)
        tetra.GetPointIds().SetId(1, 1)
        tetra.GetPointIds().SetId(2, 2)
        tetra.GetPointIds().SetId(3, 3)

        # Add the cell to the unstructured grid
        unstructuredGrid.InsertNextCell(tetra.GetCellType(), tetra.GetPointIds())
        modelNode.SetAndObserveUnstructuredGrid(unstructuredGrid)

        # Set up the mock parameter node
        obj = MockParameterNode()
        obj._currentMappingMRMLNode = modelNode

        # Call the function
        mrmlModelNodeGridToSofaTetrahedronTopologyContainer(obj, 'FEM.Container')

        # Assert that the SOFA node's tetrahedra and positions are set
        expected_positions = [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
        expected_tetrahedra = [[0, 1, 2, 3]]

        self.assertEqual(obj._rootNode['FEM.Container'].position, expected_positions)
        self.assertEqual(obj._rootNode['FEM.Container'].tetrahedra, expected_tetrahedra)

    def test_markupsFiducialNodeToSofaPointer(self):
        """Test markupsFiducialNodeToSofaPointer function."""
        # Create a mock parameter node
        class MockParameterNode:
            def __init__(self):
                self._currentMappingMRMLNode = None
                self._rootNode = {'AttachPoint.mouseInteractor': MockSofaNode()}

        class MockSofaNode:
            def __init__(self):
                self.position = None

        # Create a fiducial node
        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        fiducialNode.AddControlPoint([1.0, 2.0, 3.0])

        # Set up the mock parameter node
        obj = MockParameterNode()
        obj._currentMappingMRMLNode = fiducialNode

        # Call the function
        markupsFiducialNodeToSofaPointer(obj, 'AttachPoint.mouseInteractor')

        # Expected position
        expected_position = [list(fiducialNode.GetNthControlPointPosition(0)) * 3]

        # Assert that the SOFA node's position is set
        self.assertEqual(obj._rootNode['AttachPoint.mouseInteractor'].position, expected_position)

    def test_markupsROINodeToSofaBoxROI(self):
        """Test markupsROINodeToSofaBoxROI function."""
        # Create a mock parameter node
        class MockParameterNode:
            def __init__(self):
                self._currentMappingMRMLNode = None
                self._rootNode = {'FEM.FixedROI.BoxROI': MockSofaNode()}

        class MockSofaNode:
            def __init__(self):
                self.box = None

        # Create a ROI node
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        roiNode.SetXYZ([10, 20, 30])
        roiNode.SetRadiusXYZ(5, 10, 15)

        # Set up the mock parameter node
        obj = MockParameterNode()
        obj._currentMappingMRMLNode = roiNode

        # Call the function
        markupsROINodeToSofaBoxROI(obj, 'FEM.FixedROI.BoxROI')

        # Expected box
        expected_box = [arrayFromMarkupsROIPoints(roiNode)]

        # Assert that the SOFA node's box is set
        self.assertEqual(obj._rootNode['FEM.FixedROI.BoxROI'].box, expected_box)

    def test_sofaMechanicalObjectToMRMLModelNodeGrid(self):
        """Test sofaMechanicalObjectToMRMLModelNodeGrid function."""
        # Create a mock parameter node
        class MockParameterNode:
            def __init__(self):
                self._currentMappingMRMLNode = None
                self._rootNode = {'FEM.Collision.dofs': MockSofaNode()}

        class MockSofaNode:
            def __init__(self):
                self.position = MockPosition()

        class MockPosition:
            def array(self):
                return np.array([[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0],
                                 [7.0, 8.0, 9.0]])

        # Create a MRML Model Node with an Unstructured Grid
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(3)
        unstructuredGrid.SetPoints(points)
        modelNode.SetAndObserveUnstructuredGrid(unstructuredGrid)

        # Set up the mock parameter node
        obj = MockParameterNode()
        obj._currentMappingMRMLNode = modelNode

        # Call the function
        sofaMechanicalObjectToMRMLModelNodeGrid(obj, 'FEM.Collision.dofs')

        # Retrieve the updated points from the MRML node
        updatedPoints = obj._currentMappingMRMLNode.GetUnstructuredGrid().GetPoints()

        # Convert VTK points to numpy array
        vtk_array = updatedPoints.GetData()
        numpy_array = vtk.util.numpy_support.vtk_to_numpy(vtk_array)

        # Expected positions
        expected_positions = np.array([[1.0, 2.0, 3.0],
                                       [4.0, 5.0, 6.0],
                                       [7.0, 8.0, 9.0]])

        # Assert that the positions match
        np.testing.assert_array_almost_equal(numpy_array, expected_positions)

    def test_sofaVonMisesStressToMRMLModelNodeGrid(self):
        """Test sofaVonMisesStressToMRMLModelNodeGrid function."""
        # Create a mock parameter node
        class MockParameterNode:
            def __init__(self):
                self._currentMappingMRMLNode = None
                self._rootNode = {'FEM.FEM': MockSofaNode()}

        class MockSofaNode:
            def __init__(self):
                self.vonMisesPerElement = MockStress()

        class MockStress:
            def array(self):
                return np.array([0.1, 0.2, 0.3])

        # Create a MRML Model Node with an Unstructured Grid
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        unstructuredGrid.Allocate(3)

        # Add cells to the unstructured grid
        for i in range(3):
            cell = vtk.vtkVertex()
            cell.GetPointIds().SetId(0, i)
            unstructuredGrid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

        modelNode.SetAndObserveUnstructuredGrid(unstructuredGrid)

        # Set up the mock parameter node
        obj = MockParameterNode()
        obj._currentMappingMRMLNode = modelNode

        # Call the function
        sofaVonMisesStressToMRMLModelNodeGrid(obj, 'FEM.FEM')

        # Retrieve the stress array from the MRML node
        stressArray = slicer.util.arrayFromModelCellData(modelNode, "VonMisesStress")

        # Expected stress values
        expected_stress = np.array([0.1, 0.2, 0.3])

        # Assert that the stress values match
        np.testing.assert_array_almost_equal(stressArray, expected_stress)

        def tearDown(self):
        """Clean up after tests."""
        slicer.mrmlScene.Clear(0)
