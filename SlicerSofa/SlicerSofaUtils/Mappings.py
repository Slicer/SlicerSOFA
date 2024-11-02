import slicer
import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk

# -----------------------------------------------------------------------------
# Mapping functions MRML->Sofa
# -----------------------------------------------------------------------------
#
def mrmlModelNodePolyToSofaTriangleTopologyContainer(obj, nodePath) -> None:
    """
    This is mapping function that will transfer geometry (points) and
    topology (cells) from a vtkPolyData stored in a vtkMRMLModel node
    to a Sofa TriangleSetTopologyContainer.

    Attributes:
        obj (ParameterNode): Parameter node for the mapping
        nodePath (str): Sofa node nodeNodePath
    """
    if not isinstance(nodePath, (str, bytes)):
        TypeError("nodePath must be a string")

    if obj._currentMappingMRMLNode is None:
        return

    # Update SOFA node with tetrahedra and positions
    obj._rootNode[nodePath].triangle = slicer.util.arrayFromModelPolyIds(obj._currentMappingMRMLNode).reshape(-1,4)[:,1:]
    obj._rootNode[nodePath].position = slicer.util.arrayFromModelPoints(obj._currentMappingMRMLNode)


def mrmlModelNodeGridToSofaTetrahedronTopologyContainer(obj, nodePath) -> None:
    """
    This is mapping function that will transfer geometry (points) and
    topology (cells) from a vtkUnstructuredGrid stored in a vtkMRMLModel node
    to a Sofa TetrahedronSetTopologyContainer.

    Attributes:
        obj (ParameterNode): Parameter node for the mapping
        nodePath (str): Sofa node nodeNodePath
    """
    if not isinstance(nodePath, (str, bytes)):
        TypeError("nodePath must be a string")

    if obj._currentMappingMRMLNode is None:
        return

    # Retrieve unstructured grid data from the model node
    unstructuredGrid = obj._currentMappingMRMLNode.GetUnstructuredGrid()
    points = unstructuredGrid.GetPoints()
    numPoints = points.GetNumberOfPoints()

    # Convert VTK points to a list for SOFA node
    pointCoords = [points.GetPoint(i) for i in range(numPoints)]

    # Parse cell data (tetrahedra connectivity)
    cells = unstructuredGrid.GetCells()
    cellArray = vtk.util.numpy_support.vtk_to_numpy(cells.GetData())
    cellConnectivity = []
    idx = 0
    for i in range(unstructuredGrid.GetNumberOfCells()):
        numPoints = cellArray[idx]
        cellConnectivity.append(cellArray[idx + 1:idx + 1 + numPoints].tolist())
        idx += numPoints + 1

    # Update SOFA node with tetrahedra and positions
    obj._rootNode[nodePath].tetrahedra = cellConnectivity
    obj._rootNode[nodePath].position = pointCoords


def markupsFiducialNodeToSofaPointer(obj, nodePath) -> None:
    """
    This is mapping function that will transfer a 3D fiducial point
    to a pointer interactor in Sofa

    Attributes:
        obj (ParameterNode): Parameter node for the mapping
        nodePath (str): Sofa node nodePath
    """

    if not isinstance(nodePath, (str, bytes)):
       TypeError("nodePath must be a string")

    if obj._currentMappingMRMLNode is None:
        return

    # Set the SOFA node position based on the first control point of the fiducial node
    obj._rootNode[nodePath].position = [list(self._currentMappingMRMLNode.GetNthControlPointPosition(0)) * 3]

def markupsROINodeToSofaBoxROI(obj, nodePath) -> None:
    """
    This is mapping function that will transfer a markups ROI
    to a Sofa ROI Box

    Attributes:
        obj (ParameterNode): Parameter node for the mapping
        nodePath (str): Sofa node nodePath
    """

    if not isinstance(nodePath, (str, bytes)):
       TypeError("nodePath must be a string")

    if obj._currentMappingMRMLNode is None:
        return

    obj._rootNode[nodePath].box = [arrayFromMarkupsROIPoints(obj._currentMappingMRMLNode)]

# -----------------------------------------------------------------------------
# Mapping functions Sofa->MRML
# -----------------------------------------------------------------------------

def sofaMechanicalObjectToMRMLModelNodePoly(obj, nodePath) -> None:
    """
    This is mapping function that will transfer geometry from a Sofa
    mechanical object to a vtkPolyData stored in a vtkMRMLModel node.

    Attributes:
        obj (ParameterNode): Parameter node for the mapping
        nodePath (str): Sofa node nodePath
    """
    if not isinstance(nodePath, (str, bytes)):
       TypeError("nodePath must be a string")

    if obj._currentMappingMRMLNode is None:
        return

    if obj._currentMappingMRMLNode.GetPolyData() is None:
        polyData = vtk.vtkPolyData()
        obj._currentMappingMRMLNode.SetAndObservePolyData(polyData)

    surfacePointsArray = obj._rootNode[nodePath].position.array()
    surfaceModelPointsArray = slicer.util.arrayFromModelPoints(obj._currentMappingMRMLNode)
    surfaceModelPointsArray[:] = surfacePointsArray
    slicer.util.arrayFromModelPointsModified(obj._currentMappingMRMLNode)

def sofaMechanicalObjectToMRMLModelNodeGrid(obj, nodePath) -> None:
    """
    This is mapping function that will transfer geometry from a Sofa
    mechanical object to a vtkUnstructuredGrid stored in a vtkMRMLModel node.

    Attributes:
        obj (ParameterNode): Parameter node for the mapping
        nodePath (str): Sofa node nodePath
    """
    if not isinstance(nodePath, (str, bytes)):
       TypeError("nodePath must be a string")

    if obj._currentMappingMRMLNode is None:
        return

    # Convert SOFA node positions to VTK points
    convertedPoints = numpy_to_vtk(num_array=obj._rootNode[nodePath].position.array(), deep=True, array_type=vtk.VTK_FLOAT)
    points = vtk.vtkPoints()
    points.SetData(convertedPoints)

    if obj._currentMappingMRMLNode.GetUnstructuredGrid() is None:
       unstructuredGrid = vtk.vtkUnstructuredGrid()
       obj._currentMappingMRMLNode.SetAndObserveMesh(unstructuredGrid)

    # Update the VTK model node's points and mark it as modified
    obj._currentMappingMRMLNode.GetUnstructuredGrid().SetPoints(points)
    obj._currentMappingMRMLNode.Modified()


def sofaSparseGridTopologyToMRMLModelNodeGrid(obj, nodePath) -> None:
    """
    This is mapping function that will transfer topology from a Sofa
    SparseGridTopology to a vtkUnstructuredGrid stored in a vtkMRMLModel node.

    Attributes:
        obj (ParameterNode): Parameter node for the mapping
        nodePath (str): Sofa node nodePath
    """

    if not isinstance(nodePath, (str, bytes)):
       TypeError("nodePath must be a string")

    if obj._currentMappingMRMLNode is None:
        return

    # Create a vtkCellArray to store the cells
    cell_array = vtk.vtkCellArray()

    # Iterate over the cell connectivity to create hexahedral cells
    for cell in obj._rootNode[nodePath].hexahedra.array():
        if len(cell) != 8:
            raise ValueError("Each hexahedral cell should have exactly 8 points.")

        hexahedron = vtk.vtkHexahedron()

        for i in range(8):
            hexahedron.GetPointIds().SetId(i, cell[i])

        cell_array.InsertNextCell(hexahedron)

    # Set the cells (hexahedrons) into the unstructured grid
    obj._currentMappingMRMLNode.GetUnstructuredGrid().SetCells(vtk.VTK_HEXAHEDRON, cell_array)


def sofaVonMisesStressToMRMLModelNodeGrid(obj, nodePath) -> None:
    """
    This is a mapping function that will transfer a von Mises stress array
    to an array in a vtkUnstructuredGrid stored in a vtkMRMLModel node.

    Attributes:
        obj (ParameterNode): Parameter node for the mapping
        nodePath (str): Sofa node nodePath
    """
    if not isinstance(nodePath, (str, bytes)):
       TypeError("nodePath must be a string")

    if obj._currentMappingMRMLNode is None:
        return

    if obj._currentMappingMRMLNode.GetUnstructuredGrid().GetCellData().GetArray("VonMisesStress") is None:
        # Create a stress array (this is an initialization)
        labelsArray = slicer.util.arrayFromModelCellData(obj._currentMappingMRMLNode, "labels")
        stressVTKArray = vtk.vtkFloatArray()
        stressVTKArray.SetNumberOfValues(labelsArray.shape[0])
        stressVTKArray.SetName("VonMisesStress")
        obj._currentMappingMRMLNode.GetUnstructuredGrid().GetCellData().AddArray(stressVTKArray)

    stressArray = slicer.util.arrayFromModelCellData(obj._currentMappingMRMLNode, "VonMisesStress")
    stressArray[:] = obj._rootNode[nodePath].vonMisesPerElement.array()
    slicer.util.arrayFromModelCellDataModified(obj._currentMappingMRMLNode, "VonMisesStress")

# -----------------------------------------------------------------------------
# Decorator: RunOnce
# -----------------------------------------------------------------------------
def RunOnce(func):
    """
    Decorator that marks a mapping function to be executed only once.

    Attributes:
        runOnce (bool): Flag indicating the function should run only once.
    """
    func.runOnce = True
    return func

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def arrayFromMarkupsROIPoints(roiNode):
    """
    Utility function to return RAS (Right-Anterior-Superior) boundaries from a vtkMRMLMarkupsROINode.

    Args:
        roiNode (vtkMRMLMarkupsROINode): The ROI node from which to extract boundaries.

    Returns:
        list: A list containing [R_min, A_min, S_min, R_max, A_max, S_max].
              Returns [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] if roiNode is None.
    """
    if roiNode is None:
        return [0.0] * 6

    center = [0] * 3
    roiNode.GetCenter(center)
    size = roiNode.GetSize()

    # Calculate min and max RAS bounds
    R_min = center[0] - size[0] / 2
    R_max = center[0] + size[0] / 2
    A_min = center[1] - size[1] / 2
    A_max = center[1] + size[1] / 2
    S_min = center[2] - size[2] / 2
    S_max = center[2] + size[2] / 2

    return [R_min, A_min, S_min, R_max, A_max, S_max]

def arrayVectorFromMarkupsLinePoints(lineNode):
    """
    Utility function to return the vector from a vtkMRMLMarkupsLineNode.

    Args:
        lineNode (vtkMRMLMarkupsLineNode): The line node from which to extract the vector.

    Returns:
        list: A list containing the vector components [x, y, z].
              Returns [0.0, 0.0, 0.0] if lineNode is None.
    """
    if lineNode is None:
        return [0.0] * 3

    # # Calculate direction vector and normalize
    controlPoints = slicer.util.arrayFromMarkupsControlPoints(lineNode)
    vector = controlPoints[1]-controlPoints[0]
    norm = np.linalg.norm(vector)
    if norm == 0:
       return vector  # Return the original vector if norm is zero to avoid division by zero
    return vector / norm
