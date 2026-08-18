"""Known-defect test: displacement grid geometry under non-cubic dimensions.

Registered in CMakeLists.txt with WILL_FAIL TRUE.  It documents a real defect
that predates the TransformToParent change:

  _createGridTransformPipeline allocates the displacement grid with its axes
  reversed -- SetDimensions(z, y, x) -- to match a numpy view shaped
  (x, y, z, 3), because VTK image point ordering runs fastest over the first
  dimension.  But _updateProbingImage then copies probeGrid's origin and
  spacing onto it *without* the matching permutation.  For a cubic grid the
  permutation is the identity and the two agree; for any non-cubic grid the
  displacement field is applied over the wrong physical extent.

The shipped defaults are cubic (20,20,20; reset to 10,10,10), which masks the
defect entirely -- but the three UI spinboxes are independent, so any user who
narrows one axis silently gets a misplaced deformation.

When the defect is fixed, this test starts passing and WILL_FAIL makes the
build go red.  That is the signal to drop the WILL_FAIL property and fold the
assertion into SparseGridSimulationTest.
"""

import slicer
from SparseGridSimulation import GridDimensions, SparseGridSimulationTest

NON_CUBIC = GridDimensions(x=6, y=10, z=14)

slicer.mrmlScene.Clear()

harness = SparseGridSimulationTest()
logic = harness._buildLogic(dimensions=NON_CUBIC)

logic.startSimulation()
logic.simulationStep()

displacementGrid = logic.getParameterNode().gridTransformNode \
                        .GetTransformToParent().GetDisplacementGrid()

probeBounds = logic.probeGrid.GetBounds()
gridBounds = displacementGrid.GetBounds()

logic.stopSimulation()

print(f"grid dimensions      : x={NON_CUBIC.x} y={NON_CUBIC.y} z={NON_CUBIC.z}")
print(f"probe grid bounds    : {['%.4f' % v for v in probeBounds]}")
print(f"displacement bounds  : {['%.4f' % v for v in gridBounds]}")

mismatched = [
    (axis, expected, actual)
    for axis, (expected, actual) in enumerate(zip(probeBounds, gridBounds))
    if abs(expected - actual) > 1e-6
]

if mismatched:
    for axis, expected, actual in mismatched:
        print(f"MISMATCH bound {axis}: probe {expected:.4f} != displacement {actual:.4f}")
    raise AssertionError(
        f"displacement grid does not cover the probed region on "
        f"{len(mismatched)} of 6 bounds")

print("displacement grid covers the probed region")
