
#-----------------------------------------------------------------------------
# Known defect: the displacement grid is allocated with reversed axes but
# receives probeGrid's origin/spacing unpermuted, so its physical extent is
# wrong for any non-cubic grid.  Registered as WILL_FAIL so the defect is
# tracked in CI; when it is fixed this test goes green and the build turns red,
# prompting removal of the property.
slicer_add_python_unittest(SCRIPT SparseGridSimulationGeometryTest.py)
set_tests_properties(py_SparseGridSimulationGeometryTest PROPERTIES WILL_FAIL TRUE)
