#-----------------------------------------------------------------------------
set(MODULE_NAME SlicerSofa)


#-----------------------------------------------------------------------------
configure_file(SofaSimulation/SlicerSofaRootSource.py.in ${CMAKE_CURRENT_SOURCE_DIR}/SofaSimulation/SlicerSofaRoot.py @ONLY)
configure_file(SofaSimulation/SlicerSofaRootBuild.py.in
  ${CMAKE_BINARY_DIR}/${Slicer_QTSCRIPTEDMODULES_LIB_DIR}/SofaSimulation/SlicerSofaRoot.py)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  SofaSimulation/__init__.py
  SofaSimulation/SimulationController.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
