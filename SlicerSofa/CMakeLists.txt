#-----------------------------------------------------------------------------
set(MODULE_NAME SlicerSofa)


#-----------------------------------------------------------------------------
configure_file(SofaEnvironment/__init__source.py.in ${CMAKE_CURRENT_SOURCE_DIR}/SofaEnvironment/__init__.py @ONLY)
configure_file(SofaEnvironment/__init__build.py.in ${CMAKE_BINARY_DIR}/${Slicer_QTSCRIPTEDMODULES_LIB_DIR}/SofaEnvironment/__init__.py)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  SofaEnvironment/__init__.py
  )

set(MODULE_PYTHON_RESOURCES
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
