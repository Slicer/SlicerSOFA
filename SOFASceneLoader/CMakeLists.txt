#-----------------------------------------------------------------------------
set(MODULE_NAME SOFASceneLoader)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/UI/${MODULE_NAME}.ui
  Resources/${MODULE_NAME}Module.qrc
  Resources/Scenes/SOFALogo/SOFALogo.py
  Resources/Scenes/SOFALogo/mesh/A.vtk
  Resources/Scenes/SOFALogo/mesh/ACoarse.obj
  Resources/Scenes/SOFALogo/mesh/AVisu.obj
  Resources/Scenes/SOFALogo/mesh/FSkel.obj
  Resources/Scenes/SOFALogo/mesh/FVisu.obj
  Resources/Scenes/SOFALogo/mesh/Logo.vtk
  Resources/Scenes/SOFALogo/mesh/LogoColli.sph
  Resources/Scenes/SOFALogo/mesh/LogoVisu.obj
  Resources/Scenes/SOFALogo/mesh/O.obj
  Resources/Scenes/SOFALogo/mesh/S.vtk
  Resources/Scenes/SOFALogo/mesh/SVisu.obj
  Resources/Scenes/SOFALogo/textures/SOFA_LOGO.png
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES} ${MODULE_PYTHON_SIMULATIONS}
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
