set(proj Sofa)

# Set dependency list
set(${proj}_DEPENDS
  Boost
  Eigen3
  GLEW
  tinyxml2
  pybind11
 )

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj)

if(${SUPERBUILD_TOPLEVEL_PROJECT}_USE_SYSTEM_${proj})
  message(FATAL_ERROR "Enabling ${SUPERBUILD_TOPLEVEL_PROJECT}_USE_SYSTEM_${proj} is not supported !")
endif()

# Sanity checks
if(DEFINED SOFA_DIR AND NOT EXISTS ${SOFA_DIR})
  message(FATAL_ERROR "SOFA_DIR [${SOFA_DIR}] variable is defined but corresponds to nonexistent directory")
endif()

if(NOT DEFINED ${proj}_DIR AND NOT ${SUPERBUILD_TOPLEVEL_PROJECT}_USE_SYSTEM_${proj})

  # Sanity checks
  set(expected_defined_vars
    ZLIB_INCLUDE_DIR
    ZLIB_LIBRARY
    )
  foreach(var ${expected_defined_vars})
    if(NOT DEFINED ${var})
      message(FATAL_ERROR "Variable ${var} is not defined")
    endif()
  endforeach()

  # Determine Qt version (prefer Qt6, fall back to Qt5)
  if(DEFINED Qt6_DIR)
    set(_sofa_qt_version_major 6)
    set(_sofa_qt_dir ${Qt6_DIR})
    message(STATUS "SlicerSOFA/Sofa: Using Qt6 (${Qt6_DIR})")
  elseif(DEFINED Qt5_DIR)
    set(_sofa_qt_version_major 5)
    set(_sofa_qt_dir ${Qt5_DIR})
    message(STATUS "SlicerSOFA/Sofa: Using Qt5 (${Qt5_DIR})")
  else()
    message(FATAL_ERROR "Neither Qt6_DIR nor Qt5_DIR is defined")
  endif()

  set(SOFA_EXTERNAL_DIRECTORIES)

  include(FetchContent)

  # SofaPython3
  set(plugin_name "SofaPython3")
  set(${plugin_name}_SOURCE_DIR "${CMAKE_BINARY_DIR}/${plugin_name}")
  FetchContent_Populate(${plugin_name}
    SOURCE_DIR     ${${plugin_name}_SOURCE_DIR}
    GIT_REPOSITORY "https://github.com/Slicer/SofaPython3"
    GIT_TAG        "64e02c2b0d610f6752442cb177227598b98dd110" #slicer-v26.06.00-2026-08-24-64e02c2b0
    GIT_PROGRESS   1
    QUIET
    )
  list(APPEND SOFA_EXTERNAL_DIRECTORIES ${${plugin_name}_SOURCE_DIR})
  ExternalProject_Message(${proj} "${plugin_name} sources [OK]")

  # SofaSTLIB
  set(plugin_name "SofaSTLIB")
  set(${plugin_name}_SOURCE_DIR "${CMAKE_BINARY_DIR}/${plugin_name}")
  FetchContent_Populate(${plugin_name}
    SOURCE_DIR     ${${plugin_name}_SOURCE_DIR}
    GIT_REPOSITORY "https://github.com/SofaDefrost/STLIB.git"
    GIT_TAG        "d33de40005b6a002ad1b869b972b214a6389fe67" # release-v26.06
    GIT_PROGRESS   1
    QUIET
    )
  list(APPEND SOFA_EXTERNAL_DIRECTORIES ${${plugin_name}_SOURCE_DIR})
  ExternalProject_Message(${proj} "${plugin_name} sources [OK]")

  # Beam Adapter
  set(plugin_name "BeamAdapter")
  set(${plugin_name}_SOURCE_DIR "${CMAKE_BINARY_DIR}/${plugin_name}")
  FetchContent_Populate(${plugin_name}
    SOURCE_DIR     ${${plugin_name}_SOURCE_DIR}
    GIT_REPOSITORY "https://github.com/sofa-framework/beamadapter.git"
    GIT_TAG        "cac4005bd7c9f266c5c775ab88cebbd7c7fa83c7" # release-v26.06
    GIT_PROGRESS   1
    QUIET
    )
  list(APPEND SOFA_EXTERNAL_DIRECTORIES ${${plugin_name}_SOURCE_DIR})
  ExternalProject_Message(${proj} "${plugin_name} sources [OK]")

  # Registration
  set(plugin_name "Registration")
  set(${plugin_name}_SOURCE_DIR "${CMAKE_BINARY_DIR}/${plugin_name}")
  FetchContent_Populate(${plugin_name}
    SOURCE_DIR     ${${plugin_name}_SOURCE_DIR}
    GIT_REPOSITORY "https://github.com/sofa-framework/registration.git"
    GIT_TAG        "06ab29453a758b675cced5f3121f49016f84f836" # release-v26.06
    GIT_PROGRESS   1
    QUIET
    )
  list(APPEND SOFA_EXTERNAL_DIRECTORIES ${${plugin_name}_SOURCE_DIR})
  ExternalProject_Message(${proj} "${plugin_name} sources [OK]")

  # Cosserat
  set(plugin_name "Cosserat")
  set(${plugin_name}_SOURCE_DIR "${CMAKE_BINARY_DIR}/${plugin_name}")
  FetchContent_Populate(${plugin_name}
    SOURCE_DIR     ${${plugin_name}_SOURCE_DIR}
    GIT_REPOSITORY "https://github.com/SofaDefrost/Cosserat.git"
    GIT_TAG        "31aaf6d9af1b1d4feb7412c4b074aff50c92cec8" # release-v26.06
    GIT_PROGRESS   1
    QUIET
    )
  list(APPEND SOFA_EXTERNAL_DIRECTORIES ${${plugin_name}_SOURCE_DIR})
  ExternalProject_Message(${proj} "${plugin_name} sources [OK]")

  set(EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS)
  if(APPLE)
    list(APPEND EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS
      -DCMAKE_INSTALL_NAME_TOOL:FILEPATH=
      -DCMAKE_MACOSX_RPATH:BOOL=OFF
      )
  endif()
  if(UNIX AND (NOT APPLE))
    list(APPEND EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS
      -DSOFA_ENABLE_LINK_TIME_OPTIMIZATION:BOOL=OFF
    )
  endif()

  set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})
  set(EP_BINARY_DIR ${CMAKE_BINARY_DIR}/${proj}-build)

  ExternalProject_Add(${proj}
    ${${proj}_EP_ARGS}
    GIT_REPOSITORY "https://github.com/Slicer/sofa.git"
    GIT_TAG "389df6376a893351a9786b9ae25fe7635a1ce61c" #slicer-26.06.00-2026-08-24-389df6376
    URL ${SOFA_URL}
    URL_HASH ${SOFA_URL_HASH}
    DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/download
    SOURCE_DIR ${EP_SOURCE_DIR}
    BINARY_DIR ${EP_BINARY_DIR}
    CMAKE_CACHE_ARGS
      # Compiler settings
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      -DCMAKE_C_FLAGS:STRING=${ep_common_c_flags}
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_FLAGS:STRING=${ep_common_cxx_flags}
      -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
      -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
      -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
      # Options
      -DSOFA_BUILD_TESTS:BOOL=OFF
      -DAPPLICATION_RUNSOFA:BOOL=ON
      -DAPPLICATION_SCENECHECKING:BOOL=ON
      -DSOFA_INSTALL_RESOURCES_FILES:BOOL=OFF
      -DSOFA_USE_DEPENDENCY_PACK:BOOL=OFF
      # Install directories
      # NA
      # More options
      -DSofaSTLIB_ENABLED:BOOL=ON
      -DSofaBeamAdapter_ENABLED:BOOL=ON
      -DRegistration_ENABLED:BOOL=ON
      -DCosserat_ENABLED:BOOL=ON
      -DLIBRARY_SOFA_GUI:BOOL=ON
      -DLIBRARY_SOFA_GUI_COMMON:BOOL=ON
      -DMODULE_SOFA_GUI_COMPONENT:BOOL=ON
      -DPLUGIN_SOFA_GUI_BATCH:BOOL=ON
      -DSOFA_ROOT:PATH=${EP_SOURCE_DIR}
      -DSOFA_WITH_OPENGL:BOOL=ON
      # Dependencies
      -DGLEW_DIR:PATH=${GLEW_DIR}
      -DBoost_NO_BOOST_CMAKE:BOOL=FALSE # Support finding Boost as config-file package
      -DBOOST_ROOT:PATH=${Boost_DIR}
      -DEIGEN3_INCLUDE_DIR:PATH=${Eigen3_DIR}/include/eigen3
      -DQt${_sofa_qt_version_major}_DIR:PATH=${_sofa_qt_dir}
      -DTinyXML2_INCLUDE_DIR:PATH=${tinyxml2_INCLUDE_DIR}
      -DTinyXML2_LIBRARY:PATH=${tinyxml2_LIBRARY}
      # Required on Windows since SOFA 26.06 (FindTinyXML2 insists on the DLL)
      -DTinyXML2_DLL:FILEPATH=${tinyxml2_DLL}
      -DZLIB_INCLUDE_DIR:PATH=${ZLIB_INCLUDE_DIR}
      -DZLIB_LIBRARY:PATH=${ZLIB_LIBRARY}
      -DSOFA_EXTERNAL_DIRECTORIES:STRING=${SOFA_EXTERNAL_DIRECTORIES}
      # SofaPython3
      -DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}
      -DPython3_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}
      -DPython_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE}
      -DPYTHON_LIBRARIES:FILEPATH=${PYTHON_LIBRARY}
      -DPYTHON_INCLUDE_DIRS:PATH=${PYTHON_INCLUDE_DIR}
      -Dpybind11_DIR:PATH=${pybind11_DIR}/share/cmake/pybind11
      ${EXTERNAL_PROJECT_OPTIONAL_CMAKE_CACHE_ARGS}
    DEPENDS
      ${${proj}_DEPENDS}
    INSTALL_COMMAND ""
    )
  set(${proj}_DIR ${EP_BINARY_DIR})

  #-----------------------------------------------------------------------------
  # Launcher setting specific to build tree

  # library paths
  set(${proj}_LIBRARY_PATHS_LAUNCHER_BUILD
    ${${proj}_DIR}/lib
    # ${CMAKE_BINARY_DIR}/${Slicer_THIRDPARTY_BIN_DIR}
    # ${CMAKE_BINARY_DIR}/${Slicer_THIRDPARTY_BIN_DIR}/<CMAKE_CFG_INTDIR>
    )
  mark_as_superbuild(
    VARS ${proj}_LIBRARY_PATHS_LAUNCHER_BUILD
    LABELS "LIBRARY_PATHS_LAUNCHER_BUILD"
    )

  # python paths
  set(${proj}_PYTHONPATH_LAUNCHER_BUILD
    ${${proj}_DIR}/lib/python3/site-packages
    # ${CMAKE_BINARY_DIR}/${Slicer_THIRDPARTY_BIN_DIR}
    # ${CMAKE_BINARY_DIR}/${Slicer_THIRDPARTY_BIN_DIR}/<CMAKE_CFG_INTDIR>
    )
  mark_as_superbuild(
    VARS ${proj}_PYTHONPATH_LAUNCHER_BUILD
    LABELS "PYTHONPATH_LAUNCHER_BUILD"
    )

else()
  ExternalProject_Add_Empty(${proj} DEPENDS ${${proj}_DEPENDS})
endif()

mark_as_superbuild(${proj}_DIR:PATH)
